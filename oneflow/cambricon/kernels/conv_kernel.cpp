/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "cnnl.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/include/primitive/permute.h"

namespace oneflow {

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx, const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

template<typename T>
class Conv2DKernel final : public user_op::OpKernel {
 public:
  Conv2DKernel() = default;
  ~Conv2DKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    // transpose input (nchw->nhwc)
    size_t tmp_in_workspace_size =
        in->shape_view().elem_cnt() * sizeof(in->data_type());
    CnnlWorkspace tmp_in_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_in_workspace_size);
    void* tmp_in_ptr = tmp_in_cnnl_workspace.dptr();

    std::vector<int64_t> in_shapevec({in->shape_view().At(0), in->shape_view().At(1),
                                      in->shape_view().At(2),
                                      in->shape_view().At(3)});
    auto transpose = NewPermutePrimitive(ctx, in->shape_view().NumAxes());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in->data_type(), in->shape_view().NumAxes(),
                      in_shapevec.data(), in->dptr(), std::vector<int>({0, 2, 3, 1}).data(),
                      tmp_in_ptr);
    // transpose weight (nchw->nhwc), n=out_channels, c=in_channels, h=kernel_h, w=kernel_w 
    size_t tmp_weight_workspace_size =
        weight->shape_view().elem_cnt() * sizeof(weight->data_type());
    CnnlWorkspace tmp_weight_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_weight_workspace_size);
    void* tmp_weight_ptr = tmp_weight_cnnl_workspace.dptr();

    std::vector<int64_t> weight_shapevec({weight->shape_view().At(0), weight->shape_view().At(1),
                                      weight->shape_view().At(2),
                                      weight->shape_view().At(3)});
    transpose = NewPermutePrimitive(ctx, weight->shape_view().NumAxes());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), weight->data_type(), weight->shape_view().NumAxes(),
                      weight_shapevec.data(), weight->dptr(), std::vector<int>({0, 2, 3, 1}).data(),
                      tmp_weight_ptr);
    
    cnnlTensorDescriptor_t in_desc = nullptr, weight_desc=nullptr, bias_desc = nullptr, out_desc = nullptr;
    const int in_dims[4] = {static_cast<int>(in->shape_view().At(0)),
                            static_cast<int>(in->shape_view().At(2)),
                            static_cast<int>(in->shape_view().At(3)),
                            static_cast<int>(in->shape_view().At(1))};
     const int weight_dims[4] = {static_cast<int>(weight->shape_view().At(0)),
                            static_cast<int>(weight->shape_view().At(2)),
                            static_cast<int>(weight->shape_view().At(3)),
                            static_cast<int>(weight->shape_view().At(1))};
    
    const int out_dims[4] = {static_cast<int>(out->shape_view().At(0)),
                             static_cast<int>(out->shape_view().At(2)),
                             static_cast<int>(out->shape_view().At(3)),
                             static_cast<int>(out->shape_view().At(1))};
    auto dtype = ConvertToCnnlDataType(in->data_type());
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    OF_CNNL_CHECK(cnnlCreateTensorDescriptor(&in_desc));
    OF_CNNL_CHECK(cnnlCreateTensorDescriptor(&weight_desc));
    OF_CNNL_CHECK(cnnlCreateTensorDescriptor(&bias_desc));
    OF_CNNL_CHECK(cnnlCreateTensorDescriptor(&out_desc));
    OF_CNNL_CHECK(cnnlSetTensorDescriptor(in_desc, layout, dtype, 4, in_dims));
    OF_CNNL_CHECK(cnnlSetTensorDescriptor(weight_desc, layout, dtype, 4, weight_dims));
    OF_CNNL_CHECK(cnnlSetTensorDescriptor(out_desc, layout, dtype, 4, out_dims));
    if (bias != nullptr){
        const int bias_dims[1] = {static_cast<int>(bias->shape_view().At(0))};
        OF_CNNL_CHECK(cnnlSetTensorDescriptor(bias_desc, layout, dtype, 1, bias_dims));
    }

    cnnlConvolutionDescriptor_t conv_desc = nullptr;
    OF_CNNL_CHECK(cnnlCreateConvolutionDescriptor(&conv_desc));
    
    const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding_before");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const int& groups = ctx->Attr<int32_t>("groups");
    int sh = 0, sw=0, dh=0, dw=0;
    sh = strides[0];
    sw = strides[0];
    if (strides.size() > 0) { sw = strides[1]; }
    dh = dilation_rate[0];
    dw = dilation_rate[0];
    if (dilation_rate.size() > 0) { dw = dilation_rate[1]; }

    int pad_t = padding[0];
    int pad_b = padding[0];
    int pad_l = padding[1];
    int pad_r = padding[1];

    int pad[4] = {pad_t, pad_b, pad_l, pad_r};
    int stride[2] = {sh, sw};
    int dilation[2] = {dh, dw};
    int group_count = groups;
    OF_CNNL_CHECK(cnnlSetConvolutionDescriptor(conv_desc, 4, pad, stride, dilation, group_count, CNNL_DTYPE_FLOAT));
    cnnlConvolutionForwardAlgo_t algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
    OF_CNNL_CHECK(cnnlGetConvolutionForwardAlgorithm(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), conv_desc, in_desc, weight_desc, out_desc, CNNL_CONVOLUTION_FWD_FASTEST, &algo));

    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), in_desc, weight_desc, out_desc,
        bias_desc, conv_desc, algo, &(workspace_size)));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);
    void* conv_workspace_ptr = workspace.dptr();

    size_t tmp_out_workspace_size =
        out->shape_view().elem_cnt() * sizeof(out->data_type());
    CnnlWorkspace tmp_out_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                         tmp_out_workspace_size);
    void* tmp_out_ptr = tmp_out_cnnl_workspace.dptr();

    if (bias != nullptr){
        OF_CNNL_CHECK(cnnlConvolutionForward(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), conv_desc, algo, nullptr, in_desc, tmp_in_ptr,
        weight_desc, tmp_weight_ptr, bias_desc, bias->dptr(),
        &conv_workspace_ptr, workspace_size, nullptr, out_desc, tmp_out_ptr));
    }
    else{
        OF_CNNL_CHECK(cnnlConvolutionForward(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), conv_desc, algo, nullptr, in_desc, tmp_in_ptr,
        weight_desc, tmp_weight_ptr, nullptr, nullptr,
        &conv_workspace_ptr, workspace_size, nullptr, out_desc, tmp_out_ptr));
    }
    std::vector<int64_t> out_shapevec(
        {out->shape_view().At(0), out->shape_view().At(2),
         out->shape_view().At(3), out->shape_view().At(1)});
    transpose = NewPermutePrimitive(ctx, out->shape_view().NumAxes());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), out->data_type(), out->shape_view().NumAxes(),
                      out_shapevec.data(), tmp_out_ptr, std::vector<int>({0, 3, 1, 2}).data(),
                      out->mut_dptr());
    OF_CNNL_CHECK(cnnlDestroyTensorDescriptor(in_desc));
    OF_CNNL_CHECK(cnnlDestroyTensorDescriptor(weight_desc));
    OF_CNNL_CHECK(cnnlDestroyTensorDescriptor(bias_desc));
    OF_CNNL_CHECK(cnnlDestroyTensorDescriptor(out_desc));
    OF_CNNL_CHECK(cnnlDestroyConvolutionDescriptor(conv_desc));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONV2D_MLU_KERNEL(dtype)                 \
  REGISTER_USER_KERNEL("conv2d")                         \
      .SetCreateFn<Conv2DKernel<dtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_CONV2D_MLU_KERNEL(float)
REGISTER_CONV2D_MLU_KERNEL(float16)

}  // namespace oneflow
