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
#include "oneflow/cambricon/kernels/convert_memory_format_util.h"

namespace oneflow {

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx, const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

template<typename T>
class AdaptiveAvgPool2DKernel final : public user_op::OpKernel {
 public:
  AdaptiveAvgPool2DKernel() = default;
  ~AdaptiveAvgPool2DKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    // const T* in_ptr = in_tensor->dptr<T>();
    // T* out_ptr = out_tensor->mut_dptr<T>();

    const void* temp_in_ptr = in_tensor->dptr();
    void* temp_out_ptr = out_tensor->mut_dptr();

    cnnlTensorLayout_t layout =
        (data_format == "channels_last") ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NCHW;
    CnnlTensorDescriptor in_desc(in_tensor), out_desc(out_tensor);
    cnnlDataType_t dtype = ConvertToCnnlDataType(in_tensor->data_type());

    CnnlWorkspace tmp_in_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), 0);
    CnnlWorkspace tmp_out_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), 0);

    auto in_shape = Shape(in_tensor->shape_view());
    auto out_shape = Shape(out_tensor->shape_view());

    if (layout != CNNL_LAYOUT_NHWC) {
      size_t tmp_in_workspace_size =
          in_tensor->shape_view().elem_cnt() * sizeof(in_tensor->data_type());
      size_t tmp_out_workspace_size =
          out_tensor->shape_view().elem_cnt() * sizeof(out_tensor->data_type());
      tmp_in_cnnl_workspace.resize(tmp_in_workspace_size);
      tmp_out_cnnl_workspace.resize(tmp_out_workspace_size);
      mlu::ConvertMemoryFormat(ctx->stream(), in_tensor->shape_view(), in_tensor->data_type(),
                               in_tensor->dptr(), tmp_in_cnnl_workspace.dptr(), MemoryFormat::kNCHW,
                               MemoryFormat::kNHWC);
      temp_in_ptr = tmp_in_cnnl_workspace.dptr();
      temp_out_ptr = tmp_out_cnnl_workspace.dptr();
      in_shape = mlu::ComputeShapeNchwToNhwc(in_shape);
      out_shape = mlu::ComputeShapeNchwToNhwc(out_shape);
    }

    in_desc.set(in_shape.NumAxes(), in_shape.data(), dtype, CNNL_LAYOUT_NHWC);
    out_desc.set(out_shape.NumAxes(), out_shape.data(), dtype, CNNL_LAYOUT_NHWC);

    size_t _adaptive_avg_pool2d_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetAdaptivePoolingForwardWorkspaceSize(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), in_desc.desc(),
        CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, out_desc.desc(),
        &_adaptive_avg_pool2d_workspace_size));
    CnnlWorkspace adaptive2d_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                            _adaptive_avg_pool2d_workspace_size);
    void* _adaptive_avg_pool2d_workspace = adaptive2d_cnnl_workspace.dptr();
    OF_CNNL_CHECK(cnnlAdaptivePoolingForward_v2(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), in_desc.desc(), temp_in_ptr,
        CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, _adaptive_avg_pool2d_workspace,
        _adaptive_avg_pool2d_workspace_size, out_desc.desc(), temp_out_ptr, NULL, NULL));

    if (layout != CNNL_LAYOUT_NHWC) {
      mlu::ConvertMemoryFormat(ctx->stream(), out_shape, out_tensor->data_type(), temp_out_ptr,
                               out_tensor->mut_dptr(), MemoryFormat::kNHWC, MemoryFormat::kNCHW);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(dtype)                 \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d")                         \
      .SetCreateFn<AdaptiveAvgPool2DKernel<dtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(float)
REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(float16)

template<typename T>
class AdaptiveAvgPool2DGradKernel final : public user_op::OpKernel {
 public:
  AdaptiveAvgPool2DGradKernel() = default;
  ~AdaptiveAvgPool2DGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);

    CHECK_EQ(x_tensor->shape_view().NumAxes(), 4);

    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_last") {
      ComputeNHWC(ctx, dy_tensor, dx_tensor);
      return;
    }

    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();

    size_t tmp_dy_workspace_size =
        dy_tensor->shape_view().elem_cnt() * sizeof(dy_tensor->data_type());
    CnnlWorkspace tmp_dy_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_dy_workspace_size);
    void* tmp_dy_ptr = tmp_dy_cnnl_workspace.dptr();

    std::vector<int64_t> dy_shapevec(dy_tensor->shape_view().begin(),
                                     dy_tensor->shape_view().end());
    std::vector<int> dy_permutation = {0, 2, 3, 1};

    const auto& dy_transpose = NewPermutePrimitive(ctx, dy_tensor->shape_view().NumAxes());
    CHECK(dy_transpose);

    dy_transpose->Launch(ctx->stream(), dy_tensor->data_type(), dy_tensor->shape_view().NumAxes(),
                         dy_shapevec.data(), dy_ptr, dy_permutation.data(), tmp_dy_ptr);

    const std::vector<int> tmp_dy_dims = {static_cast<int>(dy_tensor->shape_view().At(0)),
                                          static_cast<int>(dy_tensor->shape_view().At(2)),
                                          static_cast<int>(dy_tensor->shape_view().At(3)),
                                          static_cast<int>(dy_tensor->shape_view().At(1))};
    const std::vector<int> tmp_dx_dims = {static_cast<int>(dx_tensor->shape_view().At(0)),
                                          static_cast<int>(dx_tensor->shape_view().At(2)),
                                          static_cast<int>(dx_tensor->shape_view().At(3)),
                                          static_cast<int>(dx_tensor->shape_view().At(1))};
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    auto dtype = ConvertToCnnlDataType(dy_tensor->data_type());

    CnnlTensorDescriptor dy_desc, dx_desc;
    dy_desc.set(4, tmp_dy_dims.data(), dtype, layout);
    dx_desc.set(4, tmp_dx_dims.data(), dtype, layout);

    size_t tmp_dx_workspace_size =
        dx_tensor->shape_view().elem_cnt() * sizeof(dy_tensor->data_type());
    CnnlWorkspace tmp_dx_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_dx_workspace_size);
    void* tmp_dx_ptr = tmp_dx_cnnl_workspace.dptr();

    OF_CNNL_CHECK(cnnlAdaptivePoolingBackward(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), dy_desc.desc(), tmp_dy_ptr, nullptr,
        nullptr, CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, dx_desc.desc(), tmp_dx_ptr));

    std::vector<int64_t> dx_shapevec({dx_tensor->shape_view().At(0), dx_tensor->shape_view().At(2),
                                      dx_tensor->shape_view().At(3),
                                      dx_tensor->shape_view().At(1)});
    const std::vector<int> dx_permutation = {0, 3, 1, 2};
    const auto& dx_transpose = NewPermutePrimitive(ctx, dx_tensor->shape_view().NumAxes());
    CHECK(dx_transpose);
    dx_transpose->Launch(ctx->stream(), dx_tensor->data_type(), dx_tensor->shape_view().NumAxes(),
                         dx_shapevec.data(), tmp_dx_ptr, dx_permutation.data(), dx_ptr);
  }

  void ComputeNHWC(user_op::KernelComputeContext* ctx, const user_op::Tensor* dy_tensor,
                   user_op::Tensor* dx_tensor) const {
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    auto dtype = ConvertToCnnlDataType(dy_tensor->data_type());
    CnnlTensorDescriptor dy_desc, dx_desc;
    dy_desc.set(dy_tensor, layout, dtype);
    dx_desc.set(dx_tensor, layout, dtype);
    OF_CNNL_CHECK(cnnlAdaptivePoolingBackward(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                              dy_desc.desc(), dy_tensor->dptr(), nullptr, nullptr,
                                              CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                              dx_desc.desc(), dx_tensor->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_AVGPOOL2D_GRAD_MLU_KERNEL(dtype)            \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d_grad")                    \
      .SetCreateFn<AdaptiveAvgPool2DGradKernel<dtype>>()              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));
REGISTER_ADAPTIVE_AVGPOOL2D_GRAD_MLU_KERNEL(float)
REGISTER_ADAPTIVE_AVGPOOL2D_GRAD_MLU_KERNEL(float16)

}  // namespace oneflow
