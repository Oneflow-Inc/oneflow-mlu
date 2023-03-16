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
#include "oneflow/cambricon/cnnl/cnnl_types.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"

namespace oneflow {

template<int Nd, typename T>
class MluMaxPoolKernel final : public user_op::OpKernel {
 public:
  MluMaxPoolKernel() = default;
  ~MluMaxPoolKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
    const std::vector<int32_t>& dilation = ctx->Attr<std::vector<int32_t>>("dilation");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    CnnlTensorDescriptor x_desc, y_desc, indice_desc;
    // layouts of x, y, indice are default set to CNNL_LAYOUT_ARRAY,
    // cnnlPoolingForward_v2 requires NCHW or NHWC
    x_desc.set(x, CNNL_LAYOUT_NCHW);
    y_desc.set(y, CNNL_LAYOUT_NCHW);
    indice_desc.set(indice, CNNL_LAYOUT_NCHW);


    cnnlPoolingDescriptor_t pooling_desc = nullptr;
    OF_CNNL_CHECK(cnnlCreatePoolingDescriptor(&pooling_desc));
    OF_CNNL_CHECK(cnnlSetPooling2dDescriptor_v2(
        /* pooling_desc       */ pooling_desc,
        /* mode               */ cnnlPoolingMode_t::CNNL_POOLING_MAX,
        /* maxpooling_nan_opt */ CNNL_NOT_PROPAGATE_NAN,
        /* window_height      */ static_cast<int>(kernel_size[0]),
        /* window_width       */ static_cast<int>(kernel_size[1]),
        /* top_padding        */ static_cast<int>(padding[0]),
        /* bottom_padding     */ static_cast<int>(padding[1]),
        /* left_padding       */ static_cast<int>(padding[2]),
        /* right_padding      */ static_cast<int>(padding[3]),
        /* vertical_stride    */ static_cast<int>(stride[0]),
        /* horizon_stride     */ static_cast<int>(stride[1]),
        /* vertical_dilation  */ static_cast<int>(dilation[0]),
        /* horizon_dilation   */ static_cast<int>(dilation[1]),
        /* ceil_mode          */ ceil_mode));

    size_t extra_input_size = 0;
    OF_CNNL_CHECK(cnnlGetPoolingExtraInputSize(
        /* handle           */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* mode             */ cnnlPoolingMode_t::CNNL_POOLING_MAX,
        /* out_w_size       */ y->shape_view().At(y->shape_view().NumAxes() - 1),
        /* out_h_size       */ y->shape_view().At(y->shape_view().NumAxes() - 2),
        /* extra_input_size */ &extra_input_size));
    CnnlWorkspace extra_input_workspace(ctx->stream()->As<ep::MluStream>(), extra_input_size);
    const void* extra_input = extra_input_workspace.dptr();
    OF_CNNL_CHECK(cnnlInitPoolingExtraInput(
        /* handle           */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* pooling_desc     */ pooling_desc,
        /* x_desc           */ x_desc.desc(),
        /* y_desc           */ y_desc.desc(),
        /* extra_host_input */ &extra_input));

    size_t pooling_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetPoolingWorkspaceSize(
        /* handle           */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* mode             */ cnnlPoolingMode_t::CNNL_POOLING_MAX,
        /* out_w_size       */ y->shape_view().At(y->shape_view().NumAxes() - 1),
        /* out_h_size       */ y->shape_view().At(y->shape_view().NumAxes() - 2),
        /* workspace_size   */ &pooling_workspace_size));
    CnnlWorkspace pooling_workspace(ctx->stream()->As<ep::MluStream>(), pooling_workspace_size);
    void* pooling_workspace_ptr = pooling_workspace.dptr();

    OF_CNNL_CHECK(cnnlPoolingForwardWithIndex(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* pooling_desc   */ pooling_desc,
        /* alpha          */ nullptr,
        /* x_desc         */ x_desc.desc(),
        /* x              */ x->dptr(),
        /* beta           */ nullptr,
        /* y_desc         */ y_desc.desc(),
        /* y              */ y->mut_dptr(),
        /* index_desc     */ indice_desc.desc(),
        /* index          */ indice->mut_dptr(),
        /* workspace      */ pooling_workspace_ptr,
        /* workspace_size */ pooling_workspace_size));

    OF_CNNL_CHECK(cnnlDestroyPoolingDescriptor(pooling_desc));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MAX_POOL_MLU_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("max_pool_2d")                                 \
      .SetCreateFn<MluMaxPoolKernel<2, dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MAX_POOL_MLU_KERNEL(float)
REGISTER_MAX_POOL_MLU_KERNEL(float16)

}  // namespace oneflow
