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
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/kernels/convert_memory_format_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

class UpsampleNearest2DMLUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearest2DMLUKernel() = default;
  ~UpsampleNearest2DMLUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    cnnlDataType_t dtype = ConvertToCnnlDataType(x_tensor->data_type());
    // prepare workspace
    CnnlWorkspace x_workspace(ctx->stream()->As<ep::MluStream>(),
                              x_tensor->shape_view().elem_cnt() * sizeof(x_tensor->data_type()));
    CnnlWorkspace y_workspace(ctx->stream()->As<ep::MluStream>(),
                              y_tensor->shape_view().elem_cnt() * sizeof(y_tensor->data_type()));
    // convert layout of x from NCHW to NHWC
    mlu::ConvertMemoryFormat(ctx->stream(), x_tensor->shape_view(), x_tensor->data_type(),
                             x_tensor->dptr(), x_workspace.dptr(), MemoryFormat::kNCHW,
                             MemoryFormat::kNHWC);
    // prepare tensor desc
    auto x_shape = mlu::ComputeShapeNchwToNhwc(Shape(x_tensor->shape_view()));
    auto y_shape = mlu::ComputeShapeNchwToNhwc(Shape(y_tensor->shape_view()));
    CnnlTensorDescriptor x_desc, y_desc;
    x_desc.set(x_tensor->shape_view().NumAxes(), x_shape.data(), dtype, CNNL_LAYOUT_NHWC);
    y_desc.set(y_tensor->shape_view().NumAxes(), y_shape.data(), dtype, CNNL_LAYOUT_NHWC);
    // launch kernel
    OF_CNNL_CHECK(cnnlInterp(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), false, false,
                             CNNL_INTERP_NEAREST, x_desc.desc(), x_workspace.dptr(), y_desc.desc(),
                             y_workspace.dptr()));
    // convert layout of y from NHWC to NCHW
    mlu::ConvertMemoryFormat(ctx->stream(), y_shape, y_tensor->data_type(), y_workspace.dptr(),
                             y_tensor->mut_dptr(), MemoryFormat::kNHWC, MemoryFormat::kNCHW);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

inline auto SimpleMatcher() {
  return (user_op::HobDeviceType() == DeviceType::kMLU)
         && ((user_op::HobDataType("x", 0) == kFloat)
             || (user_op::HobDataType("x", 0) == kFloat16));
}

REGISTER_USER_KERNEL("upsample_nearest_2d")
    .SetCreateFn<UpsampleNearest2DMLUKernel>()
    .SetIsMatchedHob(SimpleMatcher());

#undef REGISTER_VAR_CPU_KERNEL

}  // namespace
}  // namespace oneflow
