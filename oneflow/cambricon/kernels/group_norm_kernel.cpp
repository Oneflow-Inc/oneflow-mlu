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
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

class VarKernel final : public user_op::OpKernel {
 public:
  VarKernel() = default;
  ~VarKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float epsilon = static_cast<float>(ctx->Attr<double>("epsilon"));
    const int32_t num_groups = ctx->Attr<int32_t>("num_groups");

    // prepare x and y desc
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    cnnlDataType_t dtype = ConvertToCnnlDataType(x->data_type());
    CnnlTensorDescriptor x_desc, y_desc;
    if (data_format == "channels_first") {
      x_desc.set(x->shape_view().NumAxes(), x->shape_view().data(), dtype, CNNL_LAYOUT_NCHW);
      y_desc.set(y->shape_view().NumAxes(), y->shape_view().data(), dtype, CNNL_LAYOUT_NCHW);
    } else if (data_format == "channels_last") {
      x_desc.set(x->shape_view().NumAxes(), x->shape_view().data(), dtype, CNNL_LAYOUT_NHWC);
      y_desc.set(y->shape_view().NumAxes(), y->shape_view().data(), dtype, CNNL_LAYOUT_NHWC);
    } else {
      UNIMPLEMENTED();
    }

    // prepare gamma and beta desc
    cnnlTensorDescriptor_t gamma_desc_ptr = nullptr;
    CnnlTensorDescriptor gamma_desc;
    const void* gamma_ptr = nullptr;
    const void* beta_ptr = nullptr;
    if (ctx->has_input("gamma", 0) && ctx->has_input("beta", 0)) {
      const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      gamma_desc.set(gamma);
      gamma_desc_ptr = gamma_desc.desc();
      gamma_ptr = gamma->dptr();
      const user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
      beta_ptr = beta->dptr();
    }

    OF_CNNL_CHECK(cnnlGroupNormForward(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
      epsilon, num_groups, x_desc.desc(), x->dptr(), gamma_desc_ptr, gamma_ptr, beta_ptr,
      y_desc.desc(), y->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

inline auto SimpleMatcher() {
  return (user_op::HobDeviceType() == DeviceType::kMLU)
         && ((user_op::HobDataType("x", 0) == kFloat)
             || (user_op::HobDataType("x", 0) == kFloat16));
}

REGISTER_USER_KERNEL("group_norm").SetCreateFn<VarKernel>().SetIsMatchedHob(SimpleMatcher());

#undef REGISTER_VAR_CPU_KERNEL

}  // namespace
}  // namespace oneflow
