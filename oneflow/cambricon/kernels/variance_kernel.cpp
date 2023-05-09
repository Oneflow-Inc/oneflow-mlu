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
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const bool unbiased = ctx->Attr<bool>("unbiased");
    const std::vector<int32_t>& dim = ctx->Attr<std::vector<int32_t>>("dim");
    CHECK_OR_THROW(dim.size() == 1) << "Var only support int dim on MLU.";
    CnnlTensorDescriptor input_desc(input), output_desc(output);
    OF_CNNL_CHECK(cnnlVarForward(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                 static_cast<int>(dim[0]), unbiased, input_desc.desc(),
                                 input->dptr(), output_desc.desc(), output->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

inline auto SimpleMatcher() {
  return (user_op::HobDeviceType() == DeviceType::kMLU)
         && ((user_op::HobDataType("input", 0) == kFloat)
             || (user_op::HobDataType("input", 0) == kFloat16));
}

REGISTER_USER_KERNEL("var").SetCreateFn<VarKernel>().SetIsMatchedHob(SimpleMatcher());

#undef REGISTER_VAR_CPU_KERNEL

}  // namespace
}  // namespace oneflow
