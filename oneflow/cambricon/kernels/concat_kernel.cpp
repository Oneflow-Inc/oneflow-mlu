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

class ConcatKernel final : public user_op::OpKernel {
 public:
  ConcatKernel() = default;
  ~ConcatKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto& inputs = ctx->inputs();
    if (out_tensor->shape_view().elem_cnt() == 0 || inputs.empty()) { return; }
    const int axis = static_cast<int>(ctx->Attr<int64_t>("axis"));
    CnnlTensorDescriptor out_desc(out_tensor);
    // prepare inputs desc
    std::vector<CnnlTensorDescriptor> inputs_desc;
    inputs_desc.reserve(inputs.size());
    std::vector<cnnlTensorDescriptor_t> inputs_desc_ptr;
    inputs_desc_ptr.reserve(inputs.size());
    std::vector<const void*> inputs_ptr;
    inputs_ptr.reserve(inputs.size());
    for (const auto& in_arg_pair : ctx->inputs()) {
      const user_op::Tensor* in_tensor =
          ctx->Tensor4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
      if (in_tensor->shape_view().elem_cnt() == 0) { continue; }
      inputs_desc.emplace_back();
      inputs_desc.back().set(in_tensor);
      inputs_desc_ptr.emplace_back(inputs_desc.back().desc());
      inputs_ptr.emplace_back(in_tensor->dptr());
    }
    // launch kernel
    OF_CNNL_CHECK(cnnlConcat(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                             static_cast<int>(inputs_desc_ptr.size()), axis, inputs_desc_ptr.data(),
                             inputs_ptr.data(), nullptr, 0, out_desc.desc(),
                             out_tensor->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONCAT_USER_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("concat").SetCreateFn<ConcatKernel>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                          \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_CONCAT_USER_KERNEL(float)
REGISTER_CONCAT_USER_KERNEL(float16)
REGISTER_CONCAT_USER_KERNEL(int32_t)
REGISTER_CONCAT_USER_KERNEL(int64_t)

#undef REGISTER_VAR_CPU_KERNEL

}  // namespace
}  // namespace oneflow
