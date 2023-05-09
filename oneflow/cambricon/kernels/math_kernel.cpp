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
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

enum class MathOpType {
  cos,
  exp,
  sin,
  sqrt,
};

using FuncType = cnnlStatus_t (*)(cnnlHandle_t, const cnnlComputationPreference_t,
                                  const cnnlTensorDescriptor_t, const void*,
                                  const cnnlTensorDescriptor_t, void*);

template<MathOpType>
struct GetKernelFunction;

template<>
struct GetKernelFunction<MathOpType::cos> {
  FuncType operator()() { return cnnlCos_v2; }
};

template<>
struct GetKernelFunction<MathOpType::exp> {
  FuncType operator()() { return cnnlExp_v2; }
};

template<>
struct GetKernelFunction<MathOpType::sin> {
  FuncType operator()() { return cnnlSin_v2; }
};

template<>
struct GetKernelFunction<MathOpType::sqrt> {
  FuncType operator()() { return cnnlSqrt_v2; }
};

template<MathOpType op_type>
class MluMathKernel final : public user_op::OpKernel {
 public:
  MluMathKernel() = default;
  ~MluMathKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    if (x->shape_view().elem_cnt() == 0) { return; }
    CnnlTensorDescriptor x_desc(x), y_desc(y);
    auto func = GetKernelFunction<op_type>()();
    OF_CNNL_CHECK(func(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), CNNL_COMPUTATION_FAST,
                       x_desc.desc(), x->dptr(), y_desc.desc(), y->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

inline auto SimpleMatcher(const std::string& input_name) {
  return (user_op::HobDeviceType() == DeviceType::kMLU)
         && ((user_op::HobDataType(input_name, 0) == kFloat)
             || (user_op::HobDataType(input_name, 0) == kFloat16));
}

#define REGISTER_MLU_MATH_KERNEL(kernel_name, op_type) \
  REGISTER_USER_KERNEL(kernel_name)                    \
      .SetCreateFn<MluMathKernel<op_type>>()           \
      .SetIsMatchedHob(SimpleMatcher("x"));

REGISTER_MLU_MATH_KERNEL("cos", MathOpType::cos)
REGISTER_MLU_MATH_KERNEL("exp", MathOpType::exp)
REGISTER_MLU_MATH_KERNEL("sin", MathOpType::sin)
REGISTER_MLU_MATH_KERNEL("sqrt", MathOpType::sqrt)

#undef REGISTER_MLU_MATH_KERNEL

}  // namespace
}  // namespace oneflow
