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
#include <memory>

#include <cnnl.h>

#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/ep/mlu_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

enum class BinaryOpMLU {
  kAdd,
  kMul,
};

// TODO: support other data types
// CNNL 1.15.2
static void LaunchAddKernel(user_op::KernelComputeContext* ctx, Scalar src0,
                            const user_op::Tensor* in, user_op::Tensor* out) {
  CHECK(in->shape_view().NumAxes() <= CNNL_DIM_MAX)
      << "The number of dimensions is no more than CNNL_DIM_MAX";
  using DType = float;
  CnnlTensorDescriptor input_desc;
  input_desc.set(in);
  const DType alpha = 1;
  const DType beta = src0.Value<DType>();
  auto handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
  OF_CNNL_CHECK(
      cnnlTransform(handle, &alpha, input_desc.desc(), in->dptr(), &beta, out->mut_dptr()));
}

// TODO: implement
static void LaunchMulKernel() {}

template<BinaryOpMLU op>
class ScalarMathKernelMLU final : public user_op::OpKernel {
 public:
  ScalarMathKernelMLU() = default;
  ~ScalarMathKernelMLU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Scalar value;
    if (ctx->Attr<bool>("has_int_operand")) {
      value = Scalar(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      value = Scalar(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      // TODO(Jianhua Zheng): support kSub
      const bool is_add_sub_0 = (op == BinaryOpMLU::kAdd) && value.Value<double>() == 0.0;
      // TODO(Jianhua Zheng): support kDiv
      const bool is_mul_div_1 = (op == BinaryOpMLU::kMul) && value.Value<double>() == 1.0;
      if ((is_add_sub_0 || is_mul_div_1) && in->dptr() == out->dptr()) { return; }
      if (op == BinaryOpMLU::kAdd) {
        LaunchAddKernel(ctx, value, in, out);
      } else if (op == BinaryOpMLU::kMul) {
        LaunchMulKernel();
      }
    } else {
      // For 0-d Tensor
      return;
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define SCALAR_MATH_SEQ                                 \
  OF_PP_MAKE_TUPLE_SEQ("scalar_add", BinaryOpMLU::kAdd) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_mul", BinaryOpMLU::kMul)

#define REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(op_name, binary_op)                 \
  REGISTER_USER_KERNEL(op_name)                                                             \
      .SetCreateFn<ScalarMathKernelMLU<binary_op>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                       \
                       && (user_op::HobDataType("in", 0) == user_op::HobDataType("out", 0)) \
                       && (user_op::HobDataType("in", 0) == DataType::kFloat))              \
      .SetInplaceProposalFn(                                                                \
          [](const user_op::InferContext& ctx,                                              \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {        \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));               \
            return Maybe<void>::Ok();                                                       \
          });

OF_PP_FOR_EACH_TUPLE(REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL, SCALAR_MATH_SEQ)

}  // namespace oneflow
