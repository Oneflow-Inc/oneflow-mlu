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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

enum class BinaryOpMLU {
  kAdd,
  kMul,
  kSub,
};

union Param {
  int int_value;
  float float_value;
};

struct TransformParams {
  Param alpha;  // scaling factor of tensor input
  Param beta;   // bias factor of tensor input
};

template<typename T, BinaryOpMLU op>
T GetAlpha(Scalar value) {
  switch (op) {
    case BinaryOpMLU::kAdd: return T(1);
    case BinaryOpMLU::kMul: return value.Value<T>();
    case BinaryOpMLU::kSub: return 1;
    default: THROW(RuntimeError) << "Invalid op in MLU LaunchMathKernel " << op;
  }
  return 0;  // eliminating compiler warnings
}

template<typename T, BinaryOpMLU op>
T GetBeta(Scalar value) {
  switch (op) {
    case BinaryOpMLU::kAdd: return value.Value<T>();
    case BinaryOpMLU::kMul: return T(0);
    case BinaryOpMLU::kSub: return -value.Value<T>();
    default: THROW(RuntimeError) << "Invalid op in MLU LaunchMathKernel " << op;
  }
  return 0;  // eliminating compiler warnings
}

template<BinaryOpMLU op>
void SetTransformParams(DataType data_type, Scalar src0, TransformParams& params) {
  // If the data type of tensors is float or half, the data type of alpha and beta should be
  // `float*`. If the data type of tensors is int32, the data type of alpha and beta should be
  // `int*`.
  switch (data_type) {
    case DataType::kFloat:
    case DataType::kFloat16:
      params.alpha.float_value = GetAlpha<float, op>(src0);
      params.beta.float_value = GetBeta<float, op>(src0);
      break;
    case DataType::kInt32:
      params.alpha.int_value = GetAlpha<int, op>(src0);
      params.beta.int_value = GetBeta<int, op>(src0);
      break;
    // The combinations of the data types for input tensor and output tensor must be half-half,
    // float-float or int32-int32.
    default:
      THROW(RuntimeError) << "MLU LaunchMathKernel does not support data type "
                          << DataType_Name(data_type);
  }
}

template<BinaryOpMLU op>
static void LaunchMathKernel(user_op::KernelComputeContext* ctx, Scalar src0,
                             const user_op::Tensor* in, user_op::Tensor* out) {
  CHECK(in->shape_view().NumAxes() <= CNNL_DIM_MAX)
      << "The number of dimensions is no more than CNNL_DIM_MAX";
  TransformParams params;
  SetTransformParams<op>(in->data_type(), src0, params);
  CnnlTensorDescriptor input_desc;
  input_desc.set(in);
  auto handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
  OF_CNNL_CHECK(cnnlTransform(handle, &params.alpha, input_desc.desc(), in->dptr(), &params.beta,
                              out->mut_dptr()));
}

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
      LaunchMathKernel<op>(ctx, value, in, out);
    } else {
      // For 0-d Tensor
      return;
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define SCALAR_MATH_SEQ                                 \
  OF_PP_MAKE_TUPLE_SEQ("scalar_add", BinaryOpMLU::kAdd) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_mul", BinaryOpMLU::kMul) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_sub", BinaryOpMLU::kSub)

#define REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(op_name, binary_op)                 \
  REGISTER_USER_KERNEL(op_name)                                                             \
      .SetCreateFn<ScalarMathKernelMLU<binary_op>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                       \
                       && (user_op::HobDataType("in", 0) == user_op::HobDataType("out", 0)) \
                       && ((user_op::HobDataType("in", 0) == DataType::kFloat)              \
                           || (user_op::HobDataType("in", 0) == DataType::kFloat16)         \
                           || (user_op::HobDataType("in", 0) == DataType::kInt32)))         \
      .SetInplaceProposalFn(                                                                \
          [](const user_op::InferContext& ctx,                                              \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {        \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));               \
            return Maybe<void>::Ok();                                                       \
          });

OF_PP_FOR_EACH_TUPLE(REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL, SCALAR_MATH_SEQ)

}  // namespace oneflow
