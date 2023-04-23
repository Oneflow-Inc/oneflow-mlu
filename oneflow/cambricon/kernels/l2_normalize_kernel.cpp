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
#include <string>
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

class CnnlNormalizeDescriptor
    : public CnnlDescriptor<cnnlNormalizeStruct, &cnnlCreateNormalizeDescriptor,
                            &cnnlDestroyNormalizeDescriptor> {
 public:
  CnnlNormalizeDescriptor() {}
  void set(int axis[], int axis_num, cnnlNormalizeMode_t mode, cnnlNanPropagation_t nan_propagation,
           float eps) {
    OF_CNNL_CHECK(
        cnnlSetNormalizeDescriptor_v2(mut_desc(), axis, axis_num, nan_propagation, eps, 2.0, 0, 0));
  }
};

class MluL2NormalizeKernel final : public user_op::OpKernel {
 public:
  MluL2NormalizeKernel() = default;
  ~MluL2NormalizeKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* square_x_sum = ctx->Tensor4ArgNameAndIndex("square_x_sum", 0);
    const float epsilon = ctx->Attr<float>("epsilon");
    int axis[1] = {static_cast<int>(ctx->Attr<int32_t>("axis"))};
    CnnlNormalizeDescriptor norm_op_desc;
    norm_op_desc.set(axis, 1, CNNL_NORMALIZE_EUCLIDEAN, CNNL_NOT_PROPAGATE_NAN, epsilon);
    CnnlTensorDescriptor x_desc(x), y_desc(y), square_x_sum_desc(square_x_sum);
    OF_CNNL_CHECK(cnnlNormalize_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                   norm_op_desc.desc(), x_desc.desc(), x->dptr(), nullptr, nullptr,
                                   y_desc.desc(), y->mut_dptr(), square_x_sum_desc.desc(),
                                   square_x_sum->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

inline auto SimpleMatcher(const std::string& input_name) {
  return (user_op::HobDeviceType() == DeviceType::kMLU)
         && ((user_op::HobDataType(input_name, 0) == kFloat)
             || (user_op::HobDataType(input_name, 0) == kFloat16));
}

REGISTER_USER_KERNEL("l2_normalize")
    .SetCreateFn<MluL2NormalizeKernel>()
    .SetIsMatchedHob(SimpleMatcher("x"));

}  // namespace
}  // namespace oneflow
