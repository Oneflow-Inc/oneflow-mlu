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
#include <cstdint>
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

template<typename T>
class MluNormalizationKernel final : public user_op::OpKernel {
 public:
  MluNormalizationKernel() = default;
  ~MluNormalizationKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(!training);
     
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    CHECK_EQ(axis, x->shape_view().NumAxes() - 1);
    const auto epsilon = ctx->Attr<float>("epsilon");
    
    CnnlTensorDescriptor input_desc, output_desc, weight_bias_mean_var_desc;
    input_desc.set(x);
    output_desc.set(y);
    weight_bias_mean_var_desc.set(gamma);
    OF_CNNL_CHECK(cnnlBatchNormForwardInference(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), nullptr, nullptr, input_desc.desc(), x->dptr(),
                                           weight_bias_mean_var_desc.desc(), gamma->dptr(), beta->dptr(), moving_mean->dptr(),
                                           moving_variance->raw_dptr(), epsilon, output_desc.desc(), y->mut_dptr()));
    ctx->stream()->Sync();
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_MLU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("normalization").SetCreateFn<MluNormalizationKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                 \
      && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_RELU_MLU_KERNEL(float)
REGISTER_RELU_MLU_KERNEL(float16)

}  // namespace oneflow
