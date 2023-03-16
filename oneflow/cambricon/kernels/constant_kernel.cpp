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
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T>
class MluConstantKernel final : public user_op::OpKernel {
 public:
  MluConstantKernel() = default;
  ~MluConstantKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool is_floating_value = ctx->Attr<bool>("is_floating_value");
    T value;
    if (is_floating_value) {
      value = static_cast<T>(ctx->Attr<double>("floating_value"));
    } else {
      value = static_cast<T>(ctx->Attr<int64_t>("integer_value"));
    }

    CnnlTensorDescriptor out_decs;
    out_decs.set(out_tensor);
    OF_CNNL_CHECK(cnnlFill_v3(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                              CNNL_POINTER_MODE_HOST, &value, out_decs.desc(),
                              out_tensor->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FILL_MLU_KERNEL(dtype)                               \
  REGISTER_USER_KERNEL("constant")                                    \
      .SetCreateFn<MluConstantKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FILL_MLU_KERNEL(float)
REGISTER_FILL_MLU_KERNEL(float16)
REGISTER_FILL_MLU_KERNEL(int8_t)
REGISTER_FILL_MLU_KERNEL(uint8_t)
REGISTER_FILL_MLU_KERNEL(int32_t)

}  // namespace oneflow
