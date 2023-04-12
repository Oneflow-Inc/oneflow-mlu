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
#include <math.h>

#include "oneflow/cambricon/bang/bang_kernels.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class MluReluKernel final : public user_op::OpKernel {
 public:
  MluReluKernel() = default;
  ~MluReluKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto* stream = ctx->stream()->As<ep::MluStream>();
    BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                      stream->device()->ncores_per_cluster());
    if constexpr (std::is_same<T, float16>::value) {
      bang_relu_launch_half_kernel(handle, x->shape_view().elem_cnt(), x->dptr<float16>(),
                                     y->mut_dptr<float16>());
    } else {
      bang_relu_launch_kernel(handle, x->shape_view().elem_cnt(), x->dptr<float>(),
                                y->mut_dptr<float>());
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_MLU_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("relu")                                   \
      .SetCreateFn<MluReluKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_RELU_MLU_KERNEL(float)
REGISTER_RELU_MLU_KERNEL(float16)

#undef REGISTER_RELU_MLU_KERNEL

}  // namespace oneflow
