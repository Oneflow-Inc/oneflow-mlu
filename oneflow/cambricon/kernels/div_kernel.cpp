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
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

template<typename T>
class MluBroadCastDivKernel final : public user_op::OpKernel {
 public:
  MluBroadCastDivKernel() = default;
  ~MluBroadCastDivKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("z", 0);

    CnnlTensorDescriptor x_desc, y_decs, z_desc;
    x_desc.set(x);
    y_decs.set(y);
    z_desc.set(z);
    size_t div_workspace_size = 0;
    void* div_workspace = nullptr;

    OF_CNNL_CHECK(cnnlDiv_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                             CNNL_COMPUTATION_HIGH_PRECISION, x_desc.desc(), x->dptr(),
                             y_decs.desc(), y->dptr(), div_workspace, div_workspace_size,
                             z_desc.desc(), z->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIV_MLU_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("div").SetCreateFn<MluBroadCastDivKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                       \
      && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_DIV_MLU_KERNEL(float)
REGISTER_DIV_MLU_KERNEL(float16)
REGISTER_DIV_MLU_KERNEL(int8_t)
REGISTER_DIV_MLU_KERNEL(uint8_t)
REGISTER_DIV_MLU_KERNEL(int32_t)

}  // namespace oneflow
