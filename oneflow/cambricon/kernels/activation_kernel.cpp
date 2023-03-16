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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<cnnlActivationMode_t mode>
void SetCnnlActivationDescriptor(CnnlActivationDescriptor* activation_desc) {
  activation_desc->set(mode, /*prefer=*/CNNL_ACTIVATION_HIGH_PRECISION,
                       /*nanProp=*/CNNL_NOT_PROPAGATE_NAN, /*ceof=*/1.0);
}

template<typename T, cnnlActivationMode_t mode, const char* input_name, const char* output_name>
class MluActivationKernel final : public user_op::OpKernel {
 public:
  MluActivationKernel() = default;
  ~MluActivationKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex(input_name, 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex(output_name, 0);

    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set(in);
    output_desc.set(out);
    CnnlActivationDescriptor activation_desc;
    SetCnnlActivationDescriptor<mode>(&activation_desc);
    OF_CNNL_CHECK(cnnlActivationForward(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                        activation_desc.desc(), nullptr, input_desc.desc(),
                                        in->dptr(), nullptr, output_desc.desc(), out->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ACTIVATION_MLU_KERNEL(name, mode, dtype, input_name, output_name) \
  REGISTER_USER_KERNEL(name)                                                       \
      .SetCreateFn<MluActivationKernel<dtype, mode, input_name, output_name>>()    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)              \
                       && (user_op::HobDataType(input_name, 0) == GetDataType<dtype>::value));

const char x_str[] = "x";
const char y_str[] = "y";
const char in_str[] = "in";
const char out_str[] = "out";
REGISTER_ACTIVATION_MLU_KERNEL("relu", CNNL_ACTIVATION_RELU, float, x_str, y_str)
REGISTER_ACTIVATION_MLU_KERNEL("relu", CNNL_ACTIVATION_RELU, float16, x_str, y_str)
REGISTER_ACTIVATION_MLU_KERNEL("gelu", CNNL_ACTIVATION_GELU, float, in_str, out_str)
REGISTER_ACTIVATION_MLU_KERNEL("gelu", CNNL_ACTIVATION_GELU, float16, in_str, out_str)

}  // namespace oneflow
