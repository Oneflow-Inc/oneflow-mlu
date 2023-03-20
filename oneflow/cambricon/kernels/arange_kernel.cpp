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
#include "oneflow/core/framework/framework.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {


template<typename T>
class MluArangeKernel final : public user_op::OpKernel {
 public:
  MluArangeKernel() = default;
  ~MluArangeKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();
    const DataType dtype = ctx->Attr<DataType>("dtype");
    int32_t start_int = 0;
    int32_t step_int = 0;
    float start_float = 0.0;
    float step_float = 0.0;
    CnnlTensorDescriptor out_decs;
    out_decs.set(out);
    if (std::is_same_v<T, float> || std::is_same_v<T, float16>){
      start_float = static_cast<float>(ctx->Attr<double>("float_start"));
      step_float = static_cast<float>(ctx->Attr<double>("integer_delta"));
      OF_CNNL_CHECK(cnnlArange_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), CNNL_COMPUTATION_HIGH_PRECISION, (void *)&start_float, (void *)&step_float, out_decs.desc(), output));
    }else{
      start_int = static_cast<int32_t>(ctx->Attr<int64_t>("integer_start"));
      step_int = static_cast<int32_t>(ctx->Attr<int64_t>("integer_delta"));
      OF_CNNL_CHECK(cnnlArange_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), CNNL_COMPUTATION_HIGH_PRECISION, (void *)&start_int, (void *)&step_int, out_decs.desc(), output));
    }
    
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ARANGE_MLU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("arange").SetCreateFn<MluArangeKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                \
      && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

REGISTER_ARANGE_MLU_KERNEL(float)
REGISTER_ARANGE_MLU_KERNEL(float16)
REGISTER_ARANGE_MLU_KERNEL(int8_t)
REGISTER_ARANGE_MLU_KERNEL(uint8_t)
REGISTER_ARANGE_MLU_KERNEL(int32_t)
REGISTER_ARANGE_MLU_KERNEL(uint32_t)
REGISTER_ARANGE_MLU_KERNEL(int64_t)
REGISTER_ARANGE_MLU_KERNEL(uint64_t)

}  // namespace oneflow
