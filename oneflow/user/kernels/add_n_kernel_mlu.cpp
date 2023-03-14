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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/mlu/mlu_tools.h"

namespace oneflow {

namespace{
typedef struct Add_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
} Add;

}

template<typename T>
class MluAddNKernel final : public user_op::OpKernel {
 public:
  MluAddNKernel() = default;
  ~MluAddNKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    size_t in_num = ctx->inputs().size();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    void* out_dptr = static_cast<void*>(out->mut_dptr());

    std::vector<const void*> input_dptrs_vec(in_num);
    const auto& first_input_shape = ctx->Tensor4ArgNameAndIndex("in", 0)->shape();
    for (size_t i = 0; i < input_dptrs_vec.size(); ++i) {
      const auto* input_i_tensor = ctx->Tensor4ArgNameAndIndex("in", i);
      CHECK_EQ(first_input_shape, input_i_tensor->shape());
      input_dptrs_vec[i] = input_i_tensor->dptr();
    }
    size_t ndim = first_input_shape.NumAxes();
    std::vector<int> dim_vec(ndim);
    for (size_t i = 0; i < ndim; ++i) { dim_vec[i] = first_input_shape.At(i); }

    Add add;
    AddType datainfo;
    datainfo.input_dtype = convertCamDataType(T);
    datainfo.output_dtype = convertCamDataType(T);
    setTensorDesc(add.input_desc, dim_vec.size(), dim_vec.data(), datainfo.input_dtype,
                    datainfo.layout);
    setTensorDesc(add.output_desc, dim_vec.size(), dim_vec.data(), datainfo.output_dtype,
                    datainfo.layout);
    std::vector<cnnlTensorDescriptor_t> input_descs_vec{in_num, add.input_desc};
    CNNL_CHECK(cnnlAddN(ctx->device_ctx()->cambricon_handle(), input_descs_vec.data(),
                        input_dptrs_vec.data(), in_num, add.output_desc, out_dptr));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADDN_MLU_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("add_n")                                \
      .SetCreateFn<MluAddNKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU));

REGISTER_ADDN_MLU_KERNEL(CamDataType::kFLOAT32)
REGISTER_ADDN_MLU_KERNEL(CamDataType::kFLOAT64)
REGISTER_ADDN_MLU_KERNEL(CamDataType::kHALF)


}  // namespace oneflow
