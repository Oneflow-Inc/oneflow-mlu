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
#include <cassert>
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/mlu/mlu_tools.h"
#include "oneflow/core/common/util.h"

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
    std::cout << "enter cambricon add kernel" << std::endl;
    size_t in_num = ctx->inputs().size();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    void* out_dptr = static_cast<void*>(out->mut_dptr());

    std::vector<const void*> input_dptrs_vec(in_num);
    Shape first_input_shape;
    ctx->Tensor4ArgNameAndIndex("in", 0)->shape_view().ToShape(&first_input_shape);
    for (size_t i = 0; i < input_dptrs_vec.size(); ++i) {
      const auto* input_i_tensor = ctx->Tensor4ArgNameAndIndex("in", i);
      Shape input_i_shape;
      input_i_tensor->shape_view().ToShape(&input_i_shape);
      CHECK_EQ(first_input_shape, input_i_shape);
      input_dptrs_vec[i] = input_i_tensor->dptr();
    }
    assert(first_input_shape.NumAxes() == 4);
    Shape2D input_t;
    input_t.n = first_input_shape.At(0);
    input_t.h = first_input_shape.At(1);
    input_t.w = first_input_shape.At(2);
    input_t.c = first_input_shape.At(3);

    Shape2D output_t;
    output_t.n = first_input_shape.At(0);
    output_t.h = first_input_shape.At(1);
    output_t.w = first_input_shape.At(2);
    output_t.c = first_input_shape.At(3);

    Add add;
    AddType datainfo;
    
    int v = 0;
    if (GetDataType<T>::value == DataType::kFloat){
      v = 2;
    }
    else if(GetDataType<T>::value == DataType::kFloat16){
      v = 1;
    }
    datainfo.input_dtype = (cnnlDataType_t)v;;
    datainfo.output_dtype = (cnnlDataType_t)v;;
    setTensorDesc2D(add.input_desc, input_t, datainfo.input_dtype, datainfo.layout);
    setTensorDesc2D(add.output_desc, output_t, datainfo.output_dtype, datainfo.layout);
    std::vector<cnnlTensorDescriptor_t> input_descs_vec{in_num, add.input_desc};
    
    CNNL_CHECK(cnnlAddN(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), input_descs_vec.data(),
                        input_dptrs_vec.data(), in_num, add.output_desc, out_dptr));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADDN_MLU_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("add_n")                        \
      .SetCreateFn<MluAddNKernel<dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_ADDN_MLU_KERNEL(float)


}  // namespace oneflow
