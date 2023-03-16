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

std::map<std::pair<DataType, DataType>, cnnlCastDataType_t> cast_dtype_table = {
    {{kFloat, kFloat16}, CNNL_CAST_FLOAT_TO_HALF},
    {{kFloat, kInt32}, CNNL_CAST_FLOAT_TO_INT32},
    {{kFloat, kInt8}, CNNL_CAST_FLOAT_TO_INT8},
    {{kFloat, kUInt8}, CNNL_CAST_FLOAT_TO_UINT8},
    {{kFloat16, kFloat}, CNNL_CAST_HALF_TO_FLOAT},
    {{kFloat16, kInt32}, CNNL_CAST_HALF_TO_INT32},
    {{kFloat16, kInt8}, CNNL_CAST_HALF_TO_INT8},
    {{kInt32, kInt8}, CNNL_CAST_INT32_TO_INT8},
    {{kFloat16, kBool}, CNNL_CAST_HALF_TO_BOOL},
    {{kInt8, kFloat}, CNNL_CAST_INT8_TO_FLOAT},
    {{kInt8, kFloat16}, CNNL_CAST_INT8_TO_HALF},
    {{kInt8, kInt32}, CNNL_CAST_INT8_TO_INT32},
    {{kUInt8, kFloat}, CNNL_CAST_UINT8_TO_FLOAT},
    {{kUInt8, kFloat16}, CNNL_CAST_UINT8_TO_HALF},
    {{kBool, kFloat}, CNNL_CAST_BOOL_TO_FLOAT},
    {{kBool, kFloat16}, CNNL_CAST_BOOL_TO_HALF},
    {{kBool, kInt32}, CNNL_CAST_BOOL_TO_INT32},
    {{kUInt8, kInt32}, CNNL_CAST_UINT8_TO_INT32},
    {{kInt32, kInt64}, CNNL_CAST_INT32_TO_INT64},
    {{kInt64, kInt32}, CNNL_CAST_INT64_TO_INT32},
    {{kInt32, kBool}, CNNL_CAST_INT32_TO_BOOL},
    {{kUInt8, kInt64}, CNNL_CAST_UINT8_TO_INT64},
    {{kUInt64, kUInt32}, CNNL_CAST_UINT64_TO_UINT32},
    {{kInt64, kUInt32}, CNNL_CAST_INT64_TO_UINT32},
    {{kInt64, kFloat}, CNNL_CAST_INT64_TO_FLOAT},
    {{kInt64, kFloat16}, CNNL_CAST_INT64_TO_HALF},
    {{kFloat, kInt64}, CNNL_CAST_FLOAT_TO_INT64},
    {{kFloat16, kInt64}, CNNL_CAST_HALF_TO_INT64},
    {{kInt32, kFloat}, CNNL_CAST_INT32_TO_FLOAT},
};

template<typename T>
class MluCastKernel final : public user_op::OpKernel {
 public:
  MluCastKernel() = default;
  ~MluCastKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType in_data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
    const DataType out_data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();

    cnnlCastDataType_t type;
    auto it = cast_dtype_table.find(std::make_pair(in_data_type, out_data_type));
    if (it != cast_dtype_table.end()) {
      type = it->second;
    } else {
      UNIMPLEMENTED();
    }

    CnnlTensorDescriptor in_desc, out_decs;
    in_desc.set(in);
    out_decs.set(out);
    OF_CNNL_CHECK(cnnlCastDataType(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                   in_desc.desc(), in->dptr(), type, out_decs.desc(),
                                   out->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CAST_MLU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("cast").SetCreateFn<MluCastKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_CAST_MLU_KERNEL(float)
REGISTER_CAST_MLU_KERNEL(float16)
REGISTER_CAST_MLU_KERNEL(int8_t)
REGISTER_CAST_MLU_KERNEL(uint8_t)
REGISTER_CAST_MLU_KERNEL(int32_t)

}  // namespace oneflow