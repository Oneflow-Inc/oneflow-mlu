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
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

template<typename T, typename K>
class MluGatherNdKernel final : public user_op::OpKernel {
 public:
  MluGatherNdKernel() = default;
  ~MluGatherNdKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    const user_op::Tensor* params = ctx->Tensor4ArgNameAndIndex("params", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (params->shape_view().elem_cnt() == 0 || indices->shape_view().elem_cnt() == 0
        || out->shape_view().elem_cnt() == 0) {
      return;
    }
    CnnlTensorDescriptor indices_desc(indices), params_desc(params), out_desc(out);
    OF_CNNL_CHECK(cnnlGatherNd(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                               params_desc.desc(), params->dptr(), indices_desc.desc(),
                               indices->dptr(), out_desc.desc(), out->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_GATHERND_KERNEL(in_type, indices_type)                                        \
  REGISTER_USER_KERNEL("gather_nd")                                                                \
      .SetCreateFn<MluGatherNdKernel<OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>() \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceType() == DeviceType::kMLU)                                           \
          && (user_op::HobDataType("params", 0) == OF_PP_PAIR_SECOND(in_type))                     \
          && (user_op::HobDataType("indices", 0) == OF_PP_PAIR_SECOND(indices_type)));

#define GATHERND_DATA_TYPE_SEQ                    \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)   \
  OF_PP_MAKE_TUPLE_SEQ(float16, DataType::kFloat16)

#define GATHERND_INDEX_DATA_TYPE_SEQ              \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MLU_GATHERND_KERNEL, GATHERND_DATA_TYPE_SEQ,
                                 GATHERND_INDEX_DATA_TYPE_SEQ)

#undef GATHERND_DATA_TYPE_SEQ
#undef GATHERND_INDEX_DATA_TYPE_SEQ
#undef REGISTER_MLU_GATHERND_KERNEL

}  // namespace
}  // namespace oneflow
