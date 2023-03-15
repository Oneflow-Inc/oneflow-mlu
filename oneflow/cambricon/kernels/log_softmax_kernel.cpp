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
#include "cnnl.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/ep/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

std::vector<int64_t> modify_dims_based_on_layout(const std::vector<long>& dim,
                                                 const std::string& memory_format) {
  if (!dim.size()) { return dim; }
  std::vector<int64_t> target_dim;
  std::vector<int> dim_order;
  // trans tensor/stride size to cnnl desc size/stride.
  auto modify_dims_pos = [](const std::vector<int>& dim_order, const std::vector<int64_t>& input,
                            std::vector<int64_t>& out) {
    out.clear();
    for (const auto& item : dim_order) { out.push_back(input[item]); }
  };
  if (memory_format == "ChannelsLast") {
    dim_order = {0, 2, 3, 1};
    modify_dims_pos(dim_order, dim, target_dim);
  } else if (memory_format == "ChannelsLast3d") {
    dim_order = {0, 2, 3, 4, 1};
    modify_dims_pos(dim_order, dim, target_dim);
  } else if (memory_format == "Contiguous") {
    target_dim = dim;
  }
  return target_dim;
};

template<typename T>
class MluLogSoftmaxKernel final : public user_op::OpKernel {
 public:
  MluLogSoftmaxKernel() = default;
  ~MluLogSoftmaxKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("prob", 0);
    CnnlTensorDescriptor input_desc, output_desc;

    std::vector<int> addentional_dims_input = {1, static_cast<int>(in->shape_view().At(0)),
                                               static_cast<int>(in->shape_view().At(1))};
    std::vector<int> addentional_dims_output = {1, static_cast<int>(in->shape_view().At(0)),
                                                static_cast<int>(in->shape_view().At(1))};

    input_desc.set_additional_dim(in, addentional_dims_input);
    output_desc.set_additional_dim(out, addentional_dims_output);

    OF_CNNL_CHECK(cnnlSoftmaxForward(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), cnnlSoftmaxAlgorithm_t::CNNL_SOFTMAX_LOG,
        cnnlSoftmaxMode_t::CNNL_SOFTMAX_MODE_LOW_DIMENSION, nullptr, input_desc.desc(), in->dptr(),
        NULL, output_desc.desc(), out->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_LOG_SOFTMAX_MLU_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("log_softmax")                                 \
      .SetCreateFn<MluLogSoftmaxKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_LOG_SOFTMAX_MLU_KERNEL(float)
REGISTER_LOG_SOFTMAX_MLU_KERNEL(float16)
// REGISTER_LOG_SOFTMAX_MLU_KERNEL(int8_t)
// REGISTER_LOG_SOFTMAX_MLU_KERNEL(uint8_t)
// REGISTER_LOG_SOFTMAX_MLU_KERNEL(int32_t)

}  // namespace oneflow
