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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/bang/bang_internal.h"

namespace oneflow {

namespace {

// Kernel arg size has 4K limit, but currently we set process 32 tensors in each kernel.
constexpr int kMaxTuples = 32;


template<typename T, typename G>
class MluMultiTensorAdamUpdateKernel final : public user_op::OpKernel {
 public:
  MluMultiTensorAdamUpdateKernel() = default;
  ~MluMultiTensorAdamUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t n_tensor = ctx->input_size("model");
    const auto scale = ctx->Attr<double>("scale");
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");

    const float beta1 = ctx->Attr<float>("beta1");
    const float beta2 = ctx->Attr<float>("beta2");
    const float epsilon = ctx->Attr<float>("epsilon");
    const float weight_decay = ctx->Attr<float>("weight_decay");

    const bool amsgrad = ctx->Attr<bool>("amsgrad");
    const bool do_bias_correction = ctx->Attr<bool>("do_bias_correction");
    if (amsgrad) { UNIMPLEMENTED() << "Multi Tensor Adam Update do not support amsgrad = True. "; }

    const float* learning_rate_ptr = nullptr;
    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float lr_scale = ctx->Attr<float>("learning_rate_scale");

    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }

    const float bias_correction1_val = ctx->Attr<float>("bias_correction1_val");
    const float* bias_correction1_ptr = nullptr;
    if (ctx->has_input("bias_correction1", 0)) {
      const user_op::Tensor* bias_correction1 = ctx->Tensor4ArgNameAndIndex("bias_correction1", 0);
      CHECK_EQ(bias_correction1->shape_view().elem_cnt(),
               1);  // Just for Lazy Optional Input Check.
      bias_correction1_ptr = bias_correction1->dptr<float>();
    }

    const float bias_correction2_val = ctx->Attr<float>("bias_correction2_val");
    const float* bias_correction2_ptr = nullptr;
    if (ctx->has_input("bias_correction2", 0)) {
      const user_op::Tensor* bias_correction2 = ctx->Tensor4ArgNameAndIndex("bias_correction2", 0);
      CHECK_EQ(bias_correction2->shape_view().elem_cnt(),
               1);  // Just for Lazy Optional Input Check.
      bias_correction2_ptr = bias_correction2->dptr<float>();
    }

    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), ctx->Tensor4ArgNameAndIndex("model", 0)->data_type());
      CHECK_EQ(scale_by_tensor->shape_view().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape_view().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    float beta1_correction_recip = 1;
    float beta2_correction_recip = 1;
    int64_t step = 1;
    if (do_bias_correction) {
      beta1_correction_recip = 1 / (1 - std::pow(beta1, step));
      beta2_correction_recip = 1 / (1 - std::pow(beta2, step));
    }
    float epsilon_correction = epsilon / std::sqrt(beta2_correction_recip);
    float learning_rate_correction =
        learning_rate_val * beta1_correction_recip / std::sqrt(beta2_correction_recip);
    float weight_decay_correction = 1 - learning_rate_val * weight_decay;

    cnrtDataType_t cnrt_type = fromCnnlType2CnrtType(
        ConvertToCnnlDataType(ctx->Tensor4ArgNameAndIndex("model_diff", 0)->data_type()));
    cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
    cnrtDim3_t k_dim;
    uint32_t union_number = getDeviceAttr(cnrtAttrClusterCount);
    uint32_t core_dim = getDeviceAttr(cnrtAttrMcorePerCluster);
    k_dim.x = core_dim;
    k_dim.y = union_number;
    k_dim.z = 1;

    AddressList g, m, v, p;
    SizeList sizes;
    int tensor_count = 0;
    int adam_w_mode = 0;

    int32_t total_elem_cnt = 0;
    cnrtQueue_t queue = nullptr;
    OF_CNNL_CHECK(cnnlGetQueue(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), &queue));
    for (int tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
      const int64_t tensor_elem_cnt =
          ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->shape_view().elem_cnt();
      g.addresses[tensor_count] = ctx->Tensor4ArgNameAndIndex("model_diff", tensor_idx)->mut_dptr();
      m.addresses[tensor_count] = ctx->Tensor4ArgNameAndIndex("m", tensor_idx)->mut_dptr();
      v.addresses[tensor_count] = ctx->Tensor4ArgNameAndIndex("v", tensor_idx)->mut_dptr();
      p.addresses[tensor_count] = ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->mut_dptr();
      ;
      sizes.sizes[tensor_count] = tensor_elem_cnt;

      tensor_count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (tensor_count == kMaxTuples || tensor_idx == n_tensor - 1) {
        bang_fused_adam_internal(g, m, v, p, sizes, tensor_count, beta1, beta2, epsilon_correction,
                                 learning_rate_correction, adam_w_mode, weight_decay,
                                 weight_decay_correction, k_dim, k_type, queue, cnrt_type);
        tensor_count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MULTI_TENSOR_ADAM_UPDATE_KERNEL(dtype, gtype)                            \
  REGISTER_USER_KERNEL("multi_tensor_adam_update")                                        \
      .SetCreateFn<MluMultiTensorAdamUpdateKernel<dtype, gtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                     \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_MULTI_TENSOR_ADAM_UPDATE_KERNEL(float, float);

}  // namespace
}  // namespace oneflow
