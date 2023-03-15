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

namespace oneflow {

template<typename T>
class LayerNormMluKernel final : public user_op::OpKernel {
 public:
  LayerNormMluKernel() = default;
  ~LayerNormMluKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      gamma_ptr = gamma->dptr<T>();
      CHECK_EQ(gamma->shape_view().elem_cnt(), norm_size);
    }
    if (ctx->has_input("beta", 0)) { beta_ptr = ctx->Tensor4ArgNameAndIndex("beta", 0)->dptr<T>(); }
  };
};

#define REGISTER_LAYER_NORM_MLU_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("layer_norm")                                                   \
      .SetCreateFn<LayerNormMluKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                  \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {\
const int64_t begin_norm_axis = ctx->Attr<int64_t>("begin_norm_axis");\
const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);\
CnnlTensorDescriptor input_desc;\
input_desc.set(in);\
size_t tmp_buffer_size = 0;\
OF_CNNL_CHECK(cnnlGetLayerNormOpWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),\
                                              begin_norm_axis, input_desc, &tmp_buffer_size));\
return tmp_buffer_size;
}  // namespace oneflow
  );

  REGISTER_LAYER_NORM_MLU_KERNEL(float)
  REGISTER_LAYER_NORM_MLU_KERNEL(float16)

  }  // namespace oneflow
