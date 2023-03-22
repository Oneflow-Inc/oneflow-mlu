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
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"

namespace oneflow {

template<typename T>
class MluL1L2RegularizeGradientKernel final : public user_op::OpKernel {
 public:
  MluL1L2RegularizeGradientKernel() = default;
  ~MluL1L2RegularizeGradientKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  // formula: out = model_diff + l1 * (model >= 0 ? 1 : -1) + l2 * model
  // 1. create workspace filled by l1
  // 2. create workspace to store signed_l1 = l1 * (model >= 0 ? 1 : -1)
  // 3. calculate signed_l1 = l1 * (model >= 0 ? 1 : -1) with CopySign
  //    CopySign uses the sign of model and absolute value of l1 to calculate signed_l1
  // 4. calculate regularization = signed_l1 + l2 * model with AddCMul,
  //    reuse l1_workspace to save l2
  // 5. calucate out = model_diff + regularization
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");

    CnnlTensorDescriptor model_desc, model_diff_desc, out_desc;
    model_desc.set(model);
    model_diff_desc.set(model_diff);
    out_desc.set(out);

    // 1. create workspace filled by l1
    CnnlTensorDescriptor l1_desc;
    l1_desc.set(model);
    CnnlWorkspace l1_workspace(
        ctx->stream()->As<ep::MluStream>(),
        model->shape_view().elem_cnt() * GetSizeOfDataType(model->data_type()));
    T l1_value = static_cast<T>(l1);
    OF_CNNL_CHECK(cnnlFill_v3(
        /* handle       */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* pointer_mode */ CNNL_POINTER_MODE_HOST,
        /* value        */ &l1_value,
        /* output_desc  */ l1_desc.desc(),
        /* output       */ l1_workspace.dptr()));

    // 2. create workspace to store signed_l1 = l1 * (model[i] >= 0 ? 1 : -1)
    CnnlTensorDescriptor signed_l1_desc;
    signed_l1_desc.set(model);
    CnnlWorkspace signed_l1_workspace(
        ctx->stream()->As<ep::MluStream>(),
        model->shape_view().elem_cnt() * GetSizeOfDataType(model->data_type()));

    // 3. calculate signed_l1 = l1 * (model >= 0 ? 1 : -1) with CopySign
    // CopySign uses the sign of model and absolute value of l1 to calculate signed_l1
    size_t copy_sign_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetCopySignWorkspaceSize(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc     */ l1_desc.desc(),
        /* other_desc     */ model_desc.desc(),
        /* output_desc    */ signed_l1_desc.desc(),
        /* workspace_size */ &copy_sign_workspace_size));
    CnnlWorkspace copy_sign_workspace(ctx->stream()->As<ep::MluStream>(), copy_sign_workspace_size);
    OF_CNNL_CHECK(cnnlCopySign(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc     */ l1_desc.desc(),
        /* input          */ l1_workspace.dptr(),
        /* other_desc     */ model_desc.desc(),
        /* other          */ model->dptr(),
        /* workspace      */ copy_sign_workspace.dptr(),
        /* workspace_size */ copy_sign_workspace_size,
        /* output_desc    */ signed_l1_desc.desc(),
        /* output         */ signed_l1_workspace.dptr()));

    // 4. calculate regularization = signed_l1 + l2 * model with AddCMul.
    // The formula of AddCMul is (a + b * c * alpha) where a is signed_l1,
    // alpha is l2, b is 1.0f (reuse l1_workspace), c is model
    T one = static_cast<T>(1.0f);
    OF_CNNL_CHECK(cnnlFill_v3(
        /* handle       */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* pointer_mode */ CNNL_POINTER_MODE_HOST,
        /* value        */ &one,
        /* output_desc  */ l1_desc.desc(),
        /* output       */ l1_workspace.dptr()));
    
    CnnlWorkspace regularization_workspace(
        ctx->stream()->As<ep::MluStream>(),
        model->shape_view().elem_cnt() * GetSizeOfDataType(model->data_type()));
    CnnlTensorDescriptor regularization_desc;
    regularization_desc.set(model);

    size_t addcmul_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetAddcmulWorkspaceSize(
        /* handle */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* desc_a */ signed_l1_desc.desc(),
        /* desc_b */ l1_desc.desc(),
        /* desc_c */ model_desc.desc(),
        /* size   */ &addcmul_workspace_size));
    CnnlWorkspace addcmul_workspace(ctx->stream()->As<ep::MluStream>(),
                                    addcmul_workspace_size);
    T l2_value = static_cast<T>(l2);
    // regulazion = a         + b    * c     * alpha
    //            = signed_l1 + 1.0f * model * l2
    OF_CNNL_CHECK(cnnlAddcmul(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* desc_a         */ signed_l1_desc.desc(),
        /* a              */ signed_l1_workspace.dptr(),
        /* alpha          */ &l2_value,
        /* desc_b         */ l1_desc.desc(),
        /* b              */ l1_workspace.dptr(),
        /* desc_c         */ model_desc.desc(),
        /* c              */ model->dptr(),
        /* workspace      */ addcmul_workspace.dptr(),
        /* workspace_size */ addcmul_workspace_size,
        /* desc_output    */ regularization_desc.desc(),
        /* output         */ regularization_workspace.dptr()));

    // 5. calculate out = model_diff + regularization
    std::vector<cnnlTensorDescriptor_t> input_descs{model_diff_desc.desc(),
                                                    regularization_desc.desc()};
    std::vector<const void*> inputs{model_diff->dptr(), regularization_workspace.dptr()};
    size_t out_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetAddNWorkspaceSize(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_descs    */ input_descs.data(),
        /* input_num      */ 2,
        /* output_desc    */ out_desc.desc(),
        /* workspace_size */ &out_workspace_size));
    CnnlWorkspace out_workspace(ctx->stream()->As<ep::MluStream>(), out_workspace_size);
    OF_CNNL_CHECK(cnnlAddN_v2(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_descs    */ input_descs.data(),
        /* const *inputs  */ inputs.data(),
        /* input_num      */ 2,
        /* output_desc    */ out_desc.desc(),
        /* output         */ out->mut_dptr(),
        /* workspace      */ out_workspace.dptr(),
        /* workspace_size */ out_workspace_size));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_L1_L2_REGULARIZE_GRADIENT_MLU_KERNEL(dtype)          \
  REGISTER_USER_KERNEL("l1_l2_regularize_gradient")                   \
      .SetCreateFn<MluL1L2RegularizeGradientKernel<dtype>>()          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value));

REGISTER_L1_L2_REGULARIZE_GRADIENT_MLU_KERNEL(float)
REGISTER_L1_L2_REGULARIZE_GRADIENT_MLU_KERNEL(float16)

}  // namespace oneflow
