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
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename Context>
std::unique_ptr<oneflow::ep::primitive::Permute> NewPermutePrimitive(const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(DeviceType::kMLU, num_dims);
}

template<typename T>
class AdaptiveAvgPool2DKernel final : public user_op::OpKernel {
 public:
  AdaptiveAvgPool2DKernel() = default;
  ~AdaptiveAvgPool2DKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();

    size_t tmp_in_workspace_size = in_tensor->shape_view().elem_cnt();
    CnnlWorkspace tmp_in_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_in_workspace_size);
    void* tmp_in_ptr = tmp_in_cnnl_workspace.dptr();

    std::vector<int64_t> in_shapevec(
          {in_tensor->shape_view().At(0), in_tensor->shape_view().At(3), in_tensor->shape_view().At(1), in_tensor->shape_view().At(2)});
    auto transpose = NewPermutePrimitive(in_tensor->shape_view().NumAxes());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in_tensor->data_type(), in_tensor->shape_view().NumAxes(),
                    in_shapevec.data(), in_ptr,
                    std::vector<int>({0, 3, 1, 2}).data(), tmp_in_ptr);

    CnnlTensorDescriptor in_desc, out_decs;
    cnnlDataType_t cnnl_data_type = ConvertToCnnlDataType(GetDataType<T>::value);
    in_desc.set(in_tensor, CNNL_LAYOUT_NHWC, cnnl_data_type);
    out_decs.set(out_tensor, CNNL_LAYOUT_NHWC, cnnl_data_type);
    size_t _adaptive_avg_pool2d_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetAdaptivePoolingForwardWorkspaceSize(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                          in_desc.desc(), CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,out_decs.desc(),
                                          &_adaptive_avg_pool2d_workspace_size));
    CnnlWorkspace adaptive2d_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), _adaptive_avg_pool2d_workspace_size);
    void* _adaptive_avg_pool2d_workspace = adaptive2d_cnnl_workspace.dptr();
    size_t tmp_out_workspace_size = out_tensor->shape_view().elem_cnt();
    CnnlWorkspace tmp_out_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_out_workspace_size);
    void* tmp_out_ptr = tmp_out_cnnl_workspace.dptr();
    OF_CNNL_CHECK(cnnlAdaptivePoolingForward_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), 
                                                in_desc.desc(), 
                                                tmp_in_ptr, 
                                                CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 
                                                _adaptive_avg_pool2d_workspace,
                                                _adaptive_avg_pool2d_workspace_size,
                                                out_decs.desc(),
                                                tmp_out_ptr,
                                                NULL,
                                                NULL));
    std::vector<int64_t> out_shapevec(
          {out_tensor->shape_view().At(0), out_tensor->shape_view().At(1), out_tensor->shape_view().At(2), out_tensor->shape_view().At(3)});
    transpose = NewPermutePrimitive(out_tensor->shape_view().NumAxes());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in_tensor->data_type(), in_tensor->shape_view().NumAxes(),
                    out_shapevec.data(), tmp_out_ptr,
                    std::vector<int>({0, 3, 1, 2}).data(), out_ptr);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d").SetCreateFn<AdaptiveAvgPool2DKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                \
      && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(float)
REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(float16)

}  // namespace oneflow
