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
#include <cnnl.h>

#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/ep/include/primitive/blas.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

struct KernelLaunchConfig {
  const user_op::Tensor* a = nullptr;
  const user_op::Tensor* b = nullptr;
  user_op::Tensor* out = nullptr;
  // TODO(Jianhua Zheng): alpha and beta always using float?
  float alpha;
  float beta;
  ep::primitive::BlasTransposeType trans_a;
  ep::primitive::BlasTransposeType trans_b;
};

void LaunchBatchMatmulKernel(user_op::KernelComputeContext* ctx, const KernelLaunchConfig& conf) {
  CnnlMatmulDescriptor bmm_desc;
  bmm_desc.set_attr(CNNL_MATMUL_DESC_TRANSA, &conf.trans_a, sizeof(conf.trans_a));
  bmm_desc.set_attr(CNNL_MATMUL_DESC_TRANSB, &conf.trans_b, sizeof(conf.trans_b));


  // CnnlMatmulAlgoDescriptor algo_desc;
  cnnlMatMulAlgo_t matmul_algo_;
  OF_CNNL_CHECK(cnnlMatMulAlgoCreate(&matmul_algo_));


  CnnlTensorDescriptor a_desc, b_desc, out_desc;
  a_desc.set(conf.a);
  b_desc.set(conf.b);
  out_desc.set(conf.out);
  auto stream = ctx->stream()->As<ep::MluStream>();
  auto handle = stream->cnnl_handle();
  size_t workspace_size;
  // TODO(Jianhua Zheng): replace cnnlGetBatchMatMulBCastWorkspaceSize with
  // cnnlGetBatchMatMulHeuristicResult
  OF_CNNL_CHECK(cnnlGetBatchMatMulBCastWorkspaceSize(handle, a_desc.desc(), b_desc.desc(),
                                                     out_desc.desc(), &workspace_size));
  CnnlWorkspace workspace(stream, workspace_size);


  const void* alpha = &conf.alpha;
  const void* beta = &conf.beta;
  OF_CNNL_CHECK(cnnlBatchMatMulBCast_v2(handle, bmm_desc.desc(), matmul_algo_, alpha,
                                        a_desc.desc(), conf.a->dptr(), b_desc.desc(),
                                        conf.b->dptr(), beta, out_desc.desc(),
                                        conf.out->mut_dptr(), workspace.dptr(), workspace_size));
  
  // TODO: remove
  OF_CNNL_CHECK(cnnlMatMulAlgoDestroy(matmul_algo_));
}

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

template<typename Context>
ep::primitive::BlasTransposeType GetBlasTransposeType(Context* ctx, const std::string& attr) {
  return GetBlasTransposeType(ctx->template Attr<bool>(attr));
}

template<typename Context>
std::unique_ptr<ep::primitive::Memcpy> NewMemcpyPrimitive(Context* ctx) {
  return ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(
      ctx->device_type(), ep::primitive::MemcpyKind::kDtoD);
}

template<typename T>
class BatchMatmulKernelMLU final : public user_op::OpKernel {
 public:
  BatchMatmulKernelMLU() = default;
  ~BatchMatmulKernelMLU() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    KernelLaunchConfig conf;
    conf.trans_a = GetBlasTransposeType(ctx, "transpose_a");
    conf.trans_b = GetBlasTransposeType(ctx, "transpose_b");
    conf.a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const DataType data_type = conf.a->data_type();
    const int64_t num_axes = conf.a->shape_view().NumAxes();
    CHECK_GT(num_axes, 2);
    conf.b = ctx->Tensor4ArgNameAndIndex("b", 0);
    CHECK_EQ(conf.b->data_type(), data_type);
    CHECK_EQ(conf.b->shape_view().NumAxes(), num_axes);
    conf.out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(conf.out->data_type(), data_type);
    CHECK_EQ(conf.out->shape_view().NumAxes(), num_axes);
    conf.alpha = ctx->Attr<double>("alpha");
    conf.beta = 0.0;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), data_type);
      CHECK_EQ(add_to_output->shape_view(), conf.out->shape_view());
      auto memcpy = NewMemcpyPrimitive(ctx);
      CHECK(memcpy);
      memcpy->Launch(ctx->stream(), conf.out->mut_dptr(), add_to_output->dptr(),
                     add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(data_type));
      conf.beta = 1.0;
    }
    LaunchBatchMatmulKernel(ctx, conf);
  }
};

#define REGISTER_BATCH_MATMUL_USER_KERNEL(op_name, dtype)                                   \
  REGISTER_USER_KERNEL(op_name)                                                             \
      .SetCreateFn<BatchMatmulKernelMLU<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                       \
                       && (user_op::HobDataType("a", 0) == user_op::HobDataType("b", 0))    \
                       && (user_op::HobDataType("a", 0) == user_op::HobDataType("out", 0))  \
                       && (user_op::HobDataType("a", 0) == GetDataType<dtype>::value))      \
      .SetInplaceProposalFn(                                                                \
          [](const user_op::InferContext& ctx,                                              \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {        \
            if (ctx.has_input("_add_to_output", 0)) {                                       \
              OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true)); \
            }                                                                               \
            return Maybe<void>::Ok();                                                       \
          });

REGISTER_BATCH_MATMUL_USER_KERNEL("batch_matmul", float)

}  // namespace

}  // namespace oneflow
