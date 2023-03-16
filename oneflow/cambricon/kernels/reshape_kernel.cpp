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
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

class MluReshapeKernel final : public user_op::OpKernel {
 public:
  MluReshapeKernel() = default;
  ~MluReshapeKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t elem_cnt = in->shape_view().elem_cnt();
    CHECK_GE(elem_cnt, 0);
    // For 0-size tensor, we don't need to copy data, but we must
    // fill output tensor with Scalar(0) because during the backward propogation, this kernel will
    // also be used.
    if (elem_cnt == 0) {
      // TODO: fill primitive
      printf("\n TODO:use fill primitive");
      return;
    }

    CHECK_EQ(out->shape_view().elem_cnt(), elem_cnt);
    CHECK_EQ(in->data_type(), out->data_type());
    std::unique_ptr<ep::primitive::Memcpy> primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(ctx->stream()->device_type(),
                                                                  ep::primitive::MemcpyKind::kDtoD);
    CHECK(primitive) << "Can not create Memcpy primitive for device type "
                     << ctx->stream()->device_type();
    primitive->Launch(ctx->stream(), out->mut_dptr(), in->dptr(),
                      elem_cnt * GetSizeOfDataType(in->data_type()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_COPY_DATA_CONTENT_KERNEL(op_type_name)                              \
  REGISTER_USER_KERNEL(op_type_name)                                                 \
      .SetCreateFn<MluReshapeKernel>()                                               \
      .SetInplaceProposalFn(                                                         \
          [](const user_op::InferContext&,                                           \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, false));          \
            return Maybe<void>::Ok();                                                \
          });

REGISTER_COPY_DATA_CONTENT_KERNEL("reshape")

}  // namespace oneflow
