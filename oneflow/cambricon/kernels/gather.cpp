#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

class MluGatherKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MluGatherKernel() = default;
  ~MluGatherKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t axis = ctx->Attr<int64_t>("axis");

    if (out->shape_view().elem_cnt() == 0) { return; }

    const Shape in_shape = ExpandDimIf0D(in->shape_view());
    std::vector<int> in_shape_vec;
    for (const auto x : in_shape.dim_vec()) { in_shape_vec.push_back(static_cast<int>(x)); }

    CnnlTensorDescriptor in_desc(in), indices_desc(indices), out_desc(out);
    OF_CNNL_CHECK(cnnlIndexSelect(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), axis,
                                  in_desc.desc(), in->dptr(), indices_desc.desc(), indices->dptr(),
                                  out_desc.desc(), out->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("gather").SetCreateFn<MluGatherKernel>().SetIsMatchedHob(
    (user_op::HobDeviceType() == DeviceType::kMLU));

}  // namespace oneflow