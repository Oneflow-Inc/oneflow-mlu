#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_types.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/cast.h"

namespace oneflow {

class MluBiasCorrectionFactorKernel final : public user_op::OpKernel,
                                            public user_op::CudaGraphSupport {
 public:
  MluBiasCorrectionFactorKernel() = default;
  ~MluBiasCorrectionFactorKernel() override = default;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* train_step = ctx->Tensor4ArgNameAndIndex("train_step", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto beta = ctx->Attr<float>("beta");

    std::array<int, 1> scalar_shape{1};
    const auto assign_desc = [&scalar_shape, dtype = ConvertToCnnlDataType(out->data_type())](
                                 CnnlTensorDescriptor& desc) {
      desc.set(1, scalar_shape.data(), dtype);
    };
    CnnlTensorDescriptor beta_desc, train_step_desc, out_desc;
    assign_desc(beta_desc);
    assign_desc(train_step_desc);
    assign_desc(out_desc);

    const auto stream = ctx->stream()->As<ep::MluStream>();
    const auto cnnl_handle = stream->cnnl_handle();

    const auto do_fill =
        ep::primitive::NewPrimitive<ep::primitive::FillFactory>(DeviceType::kMLU, DataType::kFloat);
    const auto do_cast = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
        DeviceType::kMLU, DataType::kInt64, DataType::kFloat);

    size_t scalar_size = GetSizeOfDataType(out->data_type()) * 2;
    CnnlWorkspace workspace_tmp(stream, scalar_size);

    void* beta_dptr = workspace_tmp.dptr();
    void* train_step_dptr =
        static_cast<char*>(workspace_tmp.dptr()) + sizeof(GetSizeOfDataType(out->data_type()));

    do_fill->Launch(stream, beta_dptr, beta, 1);
    do_cast->Launch(stream, train_step->dptr(), train_step_dptr, 1);

    size_t workspace_pow_size = 0;
    OF_CNNL_CHECK(cnnlGetPowWorkspaceSize(cnnl_handle, beta_desc.desc(), train_step_desc.desc(),
                                          out_desc.desc(), &workspace_pow_size));
    const float transform_alpha_before_pow(1.0), transform_beta_before_pow(1.0);
    OF_CNNL_CHECK(cnnlTransform_v2(
        cnnl_handle, CNNL_POINTER_MODE_HOST, &transform_alpha_before_pow, train_step_desc.desc(),
        train_step_dptr, &transform_beta_before_pow, train_step_desc.desc(), train_step_dptr));

    CnnlWorkspace workspace_pow(stream, workspace_pow_size);
    OF_CNNL_CHECK(cnnlPow(cnnl_handle, CNNL_COMPUTATION_FAST, beta_desc.desc(), beta_dptr,
                          train_step_desc.desc(), train_step_dptr, workspace_pow.dptr(),
                          workspace_pow_size, out_desc.desc(), out->mut_dptr()));
    const float transform_alpha(-1.0), transform_beta(1.0);
    OF_CNNL_CHECK(cnnlTransform_v2(cnnl_handle, CNNL_POINTER_MODE_HOST, &transform_alpha,
                                   out_desc.desc(), out->dptr(), &transform_beta, out_desc.desc(),
                                   out->mut_dptr()));
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MLU_ADAM_BIAS_CORRECTION_FACTOR_KERNEL \
  REGISTER_USER_KERNEL("adam_bias_correction_factor")   \
      .SetCreateFn<MluBiasCorrectionFactorKernel>()     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU));

REGISTER_MLU_ADAM_BIAS_CORRECTION_FACTOR_KERNEL

#undef REGISTER_MLU_ADAM_BIAS_CORRECTION_FACTOR_KERNEL

}  // namespace oneflow
