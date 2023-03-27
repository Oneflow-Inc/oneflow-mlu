#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_types.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/cast.h"

namespace oneflow {

template<typename T>
class MluMultiReduceSumPowAbsKernel final : public user_op::OpKernel,
                                            public user_op::CudaGraphSupport {
 public:
  MluMultiReduceSumPowAbsKernel() = default;
  ~MluMultiReduceSumPowAbsKernel() override = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache*) const override {
    using T_FOR_CNNL = float;
    const auto cnnl_type = ConvertToCnnlDataType(DataType::kFloat);

    const auto stream = ctx->stream()->As<ep::MluStream>();
    const auto cnnl_handle = stream->cnnl_handle();

    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    size_t num_inputs = ctx->input_size("x");

    CnnlReduceDescriptor reduce_op_desc;
    reduce_op_desc.set(cnnl_type, {0}, CNNL_REDUCE_ADD, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);

    CnnlTensorDescriptor reduce_out_desc;
    const int64_t num_reduce_out = 1;
    reduce_out_desc.set(1, &num_reduce_out, cnnl_type);

    CnnlTensorDescriptor final_reduce_in_desc;
    final_reduce_in_desc.set(1, &num_inputs, cnnl_type);

    CnnlWorkspace workspace_for_final_reduce_in(stream, sizeof(T_FOR_CNNL) * num_inputs);

    const auto get_x_ptr = [&workspace_for_final_reduce_in](size_t i) -> void* {
      return static_cast<char*>(workspace_for_final_reduce_in.dptr()) + (i * (sizeof(T_FOR_CNNL)));
    };

    const auto get_reduce_workspace_size = [&](const CnnlTensorDescriptor& desc) {
      size_t workspace_size_for_reduce = 0;
      OF_CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(cnnl_handle, desc.desc(), reduce_out_desc.desc(),
                                                 reduce_op_desc.mut_desc(),
                                                 &workspace_size_for_reduce));
      return workspace_size_for_reduce;
    };

    const float p = ctx->Attr<float>("p");

    for (size_t i = 0; i < num_inputs; ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", i);
      const int64_t x_num_elements = x->shape_view().Count(0);

      // cast x to float if x is double
      const void* x_ptr = x->dptr();
      CnnlWorkspace workspace(stream, sizeof(T_FOR_CNNL) * x_num_elements * 2);
      if constexpr (std::is_same_v<T, double>) {
        const auto do_cast = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
            DeviceType::kMLU, DataType::kDouble, DataType::kFloat);
        do_cast->Launch(stream, x->dptr(), workspace.dptr(), x_num_elements);
        x_ptr = workspace.dptr();
      }
      void* p_ptr = static_cast<char*>(workspace.dptr()) + (sizeof(T_FOR_CNNL) * x_num_elements);
      const auto do_fill = ep::primitive::NewPrimitive<ep::primitive::FillFactory>(
          DeviceType::kMLU, DataType::kFloat);
      do_fill->Launch(stream, p_ptr, p, x_num_elements);
      CnnlTensorDescriptor x_desc;
      x_desc.set(1, &x_num_elements, cnnl_type);

      OF_CNNL_CHECK(cnnlAbs(cnnl_handle, x_desc.desc(), x_ptr, x_desc.desc(), workspace.dptr()));

      size_t workspace_size_for_pow = 0;
      OF_CNNL_CHECK(cnnlGetPowWorkspaceSize(cnnl_handle, x_desc.desc(), x_desc.desc(),
                                            x_desc.desc(), &workspace_size_for_pow));
      const size_t workspace_size_for_reduce = get_reduce_workspace_size(x_desc);

      CnnlWorkspace workspace_for_pow_and_reduce(
          stream, std::max(workspace_size_for_pow, workspace_size_for_reduce));
      OF_CNNL_CHECK(cnnlPow(cnnl_handle, CNNL_COMPUTATION_HIGH_PRECISION, x_desc.desc(),
                            workspace.dptr(), x_desc.desc(), p_ptr,
                            workspace_for_pow_and_reduce.dptr(), workspace_size_for_pow,
                            x_desc.desc(), workspace.dptr()));

      OF_CNNL_CHECK(cnnlReduce(cnnl_handle, reduce_op_desc.desc(),
                               workspace_for_pow_and_reduce.dptr(), workspace_size_for_reduce,
                               nullptr, x_desc.desc(), workspace.dptr(), 0, nullptr, nullptr,
                               reduce_out_desc.desc(), get_x_ptr(i)));
    }
    size_t workspace_size_for_final_reduce = get_reduce_workspace_size(final_reduce_in_desc);
    CnnlWorkspace workspace_for_final_reduce(stream,
                                             workspace_size_for_final_reduce + sizeof(T_FOR_CNNL));
    void* out_ptr = y->mut_dptr();
    if constexpr (std::is_same_v<T, double>) {
      out_ptr =
          static_cast<char*>(workspace_for_final_reduce.dptr()) + workspace_size_for_final_reduce;
    }
    OF_CNNL_CHECK(cnnlReduce(cnnl_handle, reduce_op_desc.desc(), workspace_for_final_reduce.dptr(),
                             workspace_size_for_final_reduce, nullptr, final_reduce_in_desc.desc(),
                             workspace_for_final_reduce_in.dptr(), 0, nullptr, nullptr,
                             reduce_out_desc.desc(), out_ptr));
    if constexpr (std::is_same_v<T, double>) {
      const auto do_cast = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
          DeviceType::kMLU, DataType::kFloat, DataType::kDouble);
      do_cast->Launch(stream, out_ptr, y->mut_dptr(), 1);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_MULTI_REDUCE_SUM_POW_ABS_KERNEL(dtype)       \
  REGISTER_USER_KERNEL("multi_reduce_sum_pow_abs")                    \
      .SetCreateFn<MluMultiReduceSumPowAbsKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_MLU_MULTI_REDUCE_SUM_POW_ABS_KERNEL(float)
REGISTER_MLU_MULTI_REDUCE_SUM_POW_ABS_KERNEL(double)

}  // namespace oneflow