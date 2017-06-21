#ifndef ONEFLOW_CORE_KERNEL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_H_

#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_manager.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/blas/cblas_template.h"
#include "oneflow/blas/cublas_template.h"

namespace oneflow {

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel() = default;

  virtual void InitFromOpProto(const OperatorProto& op_proto);

  virtual void InitModelAndModelTmpBlobs(
      const KernelCtx& ctx,
      ParallelPolicy policy,
      uint64_t parallel_id,
      uint64_t parallel_num,
      const Snapshot*,
      std::function<Blob*(const std::string&)> Blob4BnInOp) const {
    UNEXPECTED_RUN();
  }

  // for Forward / Bp Calculation in FwExecGragh node and BpExecGragh node
  // through bn_in_op2blob_ptr function get the input blob and output blob
  // the Kernel will using the input blob calculate the result and fill output
  virtual void Forward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const = 0;
  virtual void Backward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const = 0;

  //
  const std::string& Lbn4BnInOp(const std::string& bn_in_op) const {
    return op_->Lbn4BnInOp(bn_in_op);
  }

 protected:
  Kernel() = default;
  const Operator* op() const { return op_.get(); }

 private:
  std::unique_ptr<const Operator> op_;
};

using KernelWardFunc = void (Kernel::*)(
    const KernelCtx&, std::function<Blob*(const std::string&)>) const;

#define INSTANTIATE_CPU_KERNEL_CLASS(classname) \
  char gInstantiationGuardCPU##classname; \
  template class classname<DeviceType::kCPU, float>; \
  template class classname<DeviceType::kCPU, double>;
#define INSTANTIATE_GPU_KERNEL_CLASS(classname) \
  char gInstantiationGuardGPU##classname; \
  template class classname<DeviceType::kGPU, float>; \
  template class classname<DeviceType::kGPU, double>;

}  // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_KERNEL_H_
