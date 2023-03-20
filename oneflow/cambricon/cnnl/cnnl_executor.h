#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/core/common/throw.h"

namespace oneflow {

template<size_t N_WORKSPACE = 0>
class CnnlExecutor {
 public:
  CnnlExecutor() = delete;
  explicit CnnlExecutor(ep::Stream* stream) : CnnlExecutor(stream->As<ep::MluStream>()) {}
  explicit CnnlExecutor(ep::MluStream* stream) : mlu_stream_(stream) {
    for (auto& x : workspaces_) { x = std::make_shared<CnnlWorkspace>(stream); }
  }

  template<typename Callable, typename... Args>
  CnnlExecutor& Launch(Callable cnnl_func, Args... args) {
    OF_CNNL_CHECK(std::invoke(cnnl_func, mlu_stream_->cnnl_handle(), args...));
    return *this;
  }

  template<typename Callable, typename... Args>
  CnnlExecutor& AllocWorkSpace(const size_t workspace_index, Callable cal_workspace_size,
                               size_t& workspace_size, Args... args) {
    OF_CNNL_CHECK(
        std::invoke(cal_workspace_size, mlu_stream_->cnnl_handle(), args..., &workspace_size));
    return AllocWorkSpace(workspace_index, workspace_size);
  }

  CnnlExecutor& AllocWorkSpace(const size_t workspace_index, size_t workspace_size) {
    CHECK_LT_OR_THROW(workspace_index, N_WORKSPACE)
        << "Current CNNL executor only has " << N_WORKSPACE << " workspaces.";
    workspaces_[workspace_index]->resize(workspace_size);
    return *this;
  }

  void* GetWorkSpace(const size_t workspace_index) {
    CHECK_LT_OR_THROW(workspace_index, N_WORKSPACE)
        << "Current CNNL executor only has " << N_WORKSPACE << " workspaces.";

    return workspaces_[workspace_index]->dptr();
  }

 private:
  ep::MluStream* mlu_stream_ = nullptr;
  std::array<std::shared_ptr<CnnlWorkspace>, N_WORKSPACE> workspaces_;
};

}  // namespace oneflow