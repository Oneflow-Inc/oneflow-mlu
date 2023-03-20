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
#ifndef ONEFLOW_CAMBRICON_EP_MLU_STREAM_H_
#define ONEFLOW_CAMBRICON_EP_MLU_STREAM_H_

#include "oneflow/cambricon/ep/mlu_device.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/vm/caching_allocator.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"

namespace oneflow {

namespace ep {

class MluDevice;

class MluStream : public Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MluStream);
  explicit MluStream(MluDevice* device);
  ~MluStream() override;

  static constexpr uint32_t kDefaultBlockSize = 256;

  DeviceType device_type() const override;
  MluDevice* device() const override;
  Maybe<void> Sync() override;
  void RecordEvent(Event* event) override;
  void WaitEvent(Event* event) override;
  Maybe<void> GetAsyncError() override;

  Maybe<void> OnExecutionContextSetup() override;
  Maybe<void> OnExecutionContextTeardown() override;

  cnrtQueue_t mlu_stream() const;
  cnnlHandle_t cnnl_handle() const;

  vm::CachingAllocator* workspace_allocator() { return workspace_allocator_.get(); }

  CnnlWorkspace NewCnnlWorkspace(size_t workspace_size = 0);

  template<typename Callable, typename... Args>
  const MluStream* Launch(Callable cnnl_func, Args... args) const {
    OF_CNNL_CHECK(std::invoke(cnnl_func, cnnl_handle(), args...));
    return this;
  }

  template<typename Callable, typename... Args>
  const MluStream* AsignWorkSpace(CnnlWorkspace& workspace, Callable cal_workspace_size,
                                  size_t& workspace_size, Args... args) const {
    OF_CNNL_CHECK(std::invoke(cal_workspace_size, cnnl_handle(), args..., &workspace_size));
    if (workspace.mlu_stream_ == nullptr) { workspace.mlu_stream_ = const_cast<MluStream*>(this); }
    workspace.resize(workspace_size);
    return this;
  }

  const MluStream* AsignWorkSpace(CnnlWorkspace& workspace, size_t workspace_size) const {
    if (workspace.mlu_stream_ == nullptr) { workspace.mlu_stream_ = const_cast<MluStream*>(this); }
    workspace.resize(workspace_size);
    return this;
  }

 private:
  cnrtQueue_t mlu_stream_{};
  int device_index_;
  MluDevice* device_;
  cnnlHandle_t cnnl_handle_;
  std::unique_ptr<vm::CachingAllocator> workspace_allocator_;
};

}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_EP_MLU_STREAM_H_
