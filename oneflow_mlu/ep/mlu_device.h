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
#ifndef ONEFLOW_CORE_CAMBRICON_EP_MLU_DEVICE_H_
#define ONEFLOW_CORE_CAMBRICON_EP_MLU_DEVICE_H_

#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {
namespace ep {

class MluDevice : public Device {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MluDevice);
  explicit MluDevice(int device_index, DeviceManager* device_manager);
  ~MluDevice() override;

  void SetAsActiveDevice() override;
  void Reset() override;

  DeviceType device_type() const override { return DeviceType::kMLU; }
  size_t device_index() const override { return device_index_; }
  DeviceManager* device_manager() const override { return device_manager_; }

  int nclusters() const { return nclusters_; }
  int ncores_per_cluster() const { return ncores_per_cluster_; }

  Stream* CreateStream() override;
  void DestroyStream(Stream* stream) override;

  void CreateEvents(Event** events, size_t count) override;
  void DestroyEvents(Event** events, size_t count) override;

  Maybe<void> Alloc(const AllocationOptions& options, void** ptr, size_t size) override;
  void Free(const AllocationOptions& options, void* ptr) override;
  Maybe<void> AllocPinned(const AllocationOptions& options, void** ptr, size_t size) override;
  void FreePinned(const AllocationOptions& options, void* ptr) override;

  const void* GetConstZeros(DataType data_type, size_t n) const;
  const void* GetConstOnes(DataType data_type, size_t n) const;

 private:
  int device_index_;
  std::mutex events_mutex_;
  std::vector<Event*> events_;
  unsigned int event_flags_;
  DeviceManager* device_manager_;
  int nclusters_;
  int ncores_per_cluster_;
  int64_t const_buf_elem_cnt_;
  void* const_zeros_buffer_;
  void* const_ones_buffer_fp32_;
};

}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_CAMBRICON_EP_MLU_DEVICE_H_
