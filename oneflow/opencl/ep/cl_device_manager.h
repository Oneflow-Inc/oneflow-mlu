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
#ifndef ONEFLOW_COMBRICON_EP_CL_DEVICE_MANAGER_H_
#define ONEFLOW_COMBRICON_EP_CL_DEVICE_MANAGER_H_

#include "oneflow/core/ep/include/device_manager.h"

namespace oneflow {
namespace ep {

class OclDevice;

class OclDeviceManager : public DeviceManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OclDeviceManager);
  OclDeviceManager(DeviceManagerRegistry* registry);
  ~OclDeviceManager() override;

  DeviceManagerRegistry* registry() const override;
  std::shared_ptr<Device> GetDevice(size_t device_index) override;
  size_t GetDeviceCount(size_t primary_device_index) override;
  size_t GetDeviceCount() override;
  size_t GetActiveDeviceIndex() override;
  void SetActiveDeviceByIndex(size_t device_index) override;
  bool IsDeviceStreamWaitEventSupported() const override { return true; }

  std::shared_ptr<RandomGenerator> CreateRandomGenerator(uint64_t seed,
                                                         size_t device_index) override;

 private:
  std::mutex devices_mutex_;
  std::vector<std::shared_ptr<OclDevice>> devices_;
  DeviceManagerRegistry* registry_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_COMBRICON_EP_CL_DEVICE_MANAGER_H_
