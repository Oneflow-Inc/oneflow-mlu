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
#include "oneflow/opencl/common/cl_api.h"

#include <vector>

namespace oneflow {

cl_int clGetDeviceCount(int* count) {
  cl_int err = CL_SUCCESS;
  cl::Platform platform = cl::Platform::getDefault(&err);
  if (err != CL_SUCCESS) { return err; }
  std::vector<cl::Device> gpuDevices;
  err = platform.getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);
  if (err != CL_SUCCESS) { return err; }
  *count = gpuDevices.size();
  return err;
}

cl_int clGetDevice(int* device_id) {}
cl_int clSetDevice(int device_id) {}

cl_int clMalloc(void** buf, size_t size) {}
cl_int clFree(void* buf) {}

cl_int clMallocHost(void** buf, size_t size) {}
cl_int clFreeHost(void* buf) {}

cl_int clMemcpy(void* dst, const void* src, size_t size, MemcpyKind kind) {}
cl_int clMemcpyAsync(void* dst, const void* src, size_t size, MemcpyKind kind, cl::CommandQueue* queue) {}

cl_int clMemset(void* ptr, int value, size_t size) {}
cl_int clMemsetAsync(void* ptr, int value, size_t size, cl::CommandQueue* queue) {}

cl_int clEventCreateWithFlags(cl::Event** event, unsigned int flags) {}
cl_int clEventDestroy(cl::Event* event) {}
cl_int clEventRecord(cl::Event* event, cl::CommandQueue* queue) {}

cl_int clQueueCreate(cl::CommandQueue** queue) {}
cl_int clQueueDestroy(cl::CommandQueue* queue) {}
cl_int clQueueSynchronize(cl::CommandQueue* queue) {}
cl_int clQueueWaitEvent(cl::Event* event, cl::CommandQueue* queue, unsigned int flags = 0) {}

}  // namespace oneflow
