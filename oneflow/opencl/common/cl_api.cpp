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

#include <mutex>
#include "oneflow/opencl/common/cl_context.h"
#include "oneflow/opencl/common/cl_util.h"

namespace oneflow {

#define CL_CHECK_OR_RETURN(expr)           \
  {                                        \
    cl_int ret = (expr);                   \
    if (ret != CL_SUCCESS) { return ret; } \
  }

namespace {

cl_int clGetContext(clContext** context, int device_id) {
  return clContextPool::Get()->getOrCreateContext(context, device_id);
}

cl_int clGetDevices(cl::Device** devices, int* device_count) {
  return clContextPool::Get()->getDevices(devices, device_count);
}

static thread_local clContext* active_cl_context = nullptr;
static thread_local std::once_flag active_cl_context_inited_flag;

cl_int clGetActiveContext(clContext** context) {
  cl_int ret = CL_SUCCESS;
  std::call_once(active_cl_context_inited_flag,
                 [&]() { ret = clGetContext(&active_cl_context, 0); });
  *context = active_cl_context;
  return ret;
}

cl_int clSetActiveContext(clContext* context) {
  std::call_once(active_cl_context_inited_flag, [&]() { active_cl_context = context; });
  return CL_SUCCESS;
}

}  // namespace

cl_int clGetDeviceCount(int* count) { return clGetDevices(nullptr, count); }

cl_int clGetDevice(int* device_id) {
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetActiveContext(&context));
  *device_id = context->device_id;
  return CL_SUCCESS;
}

cl_int clSetDevice(int device_id) {
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetContext(&context, device_id));
  CL_CHECK_OR_RETURN(clSetActiveContext(context));
  return CL_SUCCESS;
}

cl_int clMalloc(void** buf, size_t size) {}

cl_int clFree(void* buf) {}

cl_int clMallocHost(void** buf, size_t size) {}
cl_int clFreeHost(void* buf) {}

cl_int clMemcpy(void* dst, const void* src, size_t size, MemcpyKind kind) {}
cl_int clMemcpyAsync(void* dst, const void* src, size_t size, MemcpyKind kind,
                     cl::CommandQueue* queue) {}

cl_int clMemset(void* ptr, int value, size_t size) {}
cl_int clMemsetAsync(void* ptr, int value, size_t size, cl::CommandQueue* queue) {}

cl_int clEventCreateWithFlags(cl::Event** event, unsigned int flags) {}
cl_int clEventDestroy(cl::Event* event) {}
cl_int clEventRecord(cl::Event* event, cl::CommandQueue* queue) {}
cl_int clEventQuery(cl::Event* event) {}
cl_int clEventSynchronize(cl::Event* event) {}

cl_int clQueueCreate(cl::CommandQueue** queue) {}
cl_int clQueueDestroy(cl::CommandQueue* queue) {}
cl_int clQueueSynchronize(cl::CommandQueue* queue) {}
cl_int clQueueWaitEvent(cl::Event* event, cl::CommandQueue* queue, unsigned int flags) {}

}  // namespace oneflow
