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
#include "oneflow/core/ep/include/primitive/memcpy.h"

#include "oneflow_mlu/common/mlu_util.h"
#include "oneflow_mlu/ep/mlu_stream.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace {

class MemcpyImpl : public Memcpy {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemcpyImpl);
  MemcpyImpl(MemcpyKind kind) {
    switch (kind) {
      case MemcpyKind::kHtoD: kind_ = cnrtMemcpyHostToDev; break;
      case MemcpyKind::kDtoH: kind_ = cnrtMemcpyDevToHost; break;
      case MemcpyKind::kDtoD: kind_ = cnrtMemcpyDevToDev; break;
      default: UNIMPLEMENTED();
    }
  }
  ~MemcpyImpl() override = default;

  void Launch(Stream* stream, void* dst, const void* src, size_t count) override {
    if (dst == src) { return; }
    auto* mlu_stream = stream->As<MluStream>();
    OF_MLU_CHECK(
        cnrtMemcpyAsync(dst, const_cast<void*>(src), count, mlu_stream->mlu_stream(), kind_));
    // Synchronous the stream since the host memory may not be page-locked, and cnrtMemcpyAsync will
    // not translate to synchronous automatically like cuda.
    if (kind_ != cnrtMemcpyDevToDev) { mlu_stream->Sync(); }
  }

 private:
  cnrtMemTransDir_t kind_;
};

class MemcpyFactoryImpl : public MemcpyFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemcpyFactoryImpl);
  MemcpyFactoryImpl() = default;
  ~MemcpyFactoryImpl() override = default;

  std::unique_ptr<Memcpy> New(MemcpyKind kind) override {
    return std::unique_ptr<Memcpy>(new MemcpyImpl(kind));
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kMLU, MemcpyFactory, MemcpyFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
