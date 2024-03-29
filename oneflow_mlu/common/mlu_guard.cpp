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
#include "oneflow_mlu/common/mlu_guard.h"

#include "oneflow_mlu/common/mlu_util.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

static const char* MluNotSupportInForkedSubProcess =
    "Cannot re-initialize MLU in forked subprocess. To use MLU with multiprocessing, you "
    "must add 'multiprocessing.set_start_method(\"spawn\")' in '__main__' if you are using "
    "Python's multiprocessing";

MluCurrentDeviceGuard::MluCurrentDeviceGuard(int32_t dev_id) {
  CHECK(!pthread_fork::IsForkedSubProcess()) << MluNotSupportInForkedSubProcess;
  OF_MLU_CHECK(cnrtGetDevice(&saved_dev_id_));
  OF_MLU_CHECK(cnrtSetDevice(dev_id));
}

MluCurrentDeviceGuard::MluCurrentDeviceGuard() { OF_MLU_CHECK(cnrtGetDevice(&saved_dev_id_)); }

MluCurrentDeviceGuard::~MluCurrentDeviceGuard() { OF_MLU_CHECK(cnrtSetDevice(saved_dev_id_)); }

}  // namespace oneflow
