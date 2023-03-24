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
#ifndef ONEFLOW_CAMBRICON_BANG_BANG_KERNELS_H_
#define ONEFLOW_CAMBRICON_BANG_BANG_KERNELS_H_

#include <stdint.h>

#include "cnrt.h"

namespace oneflow {

typedef struct BangHandle {
  cnrtQueue_t queue;
  uint32_t nclusters;
  uint32_t ncores_per_cluster;

  BangHandle(cnrtQueue_t q, int nclusters, int ncores)
      : queue(q), nclusters(nclusters), ncores_per_cluster(ncores) {}
} BangHandle;

// input is a 3D tensor with shape is [batch, N, length]
// indices is a 1D tensor with shape is [index_size]
template<typename T, typename K>
void bang_gather_kernel(BangHandle& handle, const T* input, int64_t batch, int64_t N,
                        int64_t length, const K* index, int64_t index_size, T* output,
                        int64_t offset);

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_BANG_BANG_KERNELS_H_
