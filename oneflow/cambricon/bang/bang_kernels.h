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

#include "oneflow/cambricon/bang/bang_handle.h"

namespace oneflow {

// input is a 3D tensor with shape [batch, N, length]
// indices is a 1D tensor with shape [index_size]
// output is a 3D tensor with shape [batch, index_size, length]
template<typename T, typename K>
void bang_gather_kernel(BangHandle& handle, const T* input, int64_t batch, int64_t N,
                        int64_t length, const K* index, int64_t index_size, T* output,
                        int64_t offset);

template<typename K>
void bang_gather_half_kernel(BangHandle& handle, const void* input, int64_t batch, int64_t N,
                             int64_t length, const K* index, int64_t index_size, void* output,
                             int64_t offset);

// input is a 3D tensor with shape [batch, segment_size, length]
// indices is a 1D tensor with shape [segment_size]
// output is a 3D tensor with shape [batch, N, length]
template<typename T, typename K>
void bang_unsorted_segment_sum_kernel(BangHandle& handle, const T* input, int64_t batch, int64_t N,
                                      int64_t length, const K* segment, int64_t segment_size,
                                      T* output, int64_t offset);

template<typename K>
void bang_unsorted_segment_sum_half_kernel(BangHandle& handle, const void* input, int64_t batch,
                                           int64_t N, int64_t length, const K* segment,
                                           int64_t segment_size, void* output, int64_t offset);

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_BANG_BANG_KERNELS_H_
