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
#pragma once

#include "oneflow/cambricon/bang/common_util.h"

namespace oneflow {
typedef void* MLUaddr;

void bang_fused_adam_internal(AddressList grad, AddressList m, AddressList v, AddressList variable,
                              SizeList sizes, int tensor_num, float beta1, float beta2,
                              float epsilon_correction, float learning_rate_correction,
                              int adam_mode, float decay, float decay_correction, cnrtDim3_t k_dim,
                              cnrtFunctionType_t k_type, cnrtQueue_t queue,
                              cnrtDataType_t cnrt_type);

}  // namespace oneflow
