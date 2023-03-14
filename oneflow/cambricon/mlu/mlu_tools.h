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
#ifndef ONEFLOW_CORE_KERNEL_MLU_TOOLS_H_
#define ONEFLOW_CORE_KERNEL_MLU_TOOLS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

#include "oneflow/cambricon/mlu/public.h"
#include "cnrt.h"
#include "cnnl.h"

namespace oneflow {

struct Shape2D {
  int n = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  inline int size() { return n * c * h * w; }
};

struct AddType{
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

void setTensorDesc2D(cnnlTensorDescriptor_t &desc,
                     Shape2D shape,
                     cnnlDataType_t dtype,
                     cnnlTensorLayout_t layout);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MLU_TOOLS_H_
