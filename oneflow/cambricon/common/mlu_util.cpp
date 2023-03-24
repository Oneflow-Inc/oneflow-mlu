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
#include "oneflow/cambricon/common/mlu_util.h"

namespace oneflow {

uint32_t getDeviceAttr(cnrtDeviceAttr_t attr) {
  int dev_ordinal = 0;
  int device_attr = 1;
  OF_MLU_CHECK(cnrtGetDevice(&dev_ordinal));
  OF_MLU_CHECK(cnrtDeviceGetAttribute(&device_attr, attr, dev_ordinal));
  if (attr == cnrtAttrNramSizePerMcore) { device_attr -= rem_for_stack; }
  return device_attr;
}

cnrtDataType_t fromCnnlType2CnrtType(cnnlDataType_t cnnl_data_type) {
  switch (cnnl_data_type) {
    case CNNL_DTYPE_HALF: return CNRT_FLOAT16;
    case CNNL_DTYPE_FLOAT: return CNRT_FLOAT32;
    case CNNL_DTYPE_INT32: return CNRT_INT32;
    case CNNL_DTYPE_INT8: return CNRT_INT8;
    case CNNL_DTYPE_UINT8: return CNRT_UINT8;
    case CNNL_DTYPE_INT16: return CNRT_INT16;
    case CNNL_DTYPE_BOOL: return CNRT_BOOL;
    default: THROW(RuntimeError) << "Invalid data type from cnnl to cnrt!"; return CNRT_INVALID;
  }
}

}  // namespace oneflow
