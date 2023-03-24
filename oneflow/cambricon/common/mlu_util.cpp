#include "oneflow/cambricon/common/mlu_util.h"

namespace oneflow{

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

} // oneflow