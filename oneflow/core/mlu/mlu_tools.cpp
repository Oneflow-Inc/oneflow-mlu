#include "oneflow/core/mlu/mlu_tools.h"

namespace oneflow {

void setTensorDesc(cnnlTensorDescriptor_t &desc,
                   Shape2D shape,
                   cnnlDataType_t dtype,
                   cnnlTensorLayout_t layout) {
  int dim[4];
  if (layout == CNNL_LAYOUT_NHWC) {
    dim[0] = shape.n;
    dim[1] = shape.h;
    dim[2] = shape.w;
    dim[3] = shape.c;
  } else if (layout == CNNL_LAYOUT_NCHW) {
    dim[0] = shape.n;
    dim[1] = shape.c;
    dim[2] = shape.h;
    dim[3] = shape.w;
  } else if (layout == CNNL_LAYOUT_HWCN) {
    dim[0] = shape.h;
    dim[1] = shape.w;
    dim[2] = shape.c;
    dim[3] = shape.n;
  } else {
    ERROR("unsupport data layout!");
  }
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 4, dim));
}

cnnlDataType_t convertCamDataType(CamDataType type){
  int v = 0;
  if(type == kHALF){
    v = 1;
  }
  else if(type == kFLOAT32){
    v = 2;
  }
  else if(type == kINT8){
    v = 3;
  }
  else if(type == kINT16){
    v = 4;
  }
  return (cnnlDataType_t)v;
}

}  // namespace oneflow
