#include "oneflow/cambricon/mlu/mlu_tools.h"

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

}  // namespace oneflow
