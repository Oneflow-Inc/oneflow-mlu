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
#ifndef ONEFLOW_CAMBRICON_CNNL_CNNL_TENSOR_DESCRIPTOR_H_
#define ONEFLOW_CAMBRICON_CNNL_CNNL_TENSOR_DESCRIPTOR_H_

#include "oneflow_mlu/cnnl/cnnl_common_descriptor.h"
#include "oneflow_mlu/cnnl/cnnl_types.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/user_op_tensor.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlTensorDescriptors.h

namespace oneflow {

class CnnlTensorDescriptor : public CnnlDescriptor<cnnlTensorStruct, &cnnlCreateTensorDescriptor,
                                                   &cnnlDestroyTensorDescriptor> {
 public:
  // Init create Tensor descriptor
  CnnlTensorDescriptor() = default;
  CnnlTensorDescriptor(const user_op::Tensor* t) { set(t); }  // NOLINT
  CnnlTensorDescriptor(const user_op::Tensor* t, cnnlTensorLayout_t layout,
                       cnnlDataType_t data_type = CNNL_DTYPE_INVALID) {
    set(t, layout, data_type);
  }
  CnnlTensorDescriptor(const user_op::Tensor* t, cnnlDataType_t dtype) { set(t, dtype); }
  CnnlTensorDescriptor(const user_op::Tensor* t, bool keep_dim, std::vector<int64_t>& keepdim_sizes,
                       cnnlDataType_t dtype = CNNL_DTYPE_INVALID) {
    set(t, keep_dim, keepdim_sizes, dtype);
  }
  // set descriptor from tensor
  void set(const user_op::Tensor* t);
  void set(const user_op::Tensor* t, cnnlTensorLayout_t layout,
           cnnlDataType_t data_type = CNNL_DTYPE_INVALID);
  void set(const user_op::Tensor* t, cnnlDataType_t dtype);
  void set_onchip_dtype(cnnlDataType_t data_type);
  void set(int position = 0, float scale = 1.0);

  void set_dim(const user_op::Tensor* t);
  void set_dim(const user_op::Tensor* t, int inputDim);

  void set_reduce(const user_op::Tensor* t);
  void set_reduce(const user_op::Tensor* t, std::vector<int64_t> keepdim);

  // for setting pooling output tensor descriptor.
  void set(const user_op::Tensor* t, bool keep_dim, std::vector<int64_t>& keepdim_sizes,
           cnnlDataType_t dtype = CNNL_DTYPE_INVALID);

  // assigned a special shape, not use tensor shape and stride info.
  void set_additional_dim(const user_op::Tensor* t, std::vector<int>& dims);

  // set_additional_dim will change layout (i.e. NCHW -> NHWC)
  // `set_reshape` only modify shape and stride
  void set_reshape(const user_op::Tensor* t, const std::vector<int>& dims);

  template<typename T>
  void set(const user_op::Tensor* t, const std::vector<T>& shape_info,
           const std::vector<T>& stride_info, cnnlTensorLayout_t layout,
           cnnlDataType_t data_type = CNNL_DTYPE_INVALID) {
    CHECK_EQ_OR_THROW(shape_info.size(), stride_info.size())
        << "shape size need equal to stride size.";
    int t_dim = shape_info.size();
    // data_type default value is CNNL_DTYPE_INVALID in this interface,
    // and can't transmit to cnnl. so call cnnl interface will using
    // tensor dtype value when data_type value is default.
    if (data_type == CNNL_DTYPE_INVALID) { data_type = ConvertToCnnlDataType(t->data_type()); }
    if (!t_dim) {
      t_dim = 1;
      std::vector<int> dim_array(1, 1);
      OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim,
                                              dim_array.data(), dim_array.data()));
      return;
    }
    std::vector<int> real_shape_info(t_dim);
    std::vector<int> real_stride_info(t_dim);
    for (int i = 0; i < t_dim; ++i) {
      real_shape_info[i] = static_cast<int>(shape_info[i]);
      real_stride_info[i] = static_cast<int>(stride_info[i]);
    }
    OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), layout, data_type, t_dim,
                                            real_shape_info.data(), real_stride_info.data()));
  }

  template<typename T>
  void set(int ndim, const T* shape, cnnlDataType_t data_type,
           cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY) {
    if (!ndim) {
      ndim = 1;
      std::vector<int> shape_info(1, 1);
      OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, ndim,
                                              shape_info.data(), shape_info.data()));
      return;
    }
    std::vector<int> shape_info(ndim, 1);
    std::vector<int> stride_info(ndim, 1);
    int value = 1;
    for (size_t i = ndim - 1; i > 0; --i) {
      shape_info[i] = static_cast<int>(shape[i]);
      stride_info[i] = value;
      value *= shape_info[i];
    }
    shape_info[0] = static_cast<int>(shape[0]);
    stride_info[0] = value;
    OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), layout, data_type, ndim,
                                            shape_info.data(), stride_info.data()));
  }

  template<typename T>
  void set(int ndim, const T* shape, const T* stride, cnnlDataType_t data_type,
           cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY) {
    if (!ndim) {
      ndim = 1;
      std::vector<int> shape_info(1, 1);
      OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, ndim,
                                              shape_info.data(), shape_info.data()));
      return;
    }
    std::vector<int> shape_info(ndim, 1);
    std::vector<int> stride_info(ndim, 1);
    for (int i = 0; i < ndim; ++i) {
      shape_info[i] = static_cast<int>(shape[i]);
      stride_info[i] = static_cast<int>(stride[i]);
    }
    OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), layout, data_type, ndim,
                                            shape_info.data(), stride_info.data()));
  }
};

class CnnlSeqDataDescriptor : public CnnlDescriptor<cnnlSeqDataStruct, &cnnlCreateSeqDataDescriptor,
                                                    &cnnlDestroySeqDataDescriptor> {
 public:
  CnnlSeqDataDescriptor() {}

  void set(const user_op::Tensor* t);
  void set(const user_op::Tensor* t, cnnlSeqDataLayout_t layout);
  void set_onchip_dtype(cnnlDataType_t onchip_dtype);
};

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_CNNL_CNNL_TENSOR_DESCRIPTOR_H_
