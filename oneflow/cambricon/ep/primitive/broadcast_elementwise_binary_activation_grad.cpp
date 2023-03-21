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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/ep/primitive/broadcast_elementwise_binary.h"
#include "oneflow/cambricon/ep/primitive/type_seq.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_binary.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace mlu {

namespace {

template<typename T>
static void SetCnnlTensorDescriptor(CnnlTensorDescriptor& desc, size_t num_dims,
                                    const int64_t* dims) {
  cnnlDataType_t data_type = ConvertToCnnlDataType(GetDataType<T>::value);
  if (!num_dims) {
    num_dims = 1;
    std::vector<int> dim_array(1, 1);
    OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(desc.mut_desc(), CNNL_LAYOUT_ARRAY, data_type, num_dims,
                                            dim_array.data(), dim_array.data()));
    return;
  }
  std::vector<int> shape_info(num_dims);
  for (size_t i = 0; i < num_dims; ++i) { shape_info[i] = static_cast<int>(dims[i]); }
  OF_CNNL_CHECK(cnnlSetTensorDescriptor(desc.mut_desc(), CNNL_LAYOUT_ARRAY, data_type, num_dims,
                                        shape_info.data()));
}

template<BinaryOp op>
cnnlActivationMode_t GetCnnlActivationMode();

template<>
cnnlActivationMode_t GetCnnlActivationMode<BinaryOp::kGeluBackwardWithDyX>() {
  return CNNL_ACTIVATION_GELU;
}

template<>
cnnlActivationMode_t GetCnnlActivationMode<BinaryOp::kReluBackwardWithDyY>() {
  return CNNL_ACTIVATION_RELU;
}

}  // namespace

template<BinaryOp op, typename T>
class BinaryActivationBackwardGrad : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryActivationBackwardGrad);
  BinaryActivationBackwardGrad() = default;
  ~BinaryActivationBackwardGrad() override = default;

  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1, void* dst) {
    // when op is relu, src0 is diff_y, src1 is y.
    // when op is gelu, src0 is diff_y, src1 is x.
    CnnlTensorDescriptor diff_y_desc;
    SetCnnlTensorDescriptor<T>(diff_y_desc, num_src0_dims, src0_dims);
    CnnlTensorDescriptor x_desc;
    SetCnnlTensorDescriptor<T>(x_desc, num_src1_dims, src1_dims);
    CnnlTensorDescriptor diff_x_desc;
    SetCnnlTensorDescriptor<T>(diff_x_desc, num_src1_dims, src1_dims);

    auto handle = stream->As<ep::MluStream>()->cnnl_handle();
    CnnlActivationDescriptor activation_desc;
    activation_desc.set(GetCnnlActivationMode<op>(), CNNL_ACTIVATION_HIGH_PRECISION,
                        CNNL_NOT_PROPAGATE_NAN, /*coef*/ 0.0);
    OF_CNNL_CHECK(cnnlActivationBackward(handle, activation_desc.desc(),
                                         /*alpha*/ nullptr,
                                         /*y_desc*/ nullptr,
                                         /*y*/ nullptr, diff_y_desc.desc(),
                                         /*diff_y*/ src0, x_desc.desc(),
                                         /*x. when op=relu_grad, replace x with y*/ src1,
                                         /*beta*/ nullptr, diff_x_desc.desc(),
                                         /*diff_x*/ dst));
  }

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1, void* dst) {
    UNIMPLEMENTED();
  }

  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              Scalar src1, void* dst) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_GRAD_ENTRY(binary_op, src_data_type_pair) \
  template<>                                                                                   \
  std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary<                   \
      binary_op, OF_PP_PAIR_FIRST(src_data_type_pair), OF_PP_PAIR_FIRST(src_data_type_pair)>(  \
      Scalar attr0, Scalar attr1) {                                                            \
    return std::unique_ptr<BroadcastElementwiseBinary>(                                        \
        new BinaryActivationBackwardGrad<binary_op, OF_PP_PAIR_FIRST(src_data_type_pair)>);    \
  }

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_GRAD_ENTRY,
                                 MLU_BINARY_ACTIVATION_BACKWARD_OP_SEQ,
                                 MLU_PRIMITIVE_ACTIVATION_BACKWARD_GRAD_TYPE_SEQ);

#undef INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_GRAD_ENTRY

}  // namespace mlu
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
