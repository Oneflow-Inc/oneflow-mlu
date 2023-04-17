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
#include <string>
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/kernels/convert_memory_format_util.h"

namespace oneflow {

namespace {

struct TensorInfo {
  TensorInfo(cnnlTensorDescriptor_t desc, void* ptr) : tensor_desc(desc), dptr(ptr) {}
  cnnlTensorDescriptor_t tensor_desc;
  void* dptr;
};

template<typename T, cnnlPoolingMode_t>
struct GetIndexTensorInfoForward;

template<typename T>
struct GetIndexTensorInfoForward<T, cnnlPoolingMode_t::CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING> {
  TensorInfo operator()(user_op::KernelComputeContext* ctx, CnnlTensorDescriptor& local_index_desc,
                        CnnlWorkspace& local_index) {
    return TensorInfo(nullptr, nullptr);
  }
};

template<typename T>
TensorInfo GetMaxIndexTensorInfoForward(user_op::KernelComputeContext* ctx,
                                 CnnlTensorDescriptor& local_index_desc,
                                 CnnlWorkspace& local_index);

template<typename T>
struct GetIndexTensorInfoForward<T, cnnlPoolingMode_t::CNNL_POOLING_MAX> {
  TensorInfo operator()(user_op::KernelComputeContext* ctx, CnnlTensorDescriptor& local_index_desc,
                        CnnlWorkspace& local_index) {
    return GetMaxIndexTensorInfoForward<T>(ctx, local_index_desc, local_index);
  }
};

template<typename T>
TensorInfo GetMaxIndexTensorInfoForward(user_op::KernelComputeContext* ctx,
                                 CnnlTensorDescriptor& local_index_desc,
                                 CnnlWorkspace& local_index) {
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("index", 0);
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
  // cnnlPoolingForwardWithIndex requires index_desc->dtype == CNNL_DTYPE_INT32 or
  // CNNL_DTYPE_INT16 But in oneflow/user/ops/max_pool_op.cpp its dtype is set as kInt64.
  // cnnlPoolingForwardWithIndex requires index dtype is int32 for float input,
  // and index dtype is int16 for half input
  auto local_index_dtype = CNNL_DTYPE_INVALID;
  if (GetDataType<T>::value == DataType::kFloat) {
    local_index_dtype = ConvertToCnnlDataType(kInt32);
    local_index.resize(sizeof(int32_t) * indice->shape_view().elem_cnt());
  } else if (GetDataType<T>::value == DataType::kFloat16) {
    local_index_dtype = ConvertToCnnlDataType(kInt16);
    local_index.resize(sizeof(int16_t) * indice->shape_view().elem_cnt() * 3);
  }
  if (data_format == "channels_last") {
    local_index_desc.set(indice->shape_view().NumAxes(), indice->shape_view().data(),
                          local_index_dtype, layout);
  } else {
    auto shape = mlu::ComputeShapeNchwToNhwc(Shape(indice->shape_view()));
    local_index_desc.set(indice->shape_view().NumAxes(), shape.data(), local_index_dtype, layout);
  }
  return TensorInfo(local_index_desc.desc(), local_index.dptr());
}

}  // namespace

template<typename T, cnnlPoolingMode_t pooling_mode>
class AdaptivePool2DKernel final : public user_op::OpKernel {
 public:
  AdaptivePool2DKernel() = default;
  ~AdaptivePool2DKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    cnnlDataType_t dtype = ConvertToCnnlDataType(in_tensor->data_type());
    CnnlTensorDescriptor in_desc, out_desc;

    if (data_format == "channels_last") {
      in_desc.set(in_tensor->shape_view().NumAxes(), in_tensor->shape_view().data(), dtype,
                  CNNL_LAYOUT_NHWC);
      out_desc.set(out_tensor->shape_view().NumAxes(), out_tensor->shape_view().data(), dtype,
                   CNNL_LAYOUT_NHWC);
      ComputeNHWC(ctx, in_desc, in_tensor->dptr(), out_desc, out_tensor->mut_dptr());
      return;
    }

    size_t tmp_in_workspace_size =
        in_tensor->shape_view().elem_cnt() * sizeof(in_tensor->data_type());
    size_t tmp_out_workspace_size =
        out_tensor->shape_view().elem_cnt() * sizeof(out_tensor->data_type());
    CnnlWorkspace tmp_in_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_in_workspace_size);
    CnnlWorkspace tmp_out_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                         tmp_out_workspace_size);
    mlu::ConvertMemoryFormat(ctx->stream(), in_tensor->shape_view(), in_tensor->data_type(),
                             in_tensor->dptr(), tmp_in_cnnl_workspace.dptr(), MemoryFormat::kNCHW,
                             MemoryFormat::kNHWC);
    void* temp_in_ptr = tmp_in_cnnl_workspace.dptr();
    void* temp_out_ptr = tmp_out_cnnl_workspace.dptr();
    auto in_shape = Shape(in_tensor->shape_view());
    auto out_shape = Shape(out_tensor->shape_view());
    auto index_shape = Shape(index_tensor->shape_view());
    in_shape = mlu::ComputeShapeNchwToNhwc(in_shape);
    out_shape = mlu::ComputeShapeNchwToNhwc(out_shape);
    index_shape = mlu::ComputeShapeNchwToNhwc(index_shape);
    in_desc.set(in_tensor->shape_view().NumAxes(), in_shape.data(), dtype, CNNL_LAYOUT_NHWC);
    out_desc.set(out_tensor->shape_view().NumAxes(), out_shape.data(), dtype, CNNL_LAYOUT_NHWC);

    ComputeNHWC(ctx, in_desc, temp_in_ptr, out_desc, temp_out_ptr);
    mlu::ConvertMemoryFormat(ctx->stream(), out_shape, out_tensor->data_type(), temp_out_ptr,
                             out_tensor->mut_dptr(), MemoryFormat::kNHWC, MemoryFormat::kNCHW);
  }

  void ComputeNHWC(user_op::KernelComputeContext* ctx, const CnnlTensorDescriptor& in_desc,
                   const void* in_ptr, const CnnlTensorDescriptor& out_desc, void* out_ptr) const {
    size_t adaptive_avg_pool2d_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetAdaptivePoolingForwardWorkspaceSize(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc     */ in_desc.desc(),
        /* mode           */ pooling_mode,
        /* output_desc    */ out_desc.desc(),
        /* workspace_size */ &adaptive_avg_pool2d_workspace_size));
    CnnlWorkspace adaptive2d_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                            adaptive_avg_pool2d_workspace_size);
    void* adaptive_avg_pool2d_workspace = adaptive2d_cnnl_workspace.dptr();
    CnnlTensorDescriptor local_index_desc;
    CnnlWorkspace local_index(ctx->stream()->As<ep::MluStream>());
    auto index_tensor_info = GetIndexTensorInfoForward<T, pooling_mode>()(ctx, local_index_desc, local_index);
    OF_CNNL_CHECK(cnnlAdaptivePoolingForward_v2(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc     */ in_desc.desc(),
        /* input          */ in_ptr,
        /* mode           */ pooling_mode,
        /* workspace      */ adaptive_avg_pool2d_workspace,
        /* workspace_size */ adaptive_avg_pool2d_workspace_size,
        /* output_desc    */ out_desc.desc(),
        /* output         */ out_ptr,
        /* index_desc     */ index_tensor_info.tensor_desc,
        /* index          */ index_tensor_info.dptr));
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("index", 0);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    // cnnlTensorLayout_t layout =
    //     (data_format == "channels_last") ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NCHW;
    CnnlTensorDescriptor indice_desc;
    if (data_format == "channels_last") {
      indice_desc.set(indice->shape_view().size(), indice->shape_view().data(),
                      ConvertToCnnlDataType(indice->data_type()), layout);
    } else {
      auto shape = mlu::ComputeShapeNchwToNhwc(Shape(indice->shape_view()));
      indice_desc.set(indice->shape_view().size(), shape.data(),
                      ConvertToCnnlDataType(indice->data_type()), layout);
    }
    auto local_index_dtype = CNNL_DTYPE_INVALID;
    if (GetDataType<T>::value == DataType::kFloat) {
      local_index_dtype = ConvertToCnnlDataType(kInt32);
    } else if (GetDataType<T>::value == DataType::kFloat16) {
      local_index_dtype = ConvertToCnnlDataType(kInt16);
    }
    // cast int32/int16 index to int64 index
    CnnlTensorDescriptor int32_index_desc;
    char* int32_index_dptr = reinterpret_cast<char*>(local_index.dptr());
    if (local_index_dtype == CNNL_DTYPE_INT16) {
      int32_index_dptr += sizeof(int16_t) * indice->shape_view().elem_cnt();
      int32_index_desc.set(indice->shape_view().NumAxes(), indice->shape_view().data(),
                           CNNL_DTYPE_INT32, layout);
      OF_CNNL_CHECK(cnnlCastDataType(
          ctx->stream()->As<ep::MluStream>()->cnnl_handle(), local_index_desc.desc(),
          local_index.dptr(), CNNL_CAST_INT16_TO_INT32, int32_index_desc.desc(), int32_index_dptr));
    }
    OF_CNNL_CHECK(cnnlCastDataType(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        (local_index_dtype == CNNL_DTYPE_INT16) ? int32_index_desc.desc() : local_index_desc.desc(),
        int32_index_dptr, CNNL_CAST_INT32_TO_INT64, indice_desc.desc(), indice->mut_dptr()));
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_POOL2D_MLU_KERNEL(name, dtype, pooling_mode) \
  REGISTER_USER_KERNEL(name)                                           \
      .SetCreateFn<AdaptivePool2DKernel<dtype, pooling_mode>>()        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)  \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_ADAPTIVE_POOL2D_MLU_KERNEL("adaptive_max_pool2d", float, CNNL_POOLING_MAX)
REGISTER_ADAPTIVE_POOL2D_MLU_KERNEL("adaptive_max_pool2d", float16, CNNL_POOLING_MAX)

namespace {

template<typename T, cnnlPoolingMode_t>
struct GetIndexTensorInfoBackward;

template<typename T>
struct GetIndexTensorInfoBackward<T, cnnlPoolingMode_t::CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING> {
  TensorInfo operator()(user_op::KernelComputeContext* ctx, CnnlTensorDescriptor& local_index_desc,
                        CnnlWorkspace& local_index) {
    return TensorInfo(nullptr, nullptr);
  }
};

template<typename T>
TensorInfo GetMaxIndexTensorInfoBackward(user_op::KernelComputeContext* ctx,
                                 CnnlTensorDescriptor& local_index_desc,
                                 CnnlWorkspace& local_index);

template<typename T>
struct GetIndexTensorInfoBackward<T, cnnlPoolingMode_t::CNNL_POOLING_MAX> {
  TensorInfo operator()(user_op::KernelComputeContext* ctx, CnnlTensorDescriptor& local_index_desc,
                        CnnlWorkspace& local_index) {
    return GetMaxIndexTensorInfoBackward<T>(ctx, local_index_desc, local_index);
  }
};

template<typename T>
TensorInfo GetMaxIndexTensorInfoBackward(user_op::KernelComputeContext* ctx,
                                 CnnlTensorDescriptor& local_index_desc,
                                 CnnlWorkspace& local_index) {
  const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("index", 0);
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
  CnnlTensorDescriptor indice_desc;
  auto indice_shape = Shape(indice->shape_view());
  const void* temp_indice = indice->dptr();
  // CnnlWorkspace temp_indice_workspace(ctx->stream()->As<ep::MluStream>());
  if (data_format != "channels_last") {
    indice_shape = mlu::ComputeShapeNchwToNhwc(indice_shape);
    // temp_indice_workspace.resize(indice_shape.elem_cnt() * GetSizeOfDataType(indice->data_type()));
    // // convert indice to NHWC
    // mlu::ConvertMemoryFormat(ctx->stream(), indice->shape_view(), indice->data_type(),
    //                          indice->dptr(), temp_indice_workspace.dptr(), MemoryFormat::kNCHW,
    //                          MemoryFormat::kNHWC);
    // temp_indice = temp_indice_workspace.dptr();
  }
  indice_desc.set(indice_shape.size(), indice_shape.data(),
                  ConvertToCnnlDataType(indice->data_type()), layout);
  // cnnlPoolingBackward requires index_desc is int32/int16, which is int64 in oneflow op
  auto local_index_dtype = CNNL_DTYPE_INVALID;
  if (GetDataType<T>::value == DataType::kFloat) {
    local_index_dtype = CNNL_DTYPE_INT32;
    local_index.resize(sizeof(int32_t) * indice_shape.elem_cnt());
  } else if (GetDataType<T>::value == DataType::kFloat16) {
    local_index_dtype = CNNL_DTYPE_INT16;
    local_index.resize(sizeof(int16_t) * indice_shape.elem_cnt() * 3);
  }
  local_index_desc.set(indice_shape.NumAxes(), indice_shape.data(), local_index_dtype, layout);
  if (local_index_dtype == CNNL_DTYPE_INT16) {
    CnnlTensorDescriptor int32_index_desc;
    int32_index_desc.set(indice_shape.NumAxes(), indice_shape.data(), CNNL_DTYPE_INT32, layout);
    char* int32_index_dptr =
        reinterpret_cast<char*>(local_index.dptr()) + sizeof(int16_t) * indice_shape.elem_cnt();
    OF_CNNL_CHECK(cnnlCastDataType(
        /* handle      */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc  */ indice_desc.desc(),
        /* input       */ temp_indice,
        /* cast_type   */ CNNL_CAST_INT64_TO_INT32,
        /* output_desc */ int32_index_desc.desc(),
        /* output      */ int32_index_dptr));

    OF_CNNL_CHECK(cnnlCastDataType(
        /* handle      */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc  */ int32_index_desc.desc(),
        /* input       */ int32_index_dptr,
        /* cast_type   */ CNNL_CAST_INT32_TO_INT16,
        /* output_desc */ local_index_desc.desc(),
        /* output      */ local_index.dptr()));
  } else {
    OF_CNNL_CHECK(cnnlCastDataType(
        /* handle      */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc  */ indice_desc.desc(),
        /* input       */ temp_indice,
        /* cast_type   */ CNNL_CAST_INT64_TO_INT32,
        /* output_desc */ local_index_desc.desc(),
        /* output      */ local_index.dptr()));
  }
  return TensorInfo(local_index_desc.desc(), local_index.dptr());
}

}  // namespace

template<typename T, cnnlPoolingMode_t pooling_mode>
class AdaptivePool2DGradKernel final : public user_op::OpKernel {
 public:
  AdaptivePool2DGradKernel() = default;
  ~AdaptivePool2DGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);

    CHECK_EQ(x_tensor->shape_view().NumAxes(), 4);

    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_last") {
      ComputeNHWC(ctx, dy_tensor, dx_tensor);
      return;
    }

    CnnlTensorDescriptor dy_desc, dx_desc, local_index_desc;
    CnnlWorkspace local_index(ctx->stream()->As<ep::MluStream>());
    auto dtype = ConvertToCnnlDataType(dy_tensor->data_type());

    size_t element_size = GetSizeOfDataType(x_tensor->data_type());
    size_t tmp_dy_workspace_size = dy_tensor->shape_view().elem_cnt() * element_size;
    CnnlWorkspace tmp_dy_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_dy_workspace_size);
    void* tmp_dy_ptr = tmp_dy_cnnl_workspace.dptr();

    mlu::ConvertMemoryFormat(ctx->stream(), dy_tensor->shape_view(), dy_tensor->data_type(),
                             dy_tensor->dptr(), tmp_dy_cnnl_workspace.dptr(), MemoryFormat::kNCHW,
                             MemoryFormat::kNHWC);

    auto dy_shape = Shape(dy_tensor->shape_view());
    auto dx_shape = Shape(dx_tensor->shape_view());
    dy_shape = mlu::ComputeShapeNchwToNhwc(dy_shape);
    dx_shape = mlu::ComputeShapeNchwToNhwc(dx_shape);
    dy_desc.set(dy_tensor->shape_view().NumAxes(), dy_shape.data(), dtype, CNNL_LAYOUT_NHWC);
    dx_desc.set(dx_tensor->shape_view().NumAxes(), dx_shape.data(), dtype, CNNL_LAYOUT_NHWC);

    size_t tmp_dx_workspace_size = dx_tensor->shape_view().elem_cnt() * element_size;
    CnnlWorkspace tmp_dx_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_dx_workspace_size);
    void* tmp_dx_ptr = tmp_dx_cnnl_workspace.dptr();

    auto tensor_info = GetIndexTensorInfoBackward<T, pooling_mode>()(ctx, local_index_desc, local_index);
    OF_CNNL_CHECK(cnnlAdaptivePoolingBackward(
        /* handle     */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* y_desc     */ dy_desc.desc(),
        /* y          */ tmp_dy_ptr,
        /* index_desc */ tensor_info.tensor_desc,
        /* index      */ tensor_info.dptr,
        /* mode       */ pooling_mode,
        /* dx_desc    */ dx_desc.desc(),
        /* dx         */ tmp_dx_ptr));

    mlu::ConvertMemoryFormat(ctx->stream(), dx_shape, dx_tensor->data_type(), tmp_dx_ptr,
                             dx_tensor->mut_dptr(), MemoryFormat::kNHWC, MemoryFormat::kNCHW);
  }

  void ComputeNHWC(user_op::KernelComputeContext* ctx, const user_op::Tensor* dy_tensor,
                   user_op::Tensor* dx_tensor) const {
    auto dtype = ConvertToCnnlDataType(dy_tensor->data_type());
    CnnlTensorDescriptor dy_desc, dx_desc, local_index_desc;
    dy_desc.set(dy_tensor->shape_view().NumAxes(), dy_tensor->shape_view().data(), dtype,
                CNNL_LAYOUT_NHWC);
    dx_desc.set(dx_tensor->shape_view().NumAxes(), dx_tensor->shape_view().data(), dtype,
                CNNL_LAYOUT_NHWC);
    CnnlWorkspace local_index(ctx->stream()->As<ep::MluStream>());
    auto tensor_info = GetIndexTensorInfoBackward<T, pooling_mode>()(ctx, local_index_desc, local_index);
    OF_CNNL_CHECK(cnnlAdaptivePoolingBackward(
        /*handle*/ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* y_desc     */ dy_desc.desc(),
        /* y          */ dy_tensor->dptr(),
        /* index_desc */ tensor_info.tensor_desc,
        /* index      */ tensor_info.dptr,
        /* mode       */ pooling_mode,
        /* dx_desc    */ dx_desc.desc(),
        /* dx         */ dx_tensor->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_POOL2D_GRAD_MLU_KERNEL(name, dtype, pooling_mode) \
  REGISTER_USER_KERNEL(name)                                                \
      .SetCreateFn<AdaptivePool2DGradKernel<dtype, pooling_mode>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)       \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));
REGISTER_ADAPTIVE_POOL2D_GRAD_MLU_KERNEL("adaptive_max_pool2d_grad", float, CNNL_POOLING_MAX)
REGISTER_ADAPTIVE_POOL2D_GRAD_MLU_KERNEL("adaptive_max_pool2d_grad", float16, CNNL_POOLING_MAX)

}  // namespace oneflow
