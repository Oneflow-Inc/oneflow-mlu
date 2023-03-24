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
namespace oneflow {

void AdamUpdateKernelUtil(cnrtQueue_t queue, cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                          cnrtDataType_t cnrt_type, int64_t n, float scale, float l1, float l2,
                          float beta1, float beta2, float epsilon, float weight_decay, bool amsgrad,
                          bool do_bias_correction, float learning_rate_val, float lr_scale,
                          float bias_correction1_val, float bias_correction2_val,
                          const float* learning_rate, const float* scale_by_ptr,
                          const int64_t* skip_if, const float* bias_correction1,
                          const float* bias_correction2, const float* model_diff, float* model,
                          float* m, float* v, float* max_v);

}  // namespace oneflow
