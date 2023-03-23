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



#if defined(__CUDACC__)
#define OF_DEVICE_FUNC __device__ __host__ __forceinline__
#else
#define OF_DEVICE_FUNC inline
#endif


template<typename T, typename G>
struct CastScaleRegularizeGradientFunctor {
  OF_DEVICE_FUNC
  T operator()(G model_diff, T model, T scale, float l1, float l2) const {
    return static_cast<T>(model_diff) * scale + l1 * ((model >= 0) - (model <= 0)) + l2 * model;
  }
};

template<typename T, typename G>
struct AdamUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, T* m, T* v, T* max_v, T scale, float l1, float l2,
                  float beta1, float beta2, float epsilon, float weight_decay, bool amsgrad,
                  float bias_correction1, float bias_correction2, float learning_rate) const {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);

    const T next_m = beta1 * *m + (1 - beta1) * model_diff_t;
    *m = next_m;

    const T next_v = beta2 * *v + (1 - beta2) * model_diff_t * model_diff_t;
    *v = next_v;

    T denom = 0;
    if (amsgrad) {
      const T next_max_v =
          *max_v > next_v ? *max_v : next_v;  // use std::max has bug in GPU kernel.
      *max_v = next_max_v;
      denom = (sqrt(next_max_v) / sqrt(bias_correction2)) + epsilon;
    } else {
      denom = (sqrt(next_v) / sqrt(bias_correction2)) + epsilon;
    }
    const T step_size = learning_rate / bias_correction1;
    *model = model_val - step_size * (next_m / denom) - learning_rate * weight_decay * model_val;
  }
};


template<typename T, typename G, typename C>
struct FusedAdamUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, C* model_copy, T* m, T* v, T* max_v, T scale,
                  float l1, float l2, float beta1, float beta2, float epsilon, float weight_decay,
                  bool amsgrad, float bias_correction1, float bias_correction2,
                  float learning_rate) const {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);

    const T next_m = beta1 * *m + (1 - beta1) * model_diff_t;
    *m = next_m;

    const T next_v = beta2 * *v + (1 - beta2) * model_diff_t * model_diff_t;
    *v = next_v;

    T denom = 0;
    if (amsgrad) {
      const T next_max_v =
          *max_v > next_v ? *max_v : next_v;  // use std::max has bug in GPU kernel.
      *max_v = next_max_v;
      denom = (sqrt(next_max_v) / sqrt(bias_correction2)) + epsilon;
    } else {
      denom = (sqrt(next_v) / sqrt(bias_correction2)) + epsilon;
    }
    const T step_size = learning_rate / bias_correction1;
    const T next_model =
        model_val - step_size * (next_m / denom) - learning_rate * weight_decay * model_val;
    *model = next_model;
    *model_copy = static_cast<C>(next_model);
  }
};

template<typename T, typename G, typename C>
struct AdamUpdateKernelUtil {
  static void Update(int64_t n, T scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, bool amsgrad,
                     bool do_bias_correction, float learning_rate_val, float lr_scale,
                     float bias_correction1_val, float bias_correction2_val,
                     const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
                     const float* bias_correction1, const float* bias_correction2,
                     const G* model_diff, T* model, C* model_copy, T* m, T* v, T* max_v);
};

}  // namespace oneflow
