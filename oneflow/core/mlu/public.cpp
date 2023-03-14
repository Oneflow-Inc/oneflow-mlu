/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include <random>
#include <iomanip>
#include <limits>
#include <cfloat>
#include <chrono>
#include "oneflow/core/mlu/public.h"

namespace oneflow {

namespace chrono = std::chrono;
static auto start_tp = chrono::steady_clock::now();
size_t epoch_elapsed() {
  return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_tp)
      .count();
}

std::string cnnlErrorString(cnnlStatus_t status) {
  switch (status) {
    default: { return "CNNL_STATUS_UNKNOWN"; }
    case CNNL_STATUS_SUCCESS: {
      return "CNNL_STATUS_SUCCESS";
    }
    case CNNL_STATUS_NOT_INITIALIZED: {
      return "CNNL_STATUS_NOT_INITIALIZED";
    }
    case CNNL_STATUS_ALLOC_FAILED: {
      return "CNNL_STATUS_ALLOC_FAILED";
    }
    case CNNL_STATUS_BAD_PARAM: {
      return "CNNL_STATUS_BAD_PARAM";
    }
    case CNNL_STATUS_INTERNAL_ERROR: {
      return "CNNL_STATUS_INTERNAL_ERROR";
    }
    case CNNL_STATUS_ARCH_MISMATCH: {
      return "CNNL_STATUS_MISMATCH";
    }
    case CNNL_STATUS_EXECUTION_FAILED: {
      return "CNNL_STATUS_EXECUTION_FAILED";
    }
    case CNNL_STATUS_NOT_SUPPORTED: {
      return "CNNL_STATUS_NOT_SUPPORTED";
    }
    case CNNL_STATUS_NUMERICAL_OVERFLOW: {
      return "CNNL_STATUS_NUMERICAL_OVERFLOW";
    }
  }
}

void cnnlCheck(cnnlStatus_t result,
               char const *const func,
               const char *const file,
               int const line) {
  if (result) {
    std::string error =
        "\"" + std::string(cnnlGetErrorString(result)) + " in " + std::string(func) + "\"";
    throw std::runtime_error(error);
  }
}

}