#ifndef ONEFLOW_CORE_PROFILER_CNPAPI_SHIM_H_
#define ONEFLOW_CORE_PROFILER_CNPAPI_SHIM_H_

#include <cstdint>

namespace oneflow {

void CnpPrepareTrace();
void CnpReleaseTrace();
void bufferCompleted(uint64_t* buffer, std::size_t size, std::size_t validSize);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PROFILER_CNPAPI_SHIM_H_
