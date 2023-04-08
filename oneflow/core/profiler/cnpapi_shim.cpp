
#include "cnpapi.h"
#include "cnpapi_activity_api.h"
#include "cnpapi_callback_api_types.h"
#include "cnpapi_cnml_id.h"
#include "cnpapi_cnrt_id.h"
#include "oneflow/core/profiler/cnpapi_shim.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/profiler/event.h"
#include "oneflow/core/profiler/profile_manager.h"

namespace oneflow {

void bufferRequested(uint64_t** buffer, size_t* size, size_t* maxNumRecords) {
  *buffer = reinterpret_cast<uint64_t*>(malloc(*size));  // NOLINT
  if (*buffer == NULL) {
    printf("Error:out of memory\n");
    exit(-1);
  }
  *maxNumRecords = 0;
}

void bufferCompleted(uint64_t* buffer, size_t size, size_t validSize) {
  using namespace profiler;
  std::set<std::shared_ptr<IEvent>> custom_events;
  std::unordered_map<std::shared_ptr<IEvent>, int64_t> corr_ids;
  auto* pmgr = CHECK_JUST(SingletonMaybe<profiler::ProfileManager>());

  cnpapiResult status;  // NOLINT
  cnpapiActivity* record = nullptr;

  if (validSize > 0) {
    do {
      status = cnpapiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CNPAPI_SUCCESS) {
        {
          switch (record->type) {
            case CNPAPI_ACTIVITY_TYPE_KERNEL: {
              auto* activity_kernel = reinterpret_cast<cnpapiActivityKernel*>(record);
              std::cout << activity_kernel->name << " " << activity_kernel->start << " "
                        << activity_kernel->end << " " << activity_kernel->received << " "
                        << activity_kernel->queued << " " << activity_kernel->queued << std::endl;
              auto custom_event = CustomEvent::Create(std::string(activity_kernel->name),
                                                      CustomEventType::kMluKernel);
              custom_event->SetStartedAt(static_cast<time_t>(activity_kernel->start));
              custom_event->SetFinishedAt(static_cast<time_t>(activity_kernel->end));
              custom_events.emplace(custom_event);
              corr_ids[custom_event] = activity_kernel->correlation_id;
            }
            default: break;
          }
        }
      } else if (status == CNPAPI_ERROR_MAX_LIMIT_REACHED) {
        break;
      }
    } while (1);
  }
  free(buffer);

  while (!pmgr->events_.empty()) {
    auto evt = pmgr->events_.front();
    pmgr->events_.pop();
    auto evt_kernel = std::dynamic_pointer_cast<KernelEvent>(evt);
    if (evt_kernel) {
      std::set<int64_t> current_corr_ids;
      if (!custom_events.empty()) {
        for (const auto& x : custom_events) {
          if (evt_kernel->AddChildEventIfSo(x)) { current_corr_ids.insert(corr_ids[x]); }
        }
        for (const auto& x : custom_events) {
          if (!evt_kernel->HasChildEvent(x) && current_corr_ids.count(corr_ids[x])) {
            evt_kernel->AddChildEvent(x);
          }
        }
        evt_kernel->WalkAmongChildren(
            [&custom_events](const std::shared_ptr<IEvent>& child) { custom_events.erase(child); });
      }
    }
    pmgr->events_result_.emplace_back(evt);
  }
}
void CnpPrepareTrace() {
  OF_CNPAPI_CHECK(cnpapiInit());
  OF_CNPAPI_CHECK(cnpapiActivityEnableLatencyTimestamps(true));

  // enable activity types
  OF_CNPAPI_CHECK(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL));

  OF_CNPAPI_CHECK(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
  // Register callbacks for buffer requests and for buffers completed by CNPAPI
  OF_CNPAPI_CHECK(cnpapiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

void CnpReleaseTrace() {
  // disable activity types
  OF_CNPAPI_CHECK(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL));
  OF_CNPAPI_CHECK(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
  OF_CNPAPI_CHECK(cnpapiActivityFlushAll());

  OF_CNPAPI_CHECK(cnpapiRelease());
}

}  // namespace oneflow