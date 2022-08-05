#include <c10/core/impl/CUDATrace.h>
#include <c10/util/CallOnce.h>

#include <mutex>

namespace c10 {
namespace impl {

std::atomic<const PyInterpreter*> CUDATrace::cudaTraceState;

void CUDATrace::set_trace(const PyInterpreter* trace) {
  static c10::once_flag flag;
  c10::call_once(
    flag,
    [&](){ cudaTraceState.store(trace); }
  );
}

} // namespace impl
} // namespace c10
