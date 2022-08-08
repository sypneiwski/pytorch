#pragma once

#include <c10/core/impl/PyInterpreter.h>

namespace c10 {
namespace impl {

struct C10_API CUDATrace {
  // When PyTorch migrates to C++20+, this should be changed to an atomic flag.
  static std::atomic<const PyInterpreter*> cudaTraceState;

  static bool haveState;

  // This function will only register the first interpreter that tries to invoke it.
  // For all of the next ones it will be a no-op.
  static void set_trace(const PyInterpreter*);
  
  static const PyInterpreter* get_trace() {
    if (!haveState)
      return nullptr;
    return cudaTraceState.load(std::memory_order_acquire);
  }
};

} // namespace impl
} // namespace c10
