#pragma once

#include <c10/core/impl/PyInterpreter.h>

namespace c10 {
namespace impl {

struct C10_API CUDATrace {
  // When PyTorch migrates to C++20+, this should be changed to an atomic flag.
  static std::atomic<const PyInterpreter*> cudaTraceState;

  static void set_trace(const PyInterpreter*);
  
  static const PyInterpreter* get_trace() {
    return cudaTraceState.load();
  }
};

} // namespace impl
} // namespace c10
