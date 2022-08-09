#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/PyInterpreter.h>

namespace c10 {
namespace impl {

template<typename Return, typename... Ts>
static Return noop_interpreter_fn(const PyInterpreter*, Ts...) {
  TORCH_INTERNAL_ASSERT(
    0,
    "attempted to call a PyInterpreter function after corresponding interpreter died");
}

void CUDATraceFunctionWrapper::disarm() {
  event_creation_fn_ = &noop_interpreter_fn;
  event_deletion_fn_ = &noop_interpreter_fn;
  event_record_fn_ = &noop_interpreter_fn;
  event_wait_fn_ = &noop_interpreter_fn;
  memory_allocation_fn_ = &noop_interpreter_fn;
  memory_deallocation_fn_ = &noop_interpreter_fn;
  stream_creation_fn_ = &noop_interpreter_fn;
}

static std::string noop_name_fn(const PyInterpreter*) {
  return "<unloaded interpreter>";
}

static void noop_decref_fn(const PyInterpreter*, PyObject*, bool) {
  // no-op
}

void PyInterpreter::disarm() noexcept {
  name_fn_ = &noop_name_fn;
  decref_fn_ = &noop_decref_fn;
  detach_fn_ = &noop_interpreter_fn;
  dispatch_fn_ = &noop_interpreter_fn;
  is_contiguous_fn_ = &noop_interpreter_fn;
  device_fn_ = &noop_interpreter_fn;
  dim_fn_ = &noop_interpreter_fn;
  strides_fn_ = &noop_interpreter_fn;
  sizes_fn_ = &noop_interpreter_fn;
  sym_sizes_fn_ = &noop_interpreter_fn;
  layout_fn_ = &noop_interpreter_fn;
  trace_cuda_functions.disarm();
}

// Defined out-of-line because it needs access to the definition of TensorImpl.
__ubsan_ignore_function__ c10::intrusive_ptr<TensorImpl> PyInterpreter::detach(
    const TensorImpl* self) const {
  return (*detach_fn_)(this, self);
}

} // namespace impl
} // namespace c10
