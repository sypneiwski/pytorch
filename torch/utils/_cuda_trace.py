from typing import Callable, Generic, List, TypeVar
import logging

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=Callable[..., None])


class CallbackRegistry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self.callback_list: List[T] = []

    def add_callback(self, cb: T) -> None:
        self.callback_list.append(cb)

    def fire_callbacks(self, *args: int) -> None:
        for cb in self.callback_list:
            try:
                cb(*args)
            except Exception as e:
                logger.exception(
                    f"Exception in callback for {self.name} registered with CUDA trace",
                    e,
                )


CUDAEventCreationCallbacks: CallbackRegistry[Callable[[int], None]] = CallbackRegistry(
    "CUDA event creation"
)
CUDAEventDeletionCallbacks: CallbackRegistry[Callable[[int], None]] = CallbackRegistry(
    "CUDA event deletion"
)
CUDAEventRecordCallbacks: CallbackRegistry[
    Callable[[int, int], None]
] = CallbackRegistry("CUDA event record")
CUDAEventWaitCallbacks: CallbackRegistry[Callable[[int, int], None]] = CallbackRegistry(
    "CUDA event wait"
)
CUDAMemoryAllocationCallbacks: CallbackRegistry[
    Callable[[int], None]
] = CallbackRegistry("CUDA memory allocation")
CUDAMemoryDeallocationCallbacks: CallbackRegistry[
    Callable[[int], None]
] = CallbackRegistry("CUDA memory deallocation")
CUDAStreamCreationCallbacks: CallbackRegistry[Callable[[int], None]] = CallbackRegistry(
    "CUDA stream creation"
)


def register_callback_for_cuda_event_creation(cb: Callable[[int], None]) -> None:
    CUDAEventCreationCallbacks.add_callback(cb)


def register_callback_for_cuda_event_deletion(cb: Callable[[int], None]) -> None:
    CUDAEventDeletionCallbacks.add_callback(cb)


def register_callback_for_cuda_event_record(cb: Callable[[int, int], None]) -> None:
    CUDAEventRecordCallbacks.add_callback(cb)


def register_callback_for_cuda_event_wait(cb: Callable[[int, int], None]) -> None:
    CUDAEventWaitCallbacks.add_callback(cb)


def register_callback_for_cuda_memory_allocation(cb: Callable[[int], None]) -> None:
    CUDAMemoryAllocationCallbacks.add_callback(cb)


def register_callback_for_cuda_memory_deallocation(cb: Callable[[int], None]) -> None:
    CUDAMemoryDeallocationCallbacks.add_callback(cb)


def register_callback_for_cuda_stream_creation(cb: Callable[[int], None]) -> None:
    CUDAStreamCreationCallbacks.add_callback(cb)
