import traceback
import sys
from typing import Callable, List, Tuple, Any
import logging


class CallbackRegistry:
    def __init__(self, name: str):
        self.name = name
        self.callback_list: List[Callable[[Tuple[Any, ...]], None]] = []

    def add_callback(self, cb: Callable[[Tuple[Any, ...]], None]) -> None:
        self.callback_list.append(cb)

    def fire_callbacks(self, *args: int) -> None:
        logger = logging.getLogger(__name__)
        for cb in self.callback_list:
            try:
                cb(*args)
            except Exception as e:
                logger.exception(
                    f"Exception in callback for {self.name} registered with CUDA trace",
                    e,
                )


CUDAEventCreationCallbacks = CallbackRegistry("CUDAEventCreation")
CUDAEventDeletionCallbacks = CallbackRegistry("CUDAEventDeletion")
CUDAEventRecordCallbacks = CallbackRegistry("CUDAEventRecord")
CUDAEventWaitCallbacks = CallbackRegistry("CUDAEventWait")
CUDAMemoryAllocationCallbacks = CallbackRegistry("CUDAMemoryAllocation")
CUDAMemoryDeallocationCallbacks = CallbackRegistry("CUDAMemoryDeallocation")
CUDAStreamCreationCallbacks = CallbackRegistry("CUDAStreamCreation")


def register_callback_for_cuda_event_creation(
    cb: Callable[[Tuple[Any, ...]], None]
) -> None:
    CUDAEventCreationCallbacks.add_callback(cb)


def register_callback_for_cuda_event_deletion(
    cb: Callable[[Tuple[Any, ...]], None]
) -> None:
    CUDAEventDeletionCallbacks.add_callback(cb)


def register_callback_for_cuda_event_record(
    cb: Callable[[Tuple[Any, ...]], None]
) -> None:
    CUDAEventRecordCallbacks.add_callback(cb)


def register_callback_for_cuda_event_wait(
    cb: Callable[[Tuple[Any, ...]], None]
) -> None:
    CUDAEventWaitCallbacks.add_callback(cb)


def register_callback_for_cuda_memory_allocation(
    cb: Callable[[Tuple[Any, ...]], None]
) -> None:
    CUDAMemoryAllocationCallbacks.add_callback(cb)


def register_callback_for_cuda_memory_deallocation(
    cb: Callable[[Tuple[Any, ...]], None]
) -> None:
    CUDAMemoryDeallocationCallbacks.add_callback(cb)


def register_callback_for_cuda_stream_creation(
    cb: Callable[[Tuple[Any, ...]], None]
) -> None:
    CUDAStreamCreationCallbacks.add_callback(cb)
