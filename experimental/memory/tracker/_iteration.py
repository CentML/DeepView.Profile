import torch
import inspect

from tracker._base import _TrackerBase
from tracker.meters import MemoryMeter
from hook_manager import HookManager


class _IterationTracker(_TrackerBase):
    def __init__(self):
        super().__init__()
        self._hook_manager = HookManager()
        self._meter = MemoryMeter()

    def start_tracking(self):
        super().start_tracking()
        self._meter.reset()
        self._hook_manager.attach_hooks_on_module(
            torch,
            self._is_callable,
            self._callable_hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.Tensor,
            self._is_callable,
            self._callable_hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.nn.functional,
            self._is_callable,
            self._callable_hook_creator,
        )

    def stop_tracking(self):
        super().stop_tracking()
        self._hook_manager.remove_hooks()

    def _callable_hook_creator(self, func):
        def hook(*args, **kwargs):
            self._meter.checkpoint()
            retval = func(*args, **kwargs)
            delta = self._meter.checkpoint()
            if delta > 0 or delta < 0:
                print(func.__name__, delta)
            return retval
        return hook

    def _is_callable(self, maybe_fn):
        return (
            inspect.isfunction(maybe_fn) or
            inspect.ismethod(maybe_fn) or
            inspect.isbuiltin(maybe_fn) or
            inspect.isroutine(maybe_fn)
        )

    def get_report(self):
        return []
