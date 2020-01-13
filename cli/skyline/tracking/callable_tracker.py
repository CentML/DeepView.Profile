import inspect

import torch

from skyline.tracking.base import TrackerBase
from skyline.tracking.hook_manager import HookManager


class CallableTracker(TrackerBase):
    def __init__(self, hook_creator):
        super().__init__()
        self._hook_manager = HookManager()
        self._hook_creator = hook_creator

    def start_tracking(self):
        super().start_tracking()
        self._hook_manager.attach_hooks_on_module(
            torch,
            _is_callable,
            self._hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.Tensor,
            lambda fn: _is_callable(fn) and fn.__name__ != 'backward',
            self._hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.nn.functional,
            _is_callable,
            self._hook_creator,
        )
        self._hook_manager.attach_hooks_on_module_using(
            torch.nn._VF,
            torch._C._VariableFunctions,
            _is_callable,
            self._hook_creator,
        )

    def stop_tracking(self):
        super().stop_tracking()
        self._hook_manager.remove_hooks()


def _is_callable(maybe_fn):
    is_callable = (
        inspect.isfunction(maybe_fn) or
        inspect.ismethod(maybe_fn) or
        inspect.isbuiltin(maybe_fn) or
        inspect.isroutine(maybe_fn)
    )
    if not is_callable:
        return False

    # By convention, _ prefixed functions in Python should not be
    # called by users (i.e. they are "private" functions)
    return maybe_fn.__name__[0] != '_'
