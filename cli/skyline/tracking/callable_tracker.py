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
            _is_callable_and_public,
            self._hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.Tensor,
            lambda fn: _is_callable_and_public(fn) and fn.__name__ != 'backward',
            self._hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.Tensor,
            _is_callable_dunder,
            self._hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.nn.functional,
            _is_callable_and_public,
            self._hook_creator,
        )
        self._hook_manager.attach_hooks_on_module_using(
            torch.nn._VF,
            torch._C._VariableFunctions,
            _is_callable_and_public,
            self._hook_creator,
        )

    def stop_tracking(self):
        super().stop_tracking()
        self._hook_manager.remove_hooks()


def _is_callable_and_public(maybe_fn):
    # By convention, _ prefixed functions in Python should not be
    # called by users (i.e. they are "private" functions)
    return _is_callable(maybe_fn) and maybe_fn.__name__[0] != '_'


def _is_callable_dunder(maybe_fn):
    """
    Returns True if maybe_fn is a callable dunder (callable named with double
    underscores) (e.g., __add__)
    """
    return (
        _is_callable(maybe_fn) and
        len(maybe_fn.__name__) > 4 and
        maybe_fn.__name__[:2] == '__' and
        maybe_fn.__name__[-2:] == '__'
    )


def _is_callable(maybe_fn):
    return (
        inspect.isfunction(maybe_fn) or
        inspect.ismethod(maybe_fn) or
        inspect.isbuiltin(maybe_fn) or
        inspect.isroutine(maybe_fn)
    )
