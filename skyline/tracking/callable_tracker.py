import inspect

import torch

from skyline.tracking.base import TrackerBase
from skyline.tracking.hook_manager import HookManager
from skyline.version_utils import Version

OLD_VF_PATH_VERSION = Version.parse_semantic_version('1.4.0')


class CallableTracker(TrackerBase):
    def __init__(self, hook_creator):
        super().__init__()
        self._hook_manager = HookManager()
        self._hook_creator = hook_creator
        self._torch_version = Version.parse_semantic_version(torch.__version__)

    def start_tracking(self):
        super().start_tracking()
        self._hook_manager.attach_hooks_on_module(
            torch,
            lambda fn: _is_callable_and_public(fn) and \
              fn.__name__ not in BLACKLISTED_TORCH_METHODS,
            self._hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.Tensor,
            lambda fn: _is_callable_and_public(fn) and \
              fn.__name__ != 'backward' and \
              fn.__name__ not in BLACKLISTED_TENSOR_METHODS,
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

        # The _VF module was moved to the torch module after version 1.4.0.
        # This is an unfortunate hack because we need to monkey patch certain
        # internal PyTorch functions to be able to identify all the operations
        # properly. The _VF module contains recurrent operations (e.g., lstm).
        vf_module = (
            torch._VF if self._torch_version is None or
                self._torch_version > OLD_VF_PATH_VERSION
            else torch.nn._VF
        )
        self._hook_manager.attach_hooks_on_module_using(
            vf_module,
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

# Original source of these blacklists:
# https://github.com/NVIDIA/apex/blob/master/apex/pyprof/nvtx/nvmarker.py
BLACKLISTED_DUNDERS = {
    '__all__',
    '__array__',
    '__array_priority__',
    '__array_wrap__',
    '__bool__',
    '__builtins__',
    '__cached__',
    '__class__',
    '__deepcopy__',
    '__delattr__',
    '__delitem__',
    '__dict__',
    '__dir__',
    '__doc__',
    '__file__',
    '__format__',
    '__getattribute__',
    '__getitem__',
    '__hash__',
    '__index__',
    '__init__',
    '__init_subclass__',
    '__iter__',
    '__len__',
    '__loader__',
    '__module__',
    '__name__',
    '__new__',
    '__nonzero__',
    '__package__',
    '__path__',
    '__reduce__',
    '__reduce_ex__',
    '__repr__',
    '__reversed__',
    '__setattr__',
    '__setitem__',
    '__setstate__',
    '__sizeof__',
    '__spec__',
    '__str__',
    '__subclasshook__',
    '__version__',
    '__weakref__',
}

BLACKLISTED_TENSOR_METHODS = {
    'size', 'dim', 'item', 'tolist',
}

BLACKLISTED_TORCH_METHODS = {
    'is_storage',
}


def _is_callable_dunder(maybe_fn):
    """
    Returns True if maybe_fn is a callable dunder (callable named with double
    underscores) (e.g., __add__)
    """
    return (
        _is_callable(maybe_fn) and
        len(maybe_fn.__name__) > 4 and
        maybe_fn.__name__[:2] == '__' and
        maybe_fn.__name__[-2:] == '__' and
        maybe_fn.__name__ not in BLACKLISTED_DUNDERS
    )


def _is_callable(maybe_fn):
    return (
        inspect.isfunction(maybe_fn) or
        inspect.ismethod(maybe_fn) or
        inspect.isbuiltin(maybe_fn) or
        inspect.isroutine(maybe_fn)
    )
