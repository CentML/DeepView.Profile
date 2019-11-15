import torch
import inspect
import collections

from tracker._base import _TrackerBase
from tracker.call_stack import CallStack
from tracker.meters import MemoryMeter
from hook_manager import HookManager


OperationContext = collections.namedtuple(
    'OperationContext',
    ['operation_name', 'stack'],
)


class _IterationTracker(_TrackerBase):
    def __init__(self):
        super().__init__()
        self._hook_manager = HookManager()
        self._meter = MemoryMeter()
        self._grad_function_contexts = {}

    def start_tracking(self):
        super().start_tracking()
        self._meter.reset()
        self._grad_function_contexts.clear()
        self._hook_manager.attach_hooks_on_module(
            torch,
            self._is_callable,
            self._callable_hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.Tensor,
            lambda fn: self._is_callable(fn) and fn.__name__ != 'backward',
            self._callable_hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.nn.functional,
            self._is_callable,
            self._callable_hook_creator,
        )
        self._hook_manager.attach_hook(
            torch.Tensor,
            'backward',
            self._backward_hook_creator,
        )

    def stop_tracking(self):
        super().stop_tracking()
        self._hook_manager.remove_hooks()

    def _populate_report(self, report_builder):
        pass

    def _callable_hook_creator(self, func):
        def hook(*args, **kwargs):
            retval = func(*args, **kwargs)

            # Early return for tensor-producing operations that are not
            # involved in the backward pass
            if (not isinstance(retval, torch.Tensor) and
                    not isinstance(retval, tuple) and
                    not isinstance(retval, list)):
                return retval
            if (isinstance(retval, torch.Tensor) and
                    (not retval.is_cuda or retval.grad_fn is None)):
                return retval

            context = OperationContext(
                operation_name=func.__name__,
                stack=CallStack.from_here(start_from=2),
            )
            self._handle_callable_result(retval, context)
            return retval

        return hook

    def _backward_hook_creator(self, func):
        def hook(*args, **kwargs):
            print('Tensor.backward() called')
            return func(*args, **kwargs)
        return hook

    def _handle_callable_result(self, retval, context):
        if isinstance(retval, torch.Tensor) and retval.grad_fn is not None:
            self._grad_function_contexts[retval.grad_fn] = context

        elif isinstance(retval, tuple) or isinstance(retval, list):
            for inner_value in retval:
                self._handle_callable_result(inner_value, context)

    def _is_callable(self, maybe_fn):
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
