import collections

import torch

from skyline.tracking.call_stack import CallStack
from skyline.tracking.base import TrackerBase
from skyline.tracking.callable_tracker import CallableTracker
from skyline.profiler.operation import OperationProfiler
from skyline.tracking.utils import flatten_operation_retval
from skyline.user_code_utils import user_code_environment

OperationInfo = collections.namedtuple(
    'OperationInfo', ['operation_name', 'stack', 'forward_ms', 'backward_ms'])


class IterationTracker(TrackerBase):
    def __init__(self):
        super().__init__()
        self._callable_tracker = CallableTracker(self._hook_creator)
        self._profiler = OperationProfiler()
        self._processing_hook = False

        self.operations = []

    def start_tracking(self):
        super().start_tracking()
        self._callable_tracker.start_tracking()

    def stop_tracking(self):
        super().stop_tracking()
        self._callable_tracker.stop_tracking()

    def _hook_creator(self, func):
        def hook(*args, **kwargs):
            # NOTE: We use self._processing_hook to handle cases where we have
            #       hooks on nested function calls.
            if self._processing_hook:
                return func(*args, **kwargs)

            self._processing_hook = True
            try:
                forward_ms = self._profiler.measure_operation_ms(
                    func, args, kwargs)
                retval = func(*args, **kwargs)

                grad_fn = self._get_grad_fn(retval)
                if grad_fn is not None:
                    grad_fn_inputs = flatten_operation_retval(retval)
                    backward_ms = self._profiler.measure_operation_ms(
                        grad_fn, grad_fn_inputs, {})
                else:
                    backward_ms = None

                self.operations.append(OperationInfo(
                    operation_name=func.__name__,
                    stack=CallStack.from_here(start_from=2),
                    forward_ms=forward_ms,
                    backward_ms=backward_ms,
                ))

            finally:
                self._processing_hook = False

            return retval
        return hook

    def _get_grad_fn(self, retval):
        if isinstance(retval, torch.Tensor) and retval.grad_fn is not None:
            return retval.grad_fn
        elif isinstance(retval, tuple) or isinstance(retval, list):
            for inner_value in retval:
                grad_fn = self._get_grad_fn(inner_value)
                if grad_fn is not None:
                    return grad_fn
        else:
            return None
