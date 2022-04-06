import torch
import logging

from skyline.profiler.autograd import AutogradEngine
from skyline.profiler.backward import BackwardHelper, backward_available

logger = logging.getLogger(__name__)


class OperationProfiler:
    def __init__(self, warm_up=3, measure_for=3):
        self._warm_up = warm_up
        self._measure_for = measure_for
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def measure_operation_ms(self, func, args, kwargs):
        for_inplace = _is_potentially_inplace(func.__name__)

        forward_args, forward_kwargs = self._get_args_for_profiling(
            args, kwargs, for_inplace)
        def forward_runnable():
            func(*forward_args, **forward_kwargs)
        forward_ms = self._measure_ms(forward_runnable)

        # We need separate copies of the arguments for the forward and backward
        # measurements because func might be inplace. Running an inplace
        # function repeatedly will affect the autograd graph, which causes
        # problems when we try to measure the backward pass.
        backward_args, backward_kwargs = self._get_args_for_profiling(
            args, kwargs, for_inplace)
        retval = func(*backward_args, **backward_kwargs)
        if not backward_available(retval):
            return forward_ms, None

        return forward_ms, self._measure_backward_ms(func.__name__, retval)

    def _get_args_for_profiling(self, args, kwargs, for_inplace=False):
        cloned_args = tuple(map(
            lambda arg: self._clone_tensors(arg, for_inplace), args))
        cloned_kwargs = {
            key: self._clone_tensors(value, for_inplace)
            for key, value in kwargs.items()
        }
        return cloned_args, cloned_kwargs

    def _clone_tensors(self, argument, for_inplace):
        if isinstance(argument, torch.Tensor):
            detached = argument.detach()
            detached.requires_grad_(argument.requires_grad)
            # We need to clone the tensor for inplace operations because they
            # cannot be executed on a leaf tensor. This adds some overhead to
            # our backward measurements (an extra CloneBackward function), but
            # it _should_ be negligible. I chose not to exclude CloneBackward
            # from the backward measurements to avoid introducing incorrectness
            # if the user actually uses clone() in their own code.
            return detached if not for_inplace else detached.clone()

        if isinstance(argument, tuple):
            return tuple(map(
                lambda arg: self._clone_tensors(arg, for_inplace), argument))

        if isinstance(argument, list):
            return list(map(
                lambda arg: self._clone_tensors(arg, for_inplace), argument))

        return argument

    def _measure_backward_ms(self, func_name, operation_outputs):
        # As of PyTorch 1.5.1, sometimes our AutogradEngine will not work
        # because the behavior of grad_functions has changed. When this
        # happens, we fall back to using PyTorch's existing backward engine.
        #
        # The reason we do not always use the existing backward engine is that
        # the existing engine has a start up overhead that is non-negligible
        # when there are many short operations involved.
        try:
            return self._measure_backward_engine_strategy(operation_outputs)
        except (RuntimeError, TypeError):
            logger.debug("%s: Falling back to PyTorch's engine", func_name)
            return self._measure_backward_torch_strategy(operation_outputs)

    def _measure_backward_engine_strategy(self, operation_outputs):
        engine = AutogradEngine.new_from(operation_outputs)
        return self._measure_ms(engine.run_backward)

    def _measure_backward_torch_strategy(self, operation_outputs):
        helper = BackwardHelper.new_from(operation_outputs)

        backward_ms = self._measure_ms(helper.run_backward)
        accum_grad_ms = self._measure_ms(helper.run_accumulate_grad)
        diff = backward_ms - accum_grad_ms

        return diff if diff >= 1e-6 else backward_ms

    def _measure_ms(self, runnable):
        for _ in range(self._warm_up):
            runnable()

        self._start_event.record()
        for _ in range(self._measure_for):
            runnable()
        self._end_event.record()
        torch.cuda.synchronize()

        return (
            self._start_event.elapsed_time(self._end_event) / self._measure_for
        )


# Populated manually from:
# https://pytorch.org/docs/stable/nn.functional.html
POTENTIALLY_INPLACE_FUNCTIONS = {
    'threshold',
    'relu',
    'hardtanh',
    'relu6',
    'elu',
    'selu',
    'celu',
    'leaky_relu',
    'rrelu',
    'dropout',
    'alpha_dropout',
    'dropout2d',
    'dropout3d',

    # In place math operations (+=, *=, -=, /=, //=)
    '__iadd__',
    '__imul__',
    '__isub__',
    '__itruediv__',
    '__ifloordiv__',
}


def _is_potentially_inplace(fn_name):
    return (
        fn_name in POTENTIALLY_INPLACE_FUNCTIONS or
        # In PyTorch, functions with a '_' suffix are in place, by convention
        (len(fn_name) > 1 and fn_name[-1] == '_' and fn_name[-2] != '_')
    )
