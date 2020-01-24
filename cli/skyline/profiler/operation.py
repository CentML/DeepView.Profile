import torch

from skyline.profiler.autograd import AutogradEngine


class OperationProfiler:
    def __init__(self, warm_up=3, measure_for=10):
        self._warm_up = warm_up
        self._measure_for = measure_for
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def measure_operation_ms(self, func, args, kwargs):
        forward_args, forward_kwargs = self._get_args_for_profiling(
            args, kwargs)
        def forward_runnable():
            func(*forward_args, **forward_kwargs)
        forward_ms = self._measure_ms(forward_runnable)

        # We need separate copies of the arguments for the forward and backward
        # measurements because func might be inplace. Running an inplace
        # function repeatedly will affect the autograd graph, which causes
        # problems when we try to measure the backward pass.
        backward_args, backward_kwargs = self._get_args_for_profiling(
            args, kwargs)
        retval = func(*backward_args, **backward_kwargs)
        if not AutogradEngine.backward_available(retval):
            return forward_ms, None

        engine = AutogradEngine.new_from(retval)
        def backward_runnable():
            engine.run_backward()

        return forward_ms, self._measure_ms(backward_runnable)

    def _get_args_for_profiling(self, args, kwargs):
        cloned_args = tuple(map(lambda arg: self._clone_tensors(arg), args))
        cloned_kwargs = {
            key: self._clone_tensors(value) for key, value in kwargs.items()
        }
        return cloned_args, cloned_kwargs

    def _clone_tensors(self, argument):
        if isinstance(argument, torch.Tensor):
            detached = argument.detach()
            detached.requires_grad_(argument.requires_grad)
            # We need to clone the tensor because the user may use an inplace
            # operation, which cannot be executed on a leaf tensor. This adds
            # some overhead to our backward measurements (an extra
            # CloneBackward function), but it _should_ be negligible. I chose
            # not to exclude CloneBackward from the backward measurements to
            # avoid introducing incorrectness if the user actually uses clone()
            # in their own code.
            return detached.clone()

        if isinstance(argument, tuple):
            return tuple(map(lambda arg: self._clone_tensors(arg), argument))

        if isinstance(argument, list):
            return list(map(lambda arg: self._clone_tensors(arg), argument))

        return argument

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
