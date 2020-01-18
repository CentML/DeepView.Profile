import torch


class OperationProfiler:
    def __init__(self, warm_up=3, measure_for=10):
        self._warm_up = warm_up
        self._measure_for = measure_for
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def measure_operation_ms(self, func, args, kwargs):
        cloned_args = tuple(map(lambda arg: self._clone_tensors(arg), args))
        cloned_kwargs = {
            key: self._clone_tensors(value) for key, value in kwargs.items()
        }

        def runnable():
            return func(*cloned_args, **cloned_kwargs)

        return self._measure_ms(runnable)

    def _clone_tensors(self, argument):
        if isinstance(argument, torch.Tensor):
            return argument.clone()

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
