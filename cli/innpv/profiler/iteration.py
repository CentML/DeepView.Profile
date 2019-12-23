import logging

import torch

from innpv.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class IterationProfiler:
    def __init__(self, iteration, input_provider):
        self._iteration = iteration
        self._input_provider = input_provider
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    @classmethod
    def new_from(cls, model_provider, input_provider, iteration_provider):
        model = model_provider()
        iteration = iteration_provider(model)
        return cls(iteration, input_provider)

    def measure_run_time_ms(self, batch_size, initial_repetitions=None):
        """
        Measures the iteration run time in milliseconds.

        NOTE: This method will raise a RuntimeError if there is not enough GPU
              memory to run the iteration.
        """
        inputs = self._input_provider(batch_size=batch_size)

        # Warm up
        self._iteration(*inputs)
        torch.cuda.synchronize()

        def measure(iterations):
            self._start_event.record()
            for _ in range(iterations):
                self._iteration(*inputs)
            self._end_event.record()
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event)

        # When measuring the iteration run time of the model, we want to use as
        # few repetitions as possible. The problem is that we do not know how
        # many repetitions we need ahead of time.
        #
        # So the idea here is to iteratively double the number of repetitions
        # we use until we get a stable measurement (less than 5% difference).
        # If the caller knows how many measurements to use, they can pass it in
        # and we'll start from there. We will stop after reaching 100
        # iterations (or 2 * initial_repetitions), whichever is larger.
        #
        # We will return the smaller number of repetitions. This can be used by
        # future callers as the value of the initial_repetitions argument.
        repetitions = 5 if initial_repetitions is None else initial_repetitions
        max_repetitions = (
            50 if initial_repetitions is None else max(50, initial_repetitions)
        )
        lesser = measure(repetitions) / repetitions
        logger.debug("Iters: %d, Measured: %f", repetitions, lesser)
        while repetitions <= max_repetitions:
            doubled = repetitions * 2
            greater = measure(doubled) / doubled
            logger.debug("Iters: %d, Measured: %f", doubled, greater)

            # Stop when the difference between the measurements is less than 5%
            if (max(lesser, greater) / min(lesser, greater)) < 1.05:
                break

            repetitions = doubled
            lesser = greater

        return min(lesser, greater), repetitions

    def measure_throughput(self, batch_size):
        try:
            run_time_ms, repetitions = self.measure_run_time_ms(batch_size)
            return batch_size / run_time_ms * 1000
        except RuntimeError as ex:
            message = str(ex)
            if 'CUDA out of memory' in message:
                raise AnalysisError(message, type(ex)) from ex
            else:
                raise
