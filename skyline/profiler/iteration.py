import collections
import logging

import torch

from skyline.exceptions import AnalysisError
from skyline.user_code_utils import user_code_environment

logger = logging.getLogger(__name__)

IterationSample = collections.namedtuple(
    "IterationSample", ["batch_size", "run_time_ms", "peak_usage_bytes"])


class IterationProfiler:
    def __init__(
        self,
        iteration,
        input_provider,
        path_to_entry_point_dir,
        project_root
    ):
        self._iteration = iteration
        self._input_provider = input_provider
        self._path_to_entry_point_dir = path_to_entry_point_dir
        self._project_root = project_root
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    @classmethod
    def new_from(
        cls,
        model_provider,
        input_provider,
        iteration_provider,
        path_to_entry_point_dir,
        project_root,
    ):
        with user_code_environment(path_to_entry_point_dir, project_root):
            model = model_provider()
            iteration = iteration_provider(model)
        return cls(
            iteration, input_provider, path_to_entry_point_dir, project_root)

    def measure_run_time_ms(self, batch_size, initial_repetitions=None):
        """
        Measures the iteration run time in milliseconds.

        NOTE: This method will raise a RuntimeError if there is not enough GPU
              memory to run the iteration.
        """
        with user_code_environment(
                self._path_to_entry_point_dir, self._project_root):
            inputs = self._input_provider(batch_size=batch_size)
            # Warm up
            self._iteration(*inputs)

        torch.cuda.synchronize()

        def measure(iterations):
            with user_code_environment(
                    self._path_to_entry_point_dir, self._project_root):
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
        repetitions = 3 if initial_repetitions is None else initial_repetitions
        max_repetitions = (
            50 if initial_repetitions is None else max(50, initial_repetitions)
        )

        torch.cuda.reset_max_memory_allocated()
        lesser = measure(repetitions) / repetitions
        peak_usage_bytes = torch.cuda.max_memory_allocated()

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

        return min(lesser, greater), peak_usage_bytes, repetitions

    def measure_run_time_ms_catch_oom(
            self, batch_size, initial_repetitions=None):
        # This function is useful when we want to explicitly handle OOM errors
        # without aborting the profiling.
        try:
            return (
                None,
                self.measure_run_time_ms(batch_size, initial_repetitions),
            )
        except AnalysisError as ex:
            message = str(ex)
            if 'CUDA out of memory' in message:
                return (ex, None)
            else:
                raise

    def sample_run_time_ms_by_batch_size(
        self,
        start_batch_size,
        start_batch_size_run_time_ms=None,
        start_batch_size_peak_usage_bytes=None,
        memory_usage_percentage=None,
        num_samples=3,
    ):
        samples = []

        # 1. Make sure we can measure the run time of the "start" batch size
        if (start_batch_size_run_time_ms is None or
                start_batch_size_peak_usage_bytes is None):
            start_run_time_ms, start_peak_usage_bytes, _ = \
                self.measure_run_time_ms(start_batch_size)
        else:
            start_run_time_ms = start_batch_size_run_time_ms
            start_peak_usage_bytes = start_batch_size_peak_usage_bytes

        samples.append(IterationSample(
            start_batch_size, start_run_time_ms, start_peak_usage_bytes))

        # 2. Perform sampling. We keep a range of "viable" batch sizes, where
        #    the upper limit is a guess on what will fit in memory. We adjust
        #    these limits as we sample.
        #
        #    We estimate the maximum batch size by assuming a linear
        #    relationship between the model's memory use and its batch size.
        #    The memory use at the initial batch size is obtained through
        #    memory tracking. If it is not specified, we just add a constant
        max_batch_size = (
            start_batch_size / memory_usage_percentage
            if memory_usage_percentage is not None
            else start_batch_size + 100
        )

        if len(samples) < num_samples:
            samples.extend(self._sample_range(
                start_batch_size,
                max_batch_size,
                num_samples=(num_samples - len(samples)),
                is_increasing=True,
            ))

        if len(samples) < num_samples:
            samples.extend(self._sample_range(
                1,
                start_batch_size,
                num_samples=(num_samples - len(samples)),
                is_increasing=False,
            ))

        return samples

    def _sample_range(
            self, min_size, max_size, num_samples, is_increasing=True):
        # The idea here is to sample the range of possible batch sizes by
        # recursively narrowing down the acceptable ranges of batch sizes.

        samples = []
        stack = [(min_size, max_size)]

        while len(samples) < num_samples and len(stack) > 0:
            lower, upper = stack.pop()
            if lower >= upper:
                continue

            next_size = self._select_batch_size(lower, upper, is_increasing)
            logger.debug(
                "[%d, %d] Sampling batch size: %d", lower, upper, next_size)
            err, result = self.measure_run_time_ms_catch_oom(next_size)
            if err is not None:
                stack.append((lower, next_size - 1))
                continue

            samples.append(IterationSample(next_size, result[0], result[1]))

            # Change the order in which we explore each range
            if is_increasing:
                stack.append((lower, next_size - 1))
                stack.append((next_size + 1, upper))
            else:
                stack.append((next_size + 1, upper))
                stack.append((lower, next_size - 1))

        return samples

    def _select_batch_size(self, lower, upper, is_increasing):
        diff = upper - lower
        base = lower if is_increasing else upper
        mult = 1 if is_increasing else -1

        if diff >= 20:
            return base + mult * 20
        elif diff >= 10:
            return base + mult * 10
        elif diff >= 5:
            return base + mult * 5
        else:
            return base + mult * 1
