import logging

import torch

logger = logging.getLogger(__name__)


def measure_throughput(
        model_provider, input_provider, iteration_provider, batch_size):
    model = model_provider()
    iteration = iteration_provider(model)

    initial_run_time_ms, repetitions = measure_run_time_ms(
        iteration, input_provider, batch_size)

    return batch_size / initial_run_time_ms * 1000


def measure_run_time_ms(
        iteration, input_provider, batch_size, initial_repetitions=None):
    """
    Measures the iteration run time in milliseconds.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    inputs = input_provider(batch_size=batch_size)

    # Warm up
    iteration(*inputs)
    torch.cuda.synchronize()

    def measure(iterations):
        start_event.record()
        for _ in range(iterations):
            iteration(*inputs)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)

    # When measuring the iteration run time of the model, we want to use as few
    # repetitions as possible. The problem is that we do not know how many
    # repetitions we need ahead of time.
    #
    # So the idea here is to iteratively double the number of repetitions we
    # use until we get a stable measurement (less than 5% difference). If the
    # caller knows how many measurements to use, they can pass it in and we'll
    # start from there. We will stop after reaching 100 iterations (or 2 *
    # initial_repetitions), whichever is larger.
    #
    # We will return the smaller number of repetitions. This can be used by
    # future callers as the value of the initial_repetitions argument.
    repetitions = 5 if initial_repetitions is None else initial_repetitions
    max_repetitions = (
        50 if initial_repetitions is None else max(50, initial_repetitions)
    )
    lesser = measure(repetitions) / repetitions
    logging.debug("Iters: %d, Measured: %f", repetitions, lesser)
    while repetitions <= max_repetitions:
        doubled = repetitions * 2
        greater = measure(doubled) / doubled
        logging.debug("Iters: %d, Measured: %f", doubled, greater)

        # Stop when the difference between the measurements is less than 5%
        if (max(lesser, greater) / min(lesser, greater)) < 1.05:
            break

        repetitions = doubled
        lesser = greater

    return min(lesser, greater), repetitions
