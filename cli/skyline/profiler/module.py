import torch
import logging

from skyline.config import Config
from skyline.exceptions import AnalysisError

logger = logging.getLogger(__name__)


def get_operation_runtimes(
        model, annotation_info, model_operations, runtime_cache):
    try:
        # 1. Create CUDA events used for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 2. Attach profiling hooks to each relevant module
        for name, module in model.named_modules():
            operation_info = model_operations.get_operation_info_by_bound_name(
                name)
            if operation_info is None:
                continue

            module.register_forward_hook(_make_profiling_hook(
                operation_info, (start_event, end_event), runtime_cache))

        # 3. Perform one forward pass to trigger the profiling
        model_input = torch.randn(
            annotation_info.input_size, device=torch.device('cuda'))
        output = model(model_input)

    except Exception as ex:
        raise AnalysisError(str(ex), type(ex))


def _make_profiling_hook(operation_info, events, runtime_cache):
    # Use a flag to prevent recursive calls to the hook
    profiling_in_progress = False

    def _hook(module, inputs, outputs):
        nonlocal profiling_in_progress
        if profiling_in_progress:
            return

        # If we've profiled this exact module + input combination before, just
        # return the cached result
        runtime_us = runtime_cache.query(module, inputs)
        if runtime_us is not None:
            operation_info.add_to_runtime_us(runtime_us)
            logger.debug(
                'Using cached runtime for %s.',
                operation_info.bound_name,
            )
            return

        logger.debug(
            'Profiling %s...',
            operation_info.bound_name,
        )
        try:
            profiling_in_progress = True
            start_event, end_event = events
            fake_inputs = tuple(
                torch.randn_like(tensor, requires_grad=True)
                for tensor in inputs
            )

            # Measure the forward pass
            for _ in range(Config.warm_up):
                output = module(*fake_inputs)

            torch.cuda.synchronize()
            start_event.record()
            for _ in range(Config.measure_for):
                output = module(*fake_inputs)
            end_event.record()
            torch.cuda.synchronize()

            forward_time_us = 1000 * (
                start_event.elapsed_time(end_event) / Config.measure_for
            )
            operation_info.add_to_runtime_us(forward_time_us)

            # Measure the backward pass
            fake_grads = torch.ones_like(output)

            for _ in range(Config.warm_up):
                output.backward(fake_grads, retain_graph=True)

            torch.cuda.synchronize()
            start_event.record()
            for _ in range(Config.measure_for):
                output.backward(fake_grads, retain_graph=True)
            end_event.record()
            torch.cuda.synchronize()

            backward_time_us = 1000 * (
                start_event.elapsed_time(end_event) / Config.measure_for
            )
            operation_info.add_to_runtime_us(backward_time_us)

            runtime_cache.store(
                module, inputs, forward_time_us + backward_time_us)

        finally:
            profiling_in_progress = False

    return _hook
