import torch

from skyline.tracking.memory.activations import ActivationsTracker
from skyline.tracking.memory.report import MemoryReportBuilder, MiscSizeType
from skyline.tracking.memory.weights import WeightsTracker
from skyline.user_code_utils import user_code_environment


def track_memory_usage(
    model_provider,
    input_provider,
    iteration_provider,
    user_code_path,
    report_file=None,
):
    _ensure_cuda_initialization()

    # Track and record memory usage associated with model creation
    weight_tracker = WeightsTracker()
    with weight_tracker.track(), user_code_environment(user_code_path):
        model = model_provider()

    with user_code_environment(user_code_path):
        iteration = iteration_provider(model)
        # Run one iteration to initialize the gradients
        iteration(*input_provider())

    # Track and record memory usage associated with stored activations
    activations_tracker = ActivationsTracker()
    activations_tracker.track_memory_usage(
        model, input_provider, user_code_path)

    # Record peak memory usage
    torch.cuda.reset_max_memory_allocated()
    with user_code_environment(user_code_path):
        iteration(*input_provider())
    peak_usage_bytes = torch.cuda.max_memory_allocated()

    # Store our tracking results
    return (MemoryReportBuilder(report_file)
            .process_tracker(weight_tracker)
            .process_tracker(activations_tracker)
            .add_misc_entry(MiscSizeType.PeakUsageBytes, peak_usage_bytes)
            .build())


def _ensure_cuda_initialization():
    if not torch.cuda.is_available():
        raise ValueError(
            'The memory tracker is only available for CUDA devices.')
    tensor = torch.randn((1,), device=torch.device('cuda'))
