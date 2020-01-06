import torch

from skyline.exceptions import exceptions_as_analysis_errors
from skyline.tracking.activations import ActivationsTracker
from skyline.tracking.report import TrackerReportBuilder, MiscSizeType
from skyline.tracking.weights import WeightsTracker


def track_memory_usage(
        model_provider, input_provider, iteration_provider, report_file=None):
    _ensure_cuda_initialization()

    # Track and record memory usage associated with model creation
    weight_tracker = WeightsTracker()
    with weight_tracker.track(), exceptions_as_analysis_errors():
        model = model_provider()

    with exceptions_as_analysis_errors():
        iteration = iteration_provider(model)
        # Run one iteration to initialize the gradients
        iteration(*input_provider())

    # Track and record memory usage associated with stored activations
    activations_tracker = ActivationsTracker()
    activations_tracker.track_memory_usage(model, input_provider)

    # Record peak memory usage
    torch.cuda.reset_max_memory_allocated()
    with exceptions_as_analysis_errors():
        iteration(*input_provider())
    peak_usage_bytes = torch.cuda.max_memory_allocated()

    # Store our tracking results
    return (TrackerReportBuilder(report_file)
            .process_tracker(weight_tracker)
            .process_tracker(activations_tracker)
            .add_misc_entry(MiscSizeType.PeakUsageBytes, peak_usage_bytes)
            .build())


def _ensure_cuda_initialization():
    if not torch.cuda.is_available():
        raise ValueError(
            'The memory tracker is only available for CUDA devices.')
    tensor = torch.randn((1,), device=torch.device('cuda'))
