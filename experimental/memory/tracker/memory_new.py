import torch

from tracker._weights import _WeightsTracker
from tracker.activations import ActivationsTracker
from tracker.report import TrackerReportBuilder, MiscSizeType


def track_memory_usage(model_provider, input_provider, report_file=None):
    _ensure_cuda_initialization()

    # Track and record memory usage associated with model creation
    weight_tracker = _WeightsTracker()
    with weight_tracker.track():
        model = model_provider()

    def run_iteration():
        output = model(*input_provider())
        output.backward()

    # Run one iteration to initialize the gradients
    run_iteration()

    # Track and record memory usage associated with stored activations
    activations_tracker = ActivationsTracker()
    activations_tracker.track_memory_usage(model, input_provider)

    # Record peak memory usage
    torch.cuda.reset_max_memory_allocated()
    run_iteration()
    peak_usage_bytes = torch.cuda.max_memory_allocated()

    # Store our tracking results
    report_builder = TrackerReportBuilder(report_file)
    report_builder.add_misc_entry(
        MiscSizeType.PeakUsageBytes, peak_usage_bytes)
    weight_tracker.populate_report(report_builder)
    activations_tracker.populate_report(report_builder)

    return report_builder.build()


def _ensure_cuda_initialization():
    if not torch.cuda.is_available():
        raise ValueError(
            'The memory tracker is only available for CUDA devices.')
    tensor = torch.randn((1,), device=torch.device('cuda'))
