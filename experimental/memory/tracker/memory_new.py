import torch

from tracker._weights import _WeightsTracker
from tracker.activations import ActivationsTracker
from tracker.report import TrackerReportBuilder


def track_memory_usage(model_provider, input_provider, report_file=None):
    _ensure_cuda_initialization()

    # Track and record memory usage associated with model creation
    weight_tracker = _WeightsTracker()
    with weight_tracker.track():
        model = model_provider()

    # Track and record memory usage associated with stored activations
    activations_tracker = ActivationsTracker()
    activations_tracker.track_memory_usage(model, input_provider)

    # Store our tracking results
    report_builder = TrackerReportBuilder(report_file)
    weight_tracker.populate_report(report_builder)
    activations_tracker.populate_report(report_builder)

    return report_builder.build()


def _ensure_cuda_initialization():
    if not torch.cuda.is_available():
        raise ValueError(
            'The memory tracker is only available for CUDA devices.')
    tensor = torch.randn((1,), device=torch.device('cuda'))
