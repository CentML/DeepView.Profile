import torch
import contextlib

from tracker._base import _TrackerBase
from tracker._weights import _WeightsTracker
from tracker._iteration import _IterationTracker
from tracker.report import TrackerReportBuilder


class MemoryTracker(_TrackerBase):
    def __init__(self):
        super().__init__()
        self._weights_tracker = None
        self._iteration_tracker = None

    @contextlib.contextmanager
    def track(self):
        self.start_tracking()
        try:
            yield self
        finally:
            self.stop_tracking()

    @contextlib.contextmanager
    def model_instantiation(self):
        self._ensure_in_tracking_mode()
        self._weights_tracker.start_tracking()
        try:
            yield None
        finally:
            self._weights_tracker.stop_tracking()

    def track_iteration(self, run_forward):
        self._ensure_in_tracking_mode()
        self._iteration_tracker.start_tracking()
        try:
            out = run_forward()
        finally:
            self._iteration_tracker.stop_tracking()

        print('AFTER FORWARD:', torch.cuda.memory_allocated())
        self._iteration_tracker.extract_gradient_functions(out)
        del out
        self._iteration_tracker.extract_memory_usage()

    def get_report(self, output_file=None):
        if self._is_tracking:
            raise ValueError(
                'Tracking reports are only available after tracking '
                'completes. Please ensure get_report() is only called outside '
                'the tracking context manager.'
            )
        if self._weights_tracker is None or self._iteration_tracker is None:
            raise ValueError(
                'Tracking reports are only available after at least one '
                'tracking pass.'
            )
        report_builder = TrackerReportBuilder(output_file)
        self.populate_report(report_builder)
        return report_builder.build()

    def start_tracking(self):
        super().start_tracking()
        self._weights_tracker = _WeightsTracker()
        self._iteration_tracker = _IterationTracker()
        self._ensure_cuda_initialization()

    def populate_report(self, report_builder):
        self._weights_tracker._populate_report(report_builder)
        self._iteration_tracker._populate_report(report_builder)

    def _ensure_cuda_initialization(self):
        tensor = torch.randn((1,), device=torch.device('cuda'))

    def _ensure_in_tracking_mode(self):
        if not self._is_tracking:
            raise ValueError(
                'This mode can only be entered within a tracking context. '
                'Please make sure that you are using a "with tracker.track():"'
                'context manager.'
            )
