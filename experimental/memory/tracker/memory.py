import torch
import contextlib

from tracker._base import _TrackerBase
from tracker._weights import _WeightsTracker
from tracker._iteration import _IterationTracker


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

    @contextlib.contextmanager
    def training_iteration(self):
        self._ensure_in_tracking_mode()
        self._iteration_tracker.start_tracking()
        try:
            yield None
        finally:
            self._iteration_tracker.stop_tracking()

    def get_report(self):
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
        return self._weights_tracker.get_report()

    def start_tracking(self):
        super().start_tracking()
        self._weights_tracker = _WeightsTracker()
        self._iteration_tracker = _IterationTracker()
        self._ensure_cuda_initialization()

    def stop_tracking(self):
        super().stop_tracking()

    def _ensure_cuda_initialization(self):
        tensor = torch.randn((1,), device=torch.device('cuda'))

    def _ensure_in_tracking_mode(self):
        if not self._is_tracking:
            raise ValueError(
                'This mode can only be entered within a tracking context. '
                'Please make sure that you are using a "with tracker.track():"'
                'context manager.'
            )
