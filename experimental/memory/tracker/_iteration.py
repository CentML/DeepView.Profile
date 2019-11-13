from tracker._base import _TrackerBase
from tracker.meters import MemoryMeter
from hook_manager import HookManager


class _IterationTracker(_TrackerBase):
    def __init__(self):
        super().__init__()
        self._hook_manager = HookManager()
        self._meter = MemoryMeter()

    def start_tracking(self):
        super().start_tracking()
        self._meter.reset()

    def stop_tracking(self):
        super().stop_tracking()

    def get_report(self):
        return []
