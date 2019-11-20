import contextlib


class TrackerBase:
    def __init__(self):
        self._is_tracking = False

    @contextlib.contextmanager
    def track(self):
        self.start_tracking()
        try:
            yield self
        finally:
            self.stop_tracking()

    def start_tracking(self):
        self._is_tracking = True

    def stop_tracking(self):
        self._is_tracking = False
