

class _TrackerBase:
    def __init__(self):
        self._is_tracking = False

    def start_tracking(self):
        self._is_tracking = True

    def stop_tracking(self):
        self._is_tracking = False

    def get_report(self):
        raise NotImplementedError
