import contextlib
import sqlite3


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

    def populate_report(self, builder):
        raise NotImplementedError


class ReportBase:
    def __init__(self, connection):
        self._connection = connection

    def __del__(self):
        self._connection.close()


class ReportBuilderBase:
    def __init__(self, file=None):
        database_file = file if file is not None else ':memory:'
        self._connection = sqlite3.connect(database_file)
        self._create_report_tables()

    def process_tracker(self, tracker):
        tracker.populate_report(self)
        return self

    def build(self):
        raise NotImplementedError

    def _create_report_tables(self):
        raise NotImplementedError
