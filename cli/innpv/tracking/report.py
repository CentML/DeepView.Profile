import enum
import sqlite3

import innpv.tracking.report_queries as queries


class EntryType(enum.Enum):
    Weight = 1
    Activation = 2


class MiscSizeType(enum.Enum):
    PeakUsageBytes = 'peak_usage_bytes'


class TrackerReport:
    def __init__(self, connection):
        self._connection = connection


class TrackerReportBuilder:
    # This is the memory tracking report file format version that will be
    # created by this builder. When changes are made to the file format, this
    # integer should be increased monotonically.
    #
    # We need to version these tracking reports to protect us from future
    # changes to the file format.
    Version = 1

    def __init__(self, file=None):
        database_file = file if file is not None else ':memory:'
        self._connection = sqlite3.connect(database_file)
        self._create_report_tables()

    def process_tracker(self, tracker):
        tracker.populate_report(self)
        return self

    def add_weight_entry(
            self, name, size_bytes, grad_size_bytes, stack_context):
        cursor = self._connection.cursor()
        cursor.execute(
            queries.add_weight_entry, (name, size_bytes, grad_size_bytes))
        self._add_stack_frames(
            cursor=cursor,
            entry_id=cursor.lastrowid,
            entry_type=EntryType.Weight,
            stack_context=stack_context,
        )
        return self

    def add_activation_entry(self, name, size_bytes, stack_context):
        cursor = self._connection.cursor()
        cursor.execute(queries.add_activation_entry, (name, size_bytes))
        self._add_stack_frames(
            cursor=cursor,
            entry_id=cursor.lastrowid,
            entry_type=EntryType.Activation,
            stack_context=stack_context,
        )
        return self

    def add_misc_entry(self, size_type: MiscSizeType, size_bytes):
        cursor = self._connection.cursor()
        cursor.execute(queries.add_misc_entry, (size_type.value, size_bytes))
        return self

    def build(self):
        self._connection.commit()
        return TrackerReport(self._connection)

    def _create_report_tables(self):
        cursor = self._connection.cursor()
        cursor.execute(queries.set_report_format_version.format(
            version=TrackerReportBuilder.Version))
        for creation_query in queries.create_report_tables.values():
            cursor.execute(creation_query)
        cursor.executemany(
            queries.add_entry_type,
            map(lambda entry: (entry.value, entry.name), EntryType),
        )
        self._connection.commit()

    def _add_stack_frames(
            self, cursor, entry_id, entry_type: EntryType, stack_context):
        cursor.execute(
            queries.add_correlation_entry, (entry_id, entry_type.value))
        correlation_id = cursor.lastrowid

        def stack_frame_generator():
            for idx, frame in enumerate(stack_context.frames):
                yield (correlation_id, idx, frame.file_name, frame.lineno)

        cursor.executemany(queries.add_stack_frame, stack_frame_generator())
