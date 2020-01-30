import collections
import os

from skyline.tracking.base import ReportBase, ReportBuilderBase
import skyline.tracking.time.report_queries as queries

RunTimeEntry = collections.namedtuple(
    'RunTimeEntry',
    ['operation_name',
     'forward_ms',
     'backward_ms',
     'file_path',
     'line_number'],
)


class OperationRunTimeReport(ReportBase):
    def __init__(self, connection):
        super().__init__(connection)

    def get_run_time_entries(self, path_prefix=None):
        cursor = self._connection.cursor()
        return map(
            lambda row: RunTimeEntry(*row),
            cursor.execute(queries.get_run_time_entries_with_context),
        )


class OperationRunTimeReportBuilder(ReportBuilderBase):
    # This is the operation run time tracking report file format version that
    # will be created by this builder. When changes are made to the file
    # format, this integer should be increased monotonically.
    #
    # We need to version these tracking reports to protect us from future
    # changes to the file format.
    Version = 1

    def __init__(self, file=None):
        super().__init__(file)

    def add_run_time_entry(
            self, operation_name, forward_ms, backward_ms, stack_context):
        cursor = self._connection.cursor()
        cursor.execute(queries.add_run_time_entry, (
            operation_name,
            forward_ms,
            backward_ms,
        ))
        entry_id = cursor.lastrowid

        def stack_frame_generator():
            for idx, frame in enumerate(stack_context.frames):
                yield (idx, frame.file_path, frame.line_number, entry_id)

        cursor.executemany(queries.add_stack_frame, stack_frame_generator())

    def build(self):
        self._connection.commit()
        return OperationRunTimeReport(self._connection)

    def _create_report_tables(self):
        cursor = self._connection.cursor()
        cursor.execute(queries.set_report_format_version.format(
            version=OperationRunTimeReportBuilder.Version))
        for creation_query in queries.create_report_tables.values():
            cursor.execute(creation_query)
        self._connection.commit()
