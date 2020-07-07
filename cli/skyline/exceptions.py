import collections
import contextlib
import os
import traceback

FileContext = collections.namedtuple(
    'FileContext',
    ['file_path', 'line_number'],
)


class AnalysisError(RuntimeError):
    def __init__(self, message, exception_type=None):
        if exception_type is None:
            super(AnalysisError, self).__init__(message)
        else:
            super(AnalysisError, self).__init__(
                '{}: {}'.format(exception_type.__name__, message))

        self.file_context = None

    def with_file_context(self, file_path, line_number=None):
        self.file_context = FileContext(
            file_path=file_path,
            line_number=line_number,
        )
        return self


class NoConnectionError(Exception):
    def __init__(self, message):
        super().__init__(message)


@contextlib.contextmanager
def exceptions_as_analysis_errors(project_root):
    try:
        yield
    except Exception as ex:
        error = AnalysisError(str(ex), type(ex))

        # Extract the relevant file context, if it is available, starting by
        # inspecting the exception itself.
        if hasattr(ex, 'filename') and ex.filename.startswith(project_root):
            _add_context_to_error(
                error, project_root, ex.filename, getattr(ex, 'lineno', None))
        else:
            stack = traceback.extract_tb(ex.__traceback__)
            for frame in reversed(stack):
                if frame.filename.startswith(project_root):
                    _add_context_to_error(
                        error, project_root, frame.filename, frame.lineno)
                    break

        raise error


def _add_context_to_error(error, project_root, file_path, line_number):
    error.with_file_context(
        file_path=os.path.relpath(file_path, start=project_root),
        line_number=line_number,
    )
