import contextlib


class AnalysisError(Exception):
    def __init__(self, message, exception_type=None):
        if exception_type is None:
            super(AnalysisError, self).__init__(message)
        else:
            super(AnalysisError, self).__init__(
                '{}: {}'.format(exception_type.__name__, message))


class NoConnectionError(Exception):
    def __init__(self, message):
        super().__init__(message)


@contextlib.contextmanager
def exceptions_as_analysis_errors():
    try:
        yield
    except Exception as ex:
        raise AnalysisError(str(ex), type(ex))
