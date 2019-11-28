

class AnalysisError(Exception):
    def __init__(self, message, exception_type=None):
        if exception_type is None:
            super(AnalysisError, self).__init__(message)
        else:
            super(AnalysisError, self).__init__(
                '{}: {}'.format(exception_type.__name__, message))
