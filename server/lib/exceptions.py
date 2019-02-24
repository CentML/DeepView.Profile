

class AnalysisError(Exception):
    def __init__(self, *args, **kwargs):
        super(AnalysisError, self).__init__(*args, **kwargs)
