import collections
import inspect


SourceLocation = collections.namedtuple(
    'SourceLocation', ['file_name', 'lineno'])


class CallStack:
    def __init__(self, frames):
        self.frames = frames

    @staticmethod
    def from_here(start_from=1):
        """
        Returns the current call stack when invoked.
        """
        stack = inspect.stack()
        context = []
        try:
            for frame_info in stack[start_from:]:
                context.append(SourceLocation(
                    file_name=frame_info.filename, lineno=frame_info.lineno))
            return CallStack(context)
        finally:
            del stack
