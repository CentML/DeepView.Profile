import collections
import inspect
import os

import torch

SourceLocation = collections.namedtuple(
    'SourceLocation', ['file_path', 'line_number', 'module_id'])


class CallStack:
    def __init__(self, frames):
        self.frames = frames

    @staticmethod
    def from_here(project_root, start_from=1):
        """
        Returns the current call stack when invoked.
        """
        stack = inspect.stack()
        context = []
        try:
            for frame_info in stack[start_from:]:
                # Only track source locations that are within the project and
                # that are within a torch.nn.Module. Note that we assume the
                # user uses "self" to refer to the current class instance.
                if not frame_info.filename.startswith(project_root):
                    continue
                if 'self' not in frame_info.frame.f_locals:
                    continue
                if not isinstance(
                        frame_info.frame.f_locals['self'], torch.nn.Module):
                    continue

                context.append(SourceLocation(
                    file_path=os.path.relpath(
                        frame_info.filename, start=project_root),
                    line_number=frame_info.lineno,
                    module_id=id(frame_info.frame.f_locals['self']),
                ))
            return CallStack(context)
        finally:
            del stack
