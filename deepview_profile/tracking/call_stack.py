import collections
import inspect
import os
import re
import torch
from deepview_profile.utils import model_location_patterns

SourceLocation = collections.namedtuple(
    "SourceLocation", ["file_path", "line_number", "module_id"]
)

def find_pattern_match(filename):
    pattern_list = model_location_patterns()
    return any(re.search(pattern, filename) for pattern in pattern_list)

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
                # Only track source locations that are within the project model (or transformers, diffusers, etc)
                # that are within a torch.nn.Module. Note that we assume the
                # user uses "self" to refer to the current class instance.

                if not (
                    frame_info.filename.startswith(project_root)
                    or find_pattern_match(frame_info.filename)
                ):
                    continue
                if "self" not in frame_info.frame.f_locals:
                    continue
                if not isinstance(frame_info.frame.f_locals["self"], torch.nn.Module):
                    continue
                context.append(
                    SourceLocation(
                        file_path=os.path.relpath(
                            frame_info.filename, start=project_root
                        ),
                        line_number=frame_info.lineno,
                        module_id=id(frame_info.frame.f_locals["self"]),
                    )
                )
            return CallStack(context)
        finally:
            del stack
