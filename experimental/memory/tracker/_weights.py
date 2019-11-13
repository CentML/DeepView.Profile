import torch
import inspect
import weakref

from tracker._base import _TrackerBase
from hook_manager import HookManager


class _WeightsTracker(_TrackerBase):
    def __init__(self):
        super().__init__()
        self._hook_manager = HookManager()
        self._module_parameters = weakref.WeakKeyDictionary()

    def start_tracking(self):
        super().start_tracking()
        self._hook_manager.attach_hook(
            torch.nn.Module,
            'register_parameter',
            self._register_parameter_hook_creator,
        )

    def stop_tracking(self):
        super().stop_tracking()
        self._hook_manager.remove_hooks()

    def get_report(self):
        for param, (name, stack) in self._module_parameters.items():
            if not param.is_cuda:
                continue
            param_size_bytes = param.element_size() * param.numel()
            print(name, param_size_bytes)
        return []

    def _register_parameter_hook_creator(self, func):
        def hook(*args, **kwargs):
            name = args[1]
            parameter = args[2]
            retval = func(*args, **kwargs)
            if parameter is not None:
                self._module_parameters[parameter] = (
                    name,
                    self._extract_stack_context(start_from=2),
                )
            return retval
        return hook

    def _extract_stack_context(self, start_from=0):
        stack = inspect.stack()
        context = []
        try:
            for frame_info in stack[start_from:]:
                context.append((frame_info.filename, frame_info.lineno))
            return context
        finally:
            del stack
