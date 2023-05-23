import torch
import inspect

from deepview_profile.tracking.base import TrackerBase
from deepview_profile.tracking.call_stack import CallStack
from deepview_profile.tracking.hook_manager import HookManager
from deepview_profile.tracking.utils import tensor_size_bytes
from deepview_profile.util_weak import WeakTensorKeyDictionary

class WeightsTracker(TrackerBase):
    def __init__(self, project_root):
        super().__init__()
        self._hook_manager = HookManager()
        self._module_parameters = WeakTensorKeyDictionary()
        self._project_root = project_root

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

    def populate_report(self, builder):
        for param, (name, stack) in self._module_parameters.items():
            if not param.is_cuda:
                continue
            builder.add_weight_entry(
                weight_name=name,
                size_bytes=tensor_size_bytes(param),
                grad_size_bytes=tensor_size_bytes(param.grad),
                stack_context=stack,
            )

    def populate_breakdown(self, builder):
        # The HierarchicalBreakdownBuilder uses the same API as the
        # MemoryReportBuilder.
        self.populate_report(builder)

    def _register_parameter_hook_creator(self, func):
        def hook(*args, **kwargs):
            name = args[1]
            parameter = args[2]
            retval = func(*args, **kwargs)
            if parameter is not None and parameter not in self._module_parameters:
                self._module_parameters[parameter] = (
                    name,
                    CallStack.from_here(self._project_root, start_from=2),
                )
            return retval
        return hook
