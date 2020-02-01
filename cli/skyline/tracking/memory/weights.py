import torch
import inspect
import weakref

from skyline.tracking.base import TrackerBase
from skyline.tracking.call_stack import CallStack
from skyline.tracking.hook_manager import HookManager
from skyline.tracking.utils import tensor_size_bytes


class WeightsTracker(TrackerBase):
    def __init__(self, project_root):
        super().__init__()
        self._hook_manager = HookManager()
        self._module_parameters = weakref.WeakKeyDictionary()
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
            if parameter is not None:
                self._module_parameters[parameter] = (
                    name,
                    CallStack.from_here(self._project_root, start_from=2),
                )
            return retval
        return hook
