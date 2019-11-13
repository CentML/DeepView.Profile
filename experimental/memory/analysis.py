import contextlib
import weakref
import inspect

from hook_manager import HookManager


class MemoryMeter:
    def __init__(self, torch):
        self._torch = torch
        self.reset()

    def reset(self):
        self._reference_usage_bytes = self._torch.cuda.memory_allocated()

    def checkpoint(self):
        current_usage_bytes = self._torch.cuda.memory_allocated()
        delta_bytes = current_usage_bytes - self._reference_usage_bytes
        self._reference_usage_bytes = current_usage_bytes
        return delta_bytes


class MemoryTracker:
    def __init__(self, torch):
        self._torch = torch
        self._hook_manager = HookManager()
        self._meter = MemoryMeter(torch)
        self._module_parameters = weakref.WeakKeyDictionary()

    @contextlib.contextmanager
    def track(self):
        self.start_tracking()
        try:
            yield self
        finally:
            self.stop_tracking()

    def start_tracking(self):
        self._force_cuda_initialization()
        self._hook_manager.attach_hook(
            self._torch.nn.Module,
            'register_parameter',
            self._register_parameter_hook_creator,
        )

    def stop_tracking(self):
        self._hook_manager.remove_hooks()

    def get_weight_report(self):
        for param, (name, stack) in self._module_parameters.items():
            if not param.is_cuda:
                continue
            param_size_bytes = param.element_size() * param.numel()
            print(name, param_size_bytes)

    def _force_cuda_initialization(self):
        tensor = self._torch.randn((1,), device=self._torch.device('cuda'))

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
