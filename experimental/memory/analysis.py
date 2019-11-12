import contextlib
import traceback
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

    @contextlib.contextmanager
    def track(self):
        self.start_tracking()
        try:
            yield self
        finally:
            self.stop_tracking()

    def start_tracking(self):
        self._force_cuda_initialization()
        self._meter.reset()
        self._hook_manager.attach_hook(
            self._torch.Tensor,
            '__new__',
            self._memory_allocation_hook_creator,
        )
        """
        self._hook_manager.attach_hook(
            self._torch.nn.Module,
            'register_parameter',
            self._memory_allocation_hook_creator,
        )
        """

    def stop_tracking(self):
        self._hook_manager.remove_hooks()

    def _force_cuda_initialization(self):
        tensor = self._torch.randn((1,), device=self._torch.device('cuda'))

    def _memory_allocation_hook_creator(self, func):
        def memory_allocation_hook(*args, **kwargs):
            print('Before:', func.__name__)
            retval = func(*args, **kwargs)
            print('After: ', func.__name__, self._meter.checkpoint())
            #traceback.print_stack()
            return retval
        return memory_allocation_hook
