import contextlib
import torch

from skyline.exceptions import _SuspendExecution
from skyline.tracking.hook_manager import HookManager


class BackwardInterceptor:
    def __init__(self):
        self._backward_hooks = HookManager()
        self.backward_root = None

    @contextlib.contextmanager
    def intercept(self):
        self._backward_hooks.attach_hook(
            torch.Tensor,
            'backward',
            self._hook_creator,
        )
        try:
            yield
        except _SuspendExecution:
            pass
        finally:
            self._backward_hooks.remove_hooks()

    def _hook_creator(self, fn):
        def hook(*args):
            self.backward_root = args[0]
            raise _SuspendExecution
        return hook
