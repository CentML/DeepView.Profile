import torch
import inspect
import collections
import gc

from tracker._base import _TrackerBase
from tracker.call_stack import CallStack
from tracker.meters import MemoryMeter
from hook_manager import HookManager


OperationContext = collections.namedtuple(
    'OperationContext',
    ['operation_name', 'stack'],
)

IterationEntry = collections.namedtuple(
    'IterationEntry',
    ['operation_context', 'size_bytes']
)


class _IterationTracker(_TrackerBase):
    def __init__(self):
        super().__init__()
        self._hook_manager = HookManager()
        self._meter = MemoryMeter()
        self._grad_function_contexts = {}
        self._grad_functions = []
        self._iteration_entries = []
        self._sum = 0

    def start_tracking(self):
        super().start_tracking()
        self._meter.reset()
        self._grad_function_contexts.clear()
        self._grad_functions.clear()
        self._iteration_entries.clear()
        self._hook_manager.attach_hooks_on_module(
            torch,
            self._is_callable,
            self._callable_hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.Tensor,
            lambda fn: self._is_callable(fn) and fn.__name__ != 'backward',
            self._callable_hook_creator,
        )
        self._hook_manager.attach_hooks_on_module(
            torch.nn.functional,
            self._is_callable,
            self._callable_hook_creator,
        )

    def stop_tracking(self):
        super().stop_tracking()
        self._hook_manager.remove_hooks()
        print('TOTAL:', self._sum)

    def extract_gradient_functions(self, output_tensor):
        all_grad_fns = get_gradient_functions(output_tensor)
        relevant_grad_fns = []
        relevant_contexts = {}

        for grad_fn in all_grad_fns:
            if grad_fn not in self._grad_function_contexts:
                continue
            relevant_grad_fns.append(grad_fn)
            relevant_contexts[grad_fn] = self._grad_function_contexts[grad_fn]

        self._grad_functions = relevant_grad_fns
        self._grad_function_contexts = relevant_contexts

    def extract_memory_usage(self):
        self._grad_functions.reverse()

        while len(self._grad_functions) > 0:
            grad_fn = self._grad_functions.pop()
            context = self._grad_function_contexts[grad_fn]
            del self._grad_function_contexts[grad_fn]

            mem_before = torch.cuda.memory_allocated()
            del grad_fn
            gc.collect()
            mem_after = torch.cuda.memory_allocated()
            delta = mem_after - mem_before
            self._iteration_entries.append(IterationEntry(
                operation_context=context,
                size_bytes=-delta,
            ))

    def populate_report(self, report_builder):
        for entry in self._iteration_entries:
            report_builder.add_iteration_entry(
                name=entry.operation_context.operation_name,
                size_bytes=entry.size_bytes,
                stack_context=entry.operation_context.stack,
            )

    def _callable_hook_creator(self, func):
        def hook(*args, **kwargs):
            self._meter.checkpoint()
            retval = func(*args, **kwargs)
            delta = self._meter.checkpoint()
            self._sum += delta

            # Early return for tensor-producing operations that are not
            # involved in the backward pass
            if (not isinstance(retval, torch.Tensor) and
                    not isinstance(retval, tuple) and
                    not isinstance(retval, list)):
                return retval
            if (isinstance(retval, torch.Tensor) and
                    (not retval.is_cuda or retval.grad_fn is None)):
                return retval

            context = OperationContext(
                operation_name=func.__name__,
                stack=CallStack.from_here(start_from=2),
            )
            self._handle_callable_result(retval, context)
            return retval

        return hook

    def _handle_callable_result(self, retval, context):
        if isinstance(retval, torch.Tensor) and retval.grad_fn is not None:
            self._grad_function_contexts[retval.grad_fn] = context

        elif isinstance(retval, tuple) or isinstance(retval, list):
            for inner_value in retval:
                self._handle_callable_result(inner_value, context)

    def _is_callable(self, maybe_fn):
        is_callable = (
            inspect.isfunction(maybe_fn) or
            inspect.ismethod(maybe_fn) or
            inspect.isbuiltin(maybe_fn) or
            inspect.isroutine(maybe_fn)
        )
        if not is_callable:
            return False

        # By convention, _ prefixed functions in Python should not be
        # called by users (i.e. they are "private" functions)
        return maybe_fn.__name__[0] != '_'


def get_gradient_functions(output_tensor):
    if output_tensor.grad_fn is None:
        return []

    result = []
    stack = [(output_tensor.grad_fn, 0)]
    visited = {output_tensor.grad_fn}

    while len(stack) > 0:
        grad_fn, visit_count = stack.pop()

        if visit_count != 0:
            result.append(grad_fn)
            continue

        stack.append((grad_fn, 1))

        for fn, _ in grad_fn.next_functions:
            if fn is None or fn in visited:
                continue
            visited.add(fn)
            stack.append((fn, 0))

    result.reverse()
    return result
