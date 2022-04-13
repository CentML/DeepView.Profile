import collections
import gc

import torch

from skyline.tracking.backward_interceptor import BackwardInterceptor
from skyline.tracking.base import TrackerBase
from skyline.tracking.call_stack import CallStack
from skyline.tracking.callable_tracker import CallableTracker
from skyline.tracking.utils import remove_dunder
from skyline.user_code_utils import user_code_environment

OperationContext = collections.namedtuple(
    'OperationContext',
    ['operation_name', 'stack'],
)

ActivationEntry = collections.namedtuple(
    'ActivationEntry',
    ['operation_name', 'stack', 'size_bytes'],
)


class ActivationsTracker:
    def __init__(self, project_root):
        self._activations = []
        self._project_root = project_root

    def track_memory_usage(self, iteration, input_provider, user_code_path):
        # 1. Run the forward pass of the model with the given inputs. We keep
        #    track of all the operations that contribute to the autograd graph.
        model_output, grad_function_contexts = \
            self._get_grad_function_contexts(
                iteration, input_provider, user_code_path)

        # 2. Traverse the autograd graph and get a topological ordering. Filter
        #    the function contexts by the gradient functions in our topological
        #    ordering.
        gradient_functions_topo_order, grad_function_contexts = \
            self._extract_relevant_gradient_functions(
                model_output, grad_function_contexts)

        # 3. Associate activation sizes with each gradient function by
        #    effectively "freeing" them one after the other and tracking the
        #    change in overall memory allocations.

        # NOTE: We reverse the list here to be able to pop from it in
        #       topological order.
        gradient_functions_topo_order.reverse()
        del model_output
        gc.collect()

        while len(gradient_functions_topo_order) > 0:
            grad_fn = gradient_functions_topo_order.pop()
            context = grad_function_contexts[grad_fn]
            del grad_function_contexts[grad_fn]

            mem_before = torch.cuda.memory_allocated()
            del grad_fn
            gc.collect()
            mem_after = torch.cuda.memory_allocated()
            delta = mem_after - mem_before
            self._activations.append(ActivationEntry(
                *context,
                size_bytes=-delta,
            ))

    def populate_report(self, builder):
        for entry in self._activations:
            builder.add_activation_entry(
                operation_name=remove_dunder(entry.operation_name),
                size_bytes=entry.size_bytes,
                stack_context=entry.stack,
            )

    def populate_breakdown(self, builder):
        # The HierarchicalReportBuilder uses the same activation entry API as
        # the MemoryReportBuilder
        self.populate_report(builder)

    def _get_grad_function_contexts(
            self, iteration, input_provider, user_code_path):
        grad_function_tracker = GradFunctionTracker(self._project_root)
        backward_interceptor = BackwardInterceptor()
        with grad_function_tracker.track(), \
                backward_interceptor.intercept(), \
                user_code_environment(user_code_path, self._project_root):
            iteration(*input_provider())
        return (
            backward_interceptor.backward_root,
            grad_function_tracker.grad_function_contexts,
        )

    def _extract_relevant_gradient_functions(
            self, model_output, grad_function_contexts):
        # 1. Get the gradient functions associated with the model output in
        #    topological order
        gradient_functions = \
            _extract_gradient_functions_in_topological_order(model_output)

        # 2. Filter the gradient functions: we only want to keep the ones we
        #    know about
        relevant_grad_fns = []
        relevant_contexts = {}
        for grad_fn in gradient_functions:
            if grad_fn not in grad_function_contexts:
                continue
            relevant_grad_fns.append(grad_fn)
            relevant_contexts[grad_fn] = grad_function_contexts[grad_fn]

        return relevant_grad_fns, relevant_contexts


class GradFunctionTracker(TrackerBase):
    def __init__(self, project_root):
        super().__init__()
        self._callable_tracker = CallableTracker(self._callable_hook_creator)
        self._project_root = project_root
        self.grad_function_contexts = {}
        self._processing_hook = False

    def start_tracking(self):
        super().start_tracking()
        self.grad_function_contexts.clear()
        self._callable_tracker.start_tracking()

    def stop_tracking(self):
        super().stop_tracking()
        self._callable_tracker.stop_tracking()

    def _callable_hook_creator(self, func):
        def hook(*args, **kwargs):
            # NOTE: We use self._processing_hook to handle cases where we have
            #       hooks on nested function calls.
            if self._processing_hook:
                return func(*args, **kwargs)

            self._processing_hook = True
            try:
                retval = func(*args, **kwargs)
            finally:
                self._processing_hook = False

            # Early return for tensor-producing operations that are not
            # involved in the backward pass
            if (not isinstance(retval, torch.Tensor) and
                    not isinstance(retval, tuple) and
                    not isinstance(retval, list)):
                return retval
            if (isinstance(retval, torch.Tensor) and
                    (not retval.is_cuda or retval.grad_fn is None)):
                return retval

            stack = CallStack.from_here(self._project_root, start_from=2)
            if len(stack.frames) == 0:
                return retval

            context = OperationContext(
                operation_name=func.__name__,
                stack=stack,
            )
            self._handle_callable_result(retval, context)
            return retval

        return hook

    def _handle_callable_result(self, retval, context):
        if isinstance(retval, torch.Tensor) and retval.grad_fn is not None:
            self.grad_function_contexts[retval.grad_fn] = context

        elif isinstance(retval, tuple) or isinstance(retval, list):
            for inner_value in retval:
                self._handle_callable_result(inner_value, context)


def _extract_gradient_functions_in_topological_order(model_output):
    """
    Given a model output (Tensor or nested list/tuple of Tensors), build a
    topological ordering of their gradient functions.
    """
    if isinstance(model_output, tuple) or isinstance(model_output, list):
        tensors = _flatten_and_filter_tensors(tensor_iterable)
    elif (isinstance(model_output, torch.Tensor) and
          model_output.grad_fn is not None):
        tensors = [model_output]
    else:
        return []

    result = []
    visited = {tensor.grad_fn for tensor in tensors}
    stack = [(grad_fn, 0) for grad_fn in visited]

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


def _flatten_and_filter_tensors(tensor_iterable):
    flattened = []
    for iterable_element in tensor_iterable:
        if (isinstance(iterable_element, torch.Tensor) and
                iterable_element.grad_fn is not None):
            flattened.append(iterable_element)
        elif (isinstance(iterable_element, tuple) or
              isinstance(iterable_element, list)):
            flattened.extend(_flatten_and_filter_tensors(iterable_element))
    return flattened
