import torch


class AutogradEngine:
    """
    Emulates the backward pass for a given model output, for timing purposes.
    """
    def __init__(self, grad_fn_ordering, input_map, initial_inputs):
        self._grad_fn_ordering = grad_fn_ordering
        self._input_holder = {
            fn: [None] * size for fn, size in input_map.items()
        }
        self._input_holder[self._grad_fn_ordering[0]] = initial_inputs

    @staticmethod
    def backward_available(operation_output):
        return _get_grad_fn(operation_output) is not None

    @classmethod
    def new_from(cls, operation_output, exclude_accumulate_grad=True):
        # Traverse the autograd graph, build input map for each grad_fn and
        # create a topological ordering
        initial_grad_fn = _get_grad_fn(operation_output)
        if initial_grad_fn is None:
            raise ValueError('No grad_fn available on the operation output.')

        ordering = []
        input_map = {}
        initial_inputs = [
            tensor.detach()
            for tensor in _flatten_operation_output(operation_output)
        ]
        input_map[initial_grad_fn] = len(initial_inputs)

        stack = [(initial_grad_fn, 0)]
        visited = {initial_grad_fn}

        # Build a topological ordering
        while len(stack) > 0:
            grad_fn, visit_count = stack.pop()
            if visit_count != 0:
                ordering.append(grad_fn)
                continue

            stack.append((grad_fn, 1))
            for next_fn, input_idx in grad_fn.next_functions:
                if next_fn is None:
                    continue

                if (exclude_accumulate_grad and
                        next_fn.name() == 'torch::autograd::AccumulateGrad'):
                    continue

                # Keep track of the inputs to each grad_fn
                if next_fn not in input_map:
                    input_map[next_fn] = 1
                input_map[next_fn] = max(input_map[next_fn], input_idx + 1)

                # Determine whether to visit this grad_fn
                if next_fn in visited:
                    continue

                visited.add(next_fn)
                stack.append((next_fn, 0))

        ordering.reverse()
        return cls(ordering, input_map, initial_inputs)

    def run_backward(self):
        for grad_fn in self._grad_fn_ordering:
            # 1. Run the backward function
            outputs = grad_fn(*(self._input_holder[grad_fn]))

            # 2. Store its outputs for the next backward function(s)
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            for (output, (next_fn, input_idx)) in zip(
                    outputs, grad_fn.next_functions):
                if next_fn is None or next_fn not in self._input_holder:
                    continue
                # NOTE: If implementing to actually calculate the gradient, we
                # need to sum gradients that "flow" into the same grad function
                # input.
                self._input_holder[next_fn][input_idx] = output


def _flatten_operation_output(operation_output):
    if isinstance(operation_output, torch.Tensor):
        return [operation_output]
    elif (not isinstance(operation_output, tuple) and
          not isinstance(operation_output, list)):
        return []

    flattened = []
    for value in operation_output:
        flattened.extend(_flatten_operation_output(value))
    return flattened


def _get_grad_fn(retval):
        if isinstance(retval, torch.Tensor) and retval.grad_fn is not None:
            return retval.grad_fn
        elif isinstance(retval, tuple) or isinstance(retval, list):
            for inner_value in retval:
                grad_fn = _get_grad_fn(inner_value)
                if grad_fn is not None:
                    return grad_fn
        else:
            return None
