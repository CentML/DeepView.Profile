import torch


class BackwardHelper:
    def __init__(self, backward_runnable, ag_dict):
        self.run_backward = backward_runnable
        self._ag_dict = ag_dict

    @classmethod
    def new_from(cls, operation_outputs):
        retval, initial_grad_fn = get_grad_fn(operation_outputs)
        if initial_grad_fn is None:
            raise ValueError('No grad_fn available on the operation output.')

        grads = torch.ones_like(retval)
        def backward_runnable():
            torch.autograd.backward(retval, grads, retain_graph=True)

        size_dict = get_accumulate_grad_inputs(
            initial_grad_fn,
            backward_runnable,
        )

        ag_dict = {
            grad_fn: torch.randn(size, device=torch.device('cuda'))
            for grad_fn, size in size_dict.items()
        }

        return cls(backward_runnable, ag_dict)

    def run_accumulate_grad(self):
        for grad_fn, grad in self._ag_dict.items():
            grad_fn(grad)


def backward_available(operation_output):
    return get_grad_fn(operation_output)[1] is not None


def flatten_operation_output(operation_output):
    if isinstance(operation_output, torch.Tensor):
        return [operation_output]
    elif (not isinstance(operation_output, tuple) and
          not isinstance(operation_output, list)):
        return []

    flattened = []
    for value in operation_output:
        flattened.extend(flatten_operation_output(value))
    return flattened


def get_grad_fn(retval):
    if isinstance(retval, torch.Tensor) and retval.grad_fn is not None:
        return retval, retval.grad_fn
    elif isinstance(retval, tuple) or isinstance(retval, list):
        for inner_value in retval:
            inner_retval, grad_fn = get_grad_fn(inner_value)
            if grad_fn is not None:
                return inner_retval, grad_fn

    return None, None


def get_accumulate_grad_inputs(initial_grad_fn, backward_runnable):
    input_dict = {}
    hook_handles = []
    def get_hook(grad_fn):
        def hook(arg1, arg2):
            if not isinstance(arg2[0], torch.Tensor):
                return
            input_dict[grad_fn] = arg2[0].size()
        return hook

    # Traverse the graph to identify all AccumulateGrad functions
    stack = [initial_grad_fn]
    visited = {initial_grad_fn}

    while len(stack) > 0:
        grad_fn = stack.pop()

        if grad_fn.name() == 'torch::autograd::AccumulateGrad':
            hook_handles.append(grad_fn.register_hook(get_hook(grad_fn)))

        for next_grad_fn, _ in grad_fn.next_functions:
            if next_grad_fn is None or next_grad_fn in visited:
                continue
            stack.append(next_grad_fn)
            visited.add(next_grad_fn)

    # Run a backward pass to get accumulate grad sizes
    backward_runnable()
    torch.cuda.synchronize()

    # Clear hooks
    for handle in hook_handles:
        handle.remove()

    return input_dict
