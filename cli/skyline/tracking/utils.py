import torch


def tensor_size_bytes(tensor):
    if tensor is None or not tensor.is_cuda:
        return 0
    return tensor.numel() * tensor.element_size()


def flatten_operation_retval(retval):
    if isinstance(retval, torch.Tensor):
        return [retval]
    elif (not isinstance(retval, tuple) and
          not isinstance(retval, list)):
        return []

    flattened = []
    for value in retval:
        flattened.extend(flatten_operation_retval(value))
    return flattened
