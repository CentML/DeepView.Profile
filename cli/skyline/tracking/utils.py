import torch


def tensor_size_bytes(tensor):
    if tensor is None or not tensor.is_cuda:
        return 0
    return tensor.numel() * tensor.element_size()
