import re

import torch

DUNDER_REGEX = re.compile('__(?P<name>.+)__')


def tensor_size_bytes(tensor):
    if tensor is None or not tensor.is_cuda:
        return 0
    return tensor.numel() * tensor.element_size()


def remove_dunder(fn_name):
    match = DUNDER_REGEX.match(fn_name)
    if match is None:
        return fn_name
    return match.group('name')
