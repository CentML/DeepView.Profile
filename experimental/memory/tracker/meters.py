import torch


class MemoryMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._reference_usage_bytes = torch.cuda.memory_allocated()

    def checkpoint(self):
        current_usage_bytes = torch.cuda.memory_allocated()
        delta_bytes = current_usage_bytes - self._reference_usage_bytes
        self._reference_usage_bytes = current_usage_bytes
        return delta_bytes
