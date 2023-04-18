import pynvml


class NVML:
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        pynvml.nvmlInit()

    def stop(self):
        pynvml.nvmlShutdown()

    def get_memory_capacity(self):
        # TODO: Support multiple devices
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle)
