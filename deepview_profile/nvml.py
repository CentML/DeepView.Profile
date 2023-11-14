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

    def get_device_names(self):
        device_names = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            device_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            device_names.append(device_name)
        return device_names
    
