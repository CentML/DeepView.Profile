import os


class Sentinel:
    def __init__(self):
        self._read_pipe = None
        self._write_pipe = None

    def start(self):
        self._read_pipe, self._write_pipe = os.pipe()

    def stop(self):
        os.close(self._write_pipe)
        os.close(self._read_pipe)
        self._read_pipe = None
        self._write_pipe = None

    @property
    def read_pipe(self):
        return self._read_pipe

    def should_exit(self, ready_descriptors):
        return self._read_pipe in ready_descriptors

    def signal_exit(self):
        os.write(self._write_pipe, b'\0')
