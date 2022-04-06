import os

_DATA_PATH = os.path.abspath(os.path.dirname(__file__))


def get_absolute_path(data_file):
    return os.path.join(_DATA_PATH, data_file)
