import os
from importlib.metadata import version, PackageNotFoundError

try:
    package_name = "deepview_profile"
    __name__ = package_name
    __version__ = version(package_name)
    __description__ = "Interactive performance profiling and debugging tool for PyTorch neural networks."

except PackageNotFoundError:
    __version__ = "unknown"
    __description__ = "unknown"

from .__main__ import main
