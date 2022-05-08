import os

import toml

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
print(dir_path)
print(os.path.join("..", dir_path, "pyproject.toml"))
package = toml.load(os.path.join(dir_path, "..", "pyproject.toml"))

__name__ = package["tool"]["poetry"]["name"]
__version__ = package["tool"]["poetry"]["version"]
__description__ = package["tool"]["poetry"]["description"]
