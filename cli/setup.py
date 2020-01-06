import codecs
import os
import re

from setuptools import setup, find_packages

# Acknowledgement: This setup.py was adapted from Hynek Schlawack's Python
#                  Packaging Guide
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty

###################################################################

NAME = "skyline"
PACKAGES = find_packages()
META_PATH = os.path.join("skyline", "__init__.py")
PYTHON_REQUIRES = ">=3.6"

PACKAGE_DATA = {
    'skyline': ['data/hints.yml'],
}

ENTRY_POINTS = {
    "console_scripts": [
        "skyline = skyline.__main__:main",
    ],
}

INSTALL_REQUIRES = [
    "pyyaml",
    "nvidia-ml-py3",
    "protobuf",
    "numpy",
    "torch>=1.2.0",
]

KEYWORDS = [
    "neural networks",
    "pytorch",
    "interactive",
    "performance",
    "visualization",
    "profiler",
    "debugger",
]

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Software Development :: Debuggers",
]

###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and return the contents of the
    resulting file. Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        version=find_meta("version"),
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
    )
