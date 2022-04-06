import codecs
import os
import re
import sys

from setuptools import setup, find_packages

# Acknowledgement: This setup.py was adapted from Hynek Schlawack's Python
#                  Packaging Guide
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty

###################################################################

NAME = "skyline-cli"
PACKAGES = find_packages()
META_PATH = os.path.join("skyline", "__init__.py")
README_PATH = os.path.join("..", "README.md")
PYTHON_REQUIRES = ">=3.6"

PACKAGE_DATA = {
    "skyline": ["data/hints.yml"],
}

ENTRY_POINTS = {
    "console_scripts": [
        "skyline = skyline.__main__:main",
    ],
}

# If this option is specified, we'll install the skyline-evaluate entry point
EVALUATE_INSTALL_FLAG = "--install-skyline-evaluate"
EVALUATE_ENTRY_POINT = "skyline-evaluate = skyline.evaluate:main"

INSTALL_REQUIRES = [
    "pyyaml",
    "nvidia-ml-py3",
    "protobuf",
    "numpy",
    "torch>=1.1.0",
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
    "Development Status :: 4 - Beta",
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
    # By default, we don't install the skyline-evaluate tool because
    # it is only used for evaluation purposes. If the user sets the
    # --install-skyline-evaluate command line flag, we'll install it.
    if EVALUATE_INSTALL_FLAG in sys.argv:
        sys.argv.remove(EVALUATE_INSTALL_FLAG)
        ENTRY_POINTS["console_scripts"].append(EVALUATE_ENTRY_POINT)

    setup(
        name=NAME,
        description=find_meta("description"),
        license=find_meta("license"),
        version=find_meta("version"),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        long_description=read(README_PATH),
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
    )
