import logging
import os
import sys

logger = logging.getLogger(__name__)


def check_skyline_preconditions(args):
    """
    This is the first function that should run before importing any other
    Skyline code.
    """
    _configure_logging(args)
    if not _validate_dependencies():
        sys.exit(1)
    if not _validate_gpu():
        sys.exit(1)


def initialize_skyline(args):
    """
    Performs common initialization tasks.
    """
    from skyline.config import Config

    project_root = os.getcwd()
    entry_point = args.entry_point
    if not _validate_paths(project_root, entry_point):
        sys.exit(1)

    Config.parse_args(args)
    Config.set_project_paths(project_root, entry_point)


def _configure_logging(args):
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.DEBUG if args.debug else logging.INFO,
    }

    if args.log_file is not None:
        kwargs["filename"] = args.log_file

    logging.basicConfig(**kwargs)


def _validate_dependencies():
    # NOTE: If you make a change here, make sure to update the INSTALL_REQUIRES
    #       list in setup.py as well.
    try:
        import yaml # pyyaml on PyPI
        import pynvml # nvidia-ml-py3 on PyPI
        import google.protobuf # protobuf on PyPI
        import numpy
        import torch
        return True
    except ImportError as ex:
        logger.error(
            "Skyline could not find the '%s' module, which is a required "
            "dependency. Please make sure all the required dependencies are "
            "installed before launching Skyline. If you use a package "
            "manager, these dependencies will be automatically installed for "
            "you.",
            ex.name,
        )
        return False


def _validate_gpu():
    import torch
    if not torch.cuda.is_available():
        logger.error(
            "Skyline did not detect a GPU on this machine. Skyline only "
            "profiles deep learning workloads on GPUs."
        )
        return False
    return True


def _validate_paths(project_root, entry_point):
    if not os.path.isabs(project_root):
        logger.error(
            "The project root that Skyline received is not an absolute path. "
            "This is an unexpected error. Please report a bug."
        )
        logger.error("Current project root: %s", project_root)
        return False

    if os.path.isabs(entry_point):
        logger.error(
            "The entry point must be specified as a relative path to the "
            "current directory. Please double check that the entry point you "
            "are providing is a relative path.",
        )
        logger.error("Current entry point path: %s", entry_point)
        return False

    full_path = os.path.join(project_root, entry_point)
    if not os.path.isfile(full_path):
        logger.error(
            "Either the specified entry point is not a file or its path was "
            "specified incorrectly. Please double check that it exists and "
            "that its path is correct.",
        )
        logger.error("Current absolute path to entry point: %s", full_path)
        return False

    return True
