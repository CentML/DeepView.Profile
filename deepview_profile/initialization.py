import logging
import sys

logger = logging.getLogger(__name__)


def check_skyline_preconditions(args):
    """
    This is the first function that should run before importing any other
    DeepView code.
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
    from deepview_profile.config import Config

    Config.parse_args(args)

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
        import yaml # pyyaml on PyPI # noqa: F401
        import pynvml # nvidia-ml-py3 on PyPI # noqa: F401
        import google.protobuf # protobuf on PyPI # noqa: F401
        import numpy # noqa: F401
        import torch # noqa: F401
        return True
    except ImportError as ex:
        logger.error(
            "DeepView could not find the '%s' module, which is a required "
            "dependency. Please make sure all the required dependencies are "
            "installed before launching DeepView. If you use a package "
            "manager, these dependencies will be automatically installed for "
            "you.",
            ex.name,
        )
        return False


def _validate_gpu():
    import torch
    if not torch.cuda.is_available():
        logger.error(
            "DeepView did not detect a GPU on this machine. DeepView only "
            "profiles deep learning workloads on GPUs."
        )
        return False
    return True
