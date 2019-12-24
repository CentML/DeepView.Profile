import logging
import os
import signal
import sys
import threading

logger = logging.getLogger(__name__)


def register_command(subparsers):
    parser = subparsers.add_parser(
        "interactive",
        help="Start a new INNPV interactive profiling session.",
    )
    parser.add_argument(
        "entry_point",
        help="The entry point file in this project that contains the INNPV "
             "provider functions.",
    )
    parser.add_argument(
        "--host",
        default="",
        help="The host address to bind to.",
    )
    parser.add_argument(
        "--port",
        default=0,
        type=int,
        help="The port to listen on.",
    )
    parser.add_argument(
        "--hints-file",
        help="Path to the performance hints configuration YAML file.",
    )
    parser.add_argument(
        "--measure-for",
        help="Number of iterations to measure when determining throughput.",
    )
    parser.add_argument(
        "--warm-up",
        help="Number of warm up iterations when determining throughput.",
    )
    parser.add_argument(
        "--log-file",
        help="The location of the log file.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Log debug messages.")
    parser.set_defaults(func=main)


def configure_logging(args):
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.DEBUG if args.debug else logging.INFO,
    }

    if args.log_file is not None:
        kwargs["filename"] = args.log_file

    logging.basicConfig(**kwargs)


def validate_paths(project_root, entry_point):
    if not os.path.isabs(project_root):
        logger.error(
            "The project root that INNPV received is not an absolute path. "
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


def validate_dependencies():
    # NOTE: If you make a change here, make sure to update the INSTALL_REQUIRES
    #       list in setup.py as well.
    try:
        import yaml # pyyaml on pip
        import pynvml # nvidia-ml-py3 on pip
        import google.protobuf # protobuf on pip
        import numpy
        import torch
        return True
    except ImportError as ex:
        logger.error(
            "INNPV could not find the '%s' module, which is a required "
            "dependency. Please make sure all the required dependencies are "
            "installed before launching INNPV. If you use a package manager, "
            "these dependencies will be automatically installed for you.",
            ex.name,
        )
        return False


def pre_main(args):
    configure_logging(args)
    if not validate_dependencies():
        sys.exit(1)


def actual_main(args):
    from innpv.config import Config
    from innpv.server import INNPVServer

    project_root = os.getcwd()
    entry_point = args.entry_point
    if not validate_paths(project_root, entry_point):
        sys.exit(1)

    Config.parse_args(args)
    Config.set_project_paths(project_root, entry_point)

    should_shutdown = threading.Event()

    def signal_handler(signal, frame):
        should_shutdown.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with INNPVServer(args.host, args.port) as server:
        _, port = server.listening_on
        logger.info(
            "INNPV interactive profiling session started! "
            "Listening on port %d.",
            port,
        )
        logger.info("Project Root (server):  %s", project_root)
        logger.info("Entry Point:            %s", entry_point)

        # Run the server until asked to terminate
        should_shutdown.wait()


def main(args):
    # We need a separate pre_main() to validate the existence of our
    # dependencies before any further INNPV imports in actual_main().
    pre_main(args)
    actual_main(args)
