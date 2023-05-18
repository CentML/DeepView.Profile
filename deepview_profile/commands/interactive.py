import logging
import signal
import threading

from deepview_profile.initialization import (
    check_skyline_preconditions,
    initialize_skyline,
)

logger = logging.getLogger(__name__)


def register_command(subparsers):
    parser = subparsers.add_parser(
        "interactive",
        help="Start a new DeepView interactive profiling session.",
    )
    parser.add_argument(
        "--host",
        default="",
        help="The host address to bind to.",
    )
    parser.add_argument(
        "--port",
        default=60120,
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

def actual_main(args):
    from deepview_profile.server import SkylineServer

    should_shutdown = threading.Event()

    def signal_handler(signal, frame):
        should_shutdown.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    with SkylineServer(args.host, args.port) as server:
        _, port = server.listening_on
        logger.info(
            "DeepView interactive profiling session started! "
            "Listening on port %d.",
            port,
        )

        # Run the server until asked to terminate
        should_shutdown.wait()


def main(args):
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)
