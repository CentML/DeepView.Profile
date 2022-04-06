import logging
import signal
import subprocess
import threading

from skyline.initialization import (
    check_skyline_preconditions,
    initialize_skyline,
)

logger = logging.getLogger(__name__)


def register_command(subparsers):
    parser = subparsers.add_parser(
        "interactive",
        help="Start a new Skyline interactive profiling session.",
    )
    parser.add_argument(
        "entry_point",
        help="The entry point file in this project that contains the Skyline "
             "provider functions.",
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
    parser.add_argument(
        "--skip-atom",
        action="store_true",
        help="Skip launching Atom.",
    )
    parser.set_defaults(func=main)


def launch_atom():
    try:
        # The atom command line executable returns by default after launching
        # Atom (i.e. it does not block and wait until Atom is closed).
        subprocess.run(["atom", "atom://skyline"], check=True)
    except FileNotFoundError:
        logger.warn(
            "Skyline was not able to launch Atom from the command line. "
            "Please make sure that Atom is installed and then launch Atom "
            "manually to use Skyline's interactive profiling interface.",
        )


def actual_main(args):
    from skyline.config import Config
    from skyline.server import SkylineServer

    should_shutdown = threading.Event()

    def signal_handler(signal, frame):
        should_shutdown.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not args.skip_atom:
        launch_atom()

    with SkylineServer(args.host, args.port) as server:
        _, port = server.listening_on
        logger.info(
            "Skyline interactive profiling session started! "
            "Listening on port %d.",
            port,
        )
        logger.info("Project Root:  %s", Config.project_root)
        logger.info("Entry Point:   %s", Config.entry_point)

        # Run the server until asked to terminate
        should_shutdown.wait()


def main(args):
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)
