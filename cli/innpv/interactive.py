import logging
import signal
import sys
import threading


def register_command(subparsers):
    parser = subparsers.add_parser(
        "interactive",
        help="Start a new INNPV interactive profiling session.",
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
        default="hints.yml",
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


def main(args):
    from innpv.config import Config
    from innpv.server import INNPVServer

    should_shutdown = threading.Event()

    def signal_handler(signal, frame):
        should_shutdown.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    configure_logging(args)
    Config.parse_args(args)

    with INNPVServer(args.host, args.port) as server:
        _, port = server.listening_on
        logging.info(
            "INNPV interactive profiling session started! "
            "Listening on port %d.",
            port,
        )
        # Run the server until asked to terminate
        should_shutdown.wait()
