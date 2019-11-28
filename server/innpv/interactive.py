import threading

should_shutdown = threading.Event()


def signal_handler(signal, frame):
    global should_shutdown
    should_shutdown.set()


def set_up_logging(log_location):
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        filename=log_location,
        filemode="w",
    )


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
        default="/tmp/innpv-server.log",
        help="The location of the log file.",
    )
    parser.set_defaults(func=main)


def main(args):
    import signal
    from innpv.config import Config
    from innpv.server import INNPVServer

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    set_up_logging(args.log_file)
    Config.parse_args(args)

    # Run the server until asked to terminate
    with INNPVServer(args.host, args.port):
        should_shutdown.wait()
