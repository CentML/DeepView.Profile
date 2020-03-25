import logging
import os
import sys
import csv

from skyline.initialization import (
    check_skyline_preconditions,
    initialize_skyline,
)

logger = logging.getLogger(__name__)


def register_command(subparsers):
    parser = subparsers.add_parser(
        "measure-batches",
        help="Make throughput and memory measurements for given batch sizes.",
    )
    parser.add_argument(
        "entry_point",
        help="The entry point file in this project that contains the Skyline "
             "provider functions.",
    )
    parser.add_argument(
        "-b", "--batch-sizes",
        help="The batch sizes to consider.",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-o", "--output",
        help="The location where the evaluation output should be stored.",
        required=True,
    )
    parser.add_argument(
        "--log-file",
        help="The location of the log file.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Log debug messages.")
    parser.set_defaults(func=main)


def make_measurements(session, batch_size, nvml):
    # This is a HACK
    session._batch_size = batch_size
    bd_msg = session.measure_breakdown(nvml)
    thpt_msg = session.measure_throughput()
    return thpt_msg.samples_per_second, bd_msg.peak_usage_bytes


def actual_main(args):
    from skyline.analysis.session import AnalysisSession
    from skyline.config import Config
    from skyline.exceptions import AnalysisError
    from skyline.nvml import NVML

    if os.path.exists(args.output):
        print(
            "ERROR: The specified output file already exists.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(args.output, 'w') as f, NVML() as nvml:
            writer = csv.writer(f)
            writer.writerow([
                'batch_size',
                'samples_per_second',
                'memory_usage_bytes',
            ])
            for batch_size in args.batch_sizes:
                session = AnalysisSession.new_from(
                    Config.project_root, Config.entry_point)
                samples_per_second, memory_usage_bytes = make_measurements(
                    session, batch_size, nvml)
                writer.writerow([
                    batch_size, samples_per_second, memory_usage_bytes,
                ])

    except AnalysisError as ex:
        print(
            "Skyline encountered an error when profiling your model:",
            file=sys.stderr,
        )
        print("->", str(ex), file=sys.stderr)
        sys.exit(1)


def main(args):
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)
