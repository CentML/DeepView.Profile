import logging
import os
import sys
import csv

from skyline.initialization import (
    check_skyline_preconditions,
    initialize_skyline,
)
from skyline.error_printing import print_analysis_error

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
        "-t", "--trials",
        help="Number of trials to run when making measurements.",
        type=int,
        required=True,
        default=5,
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


def make_measurements(session, batch_size):
    # This is a HACK
    session._batch_size = batch_size
    peak_usage_bytes = session.measure_peak_usage_bytes()
    thpt_msg = session.measure_throughput()
    return thpt_msg.samples_per_second, peak_usage_bytes


def actual_main(args):
    from skyline.analysis.session import AnalysisSession
    from skyline.config import Config
    from skyline.exceptions import AnalysisError

    if os.path.exists(args.output):
        print(
            "ERROR: The specified output file already exists.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(args.output, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'batch_size',
                'trial',
                'samples_per_second',
                'memory_usage_bytes',
            ])
            for batch_size in args.batch_sizes:
                for trial in range(args.trials):
                    session = AnalysisSession.new_from(
                        Config.project_root, Config.entry_point)
                    samples_per_second, memory_usage_bytes = make_measurements(
                        session, batch_size)
                    writer.writerow([
                        batch_size,
                        trial,
                        samples_per_second,
                        memory_usage_bytes,
                    ])

    except AnalysisError as ex:
        print_analysis_error(ex)
        sys.exit(1)


def main(args):
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)
