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
        "prediction-models",
        help="Evaluate Skyline's prediction accuracy.",
    )
    parser.add_argument(
        "entry_point",
        help="The entry point file in this project that contains the Skyline "
             "provider functions.",
    )
    parser.add_argument(
        "-b", "--batch-sizes",
        help="The starting batch sizes to build models from.",
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


def get_model(session, batch_size):
    # This is a HACK
    session._batch_size = batch_size
    thpt_msg = session.measure_throughput()
    return (
        (thpt_msg.peak_usage_bytes.slope, thpt_msg.peak_usage_bytes.bias),
        (thpt_msg.run_time_ms.slope, thpt_msg.run_time_ms.bias),
    )


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
                'run_time_ms_slope',
                'run_time_ms_bias',
                'memory_usage_bytes_slope',
                'memory_usage_bytes_bias',
            ])
            for batch_size in args.batch_sizes:
                session = AnalysisSession.new_from(
                    Config.project_root, Config.entry_point)
                memory_model, run_time_model = get_model(
                    session, batch_size)
                writer.writerow([
                    batch_size, *run_time_model, *memory_model,
                ])

    except AnalysisError as ex:
        print_analysis_error(ex)
        sys.exit(1)


def main(args):
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)
