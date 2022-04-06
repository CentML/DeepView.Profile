import logging
import os
import sys

from skyline.initialization import (
    check_skyline_preconditions,
    initialize_skyline,
)
from skyline.error_printing import print_analysis_error

logger = logging.getLogger(__name__)


def register_command(subparsers):
    parser = subparsers.add_parser(
        "time",
        help="Generate an iteration run time breakdown report.",
    )
    parser.add_argument(
        "entry_point",
        help="The entry point file in this project that contains the Skyline "
             "provider functions.",
    )
    parser.add_argument(
        "-o", "--output",
        help="The location where the iteration run time breakdown report "
             "should be stored.",
        required=True,
    )
    parser.add_argument(
        "--log-file",
        help="The location of the log file.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Log debug messages.")
    parser.set_defaults(func=main)


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
        session = AnalysisSession.new_from(
            Config.project_root, Config.entry_point)
        session.generate_run_time_breakdown_report(
            save_report_to=args.output,
        )
    except AnalysisError as ex:
        print_analysis_error(ex)
        sys.exit(1)


def main(args):
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)
