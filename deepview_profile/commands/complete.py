import logging 
import os
import sys
import json

from deepview_profile.analysis.runner import analyze_project
from deepview_profile.nvml import NVML

from deepview_profile.initialization import (
    check_skyline_preconditions,
    initialize_skyline,
)
from deepview_profile.error_printing import print_analysis_error

logger = logging.getLogger(__name__)


def register_command(subparsers):
    parser = subparsers.add_parser(
        "complete",
        help="Generate a complete time and memory usage report.",
    )
    parser.add_argument(
        "entry_point",
        help="The entry point file in this project that contains the DeepView "
             "provider functions."
    )
    parser.add_argument(
        "-o", "--output",
        help="The location where the complete report should be stored",
        required=True
    )
    parser.add_argument(
        "--log-file",
        help="The location of the log file",
    )
    parser.add_argument("--debug", action="store_true", help="Log debug messages.")
    parser.set_defaults(func=main)

def actual_main(args): 
    from deepview_profile.analysis.session import AnalysisSession
    from deepview_profile.exceptions import AnalysisError

    if os.path.exists(args.output):
        print(
            "ERROR: The specified output file already exists.",
            file=sys.stderr,
        )
        sys.exit(1)

    try: 
        project_root = os.getcwd()
        data = { 
            "analysisState": {
                "message_type": "analysis",
                "project_root": project_root,
                "project_entry_point": args.entry_point,
                "hardware_info": {},
                "throughput": {},
                "breakdown": {},
                "habitat": {},
                "additionalProviders": "",
                "energy": {},
                "utilization": {}
            }
        }

        with NVML() as nvml:
            analyzer = analyze_project(project_root, args.entry_point, nvml)

            data.analysisState.breakdown = next(analyzer)
            data.analysisState.throughput = next(analyzer)
            data.analysisState.habitat = next(analyzer)
            data.analysisState.utilization = next(analyzer)
            data.analysisState.energy = next(analyzer)

        with open(args.output, "w") as json_file:
            json.dump(data, json_file, indent=4)

    except AnalysisError as ex: 
        print_analysis_error(ex)
        sys.exit(1)


def main(args): 
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)