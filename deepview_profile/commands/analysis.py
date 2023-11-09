import logging 
import os
import sys
import json

from deepview_profile.analysis.runner import analyze_project
from deepview_profile.nvml import NVML
from deepview_profile.utils import release_memory

from google.protobuf.json_format import MessageToDict

from deepview_profile.initialization import (
    check_skyline_preconditions,
    initialize_skyline,
)
from deepview_profile.error_printing import print_analysis_error

logger = logging.getLogger(__name__)


def register_command(subparsers):
    parser = subparsers.add_parser(
        "analysis",
        help="Generate usage report for various analysis.",
    )
    parser.add_argument(
        "entry_point",
        help="The entry point file in this project that contains the DeepView "
             "provider functions."
    )
    parser.add_argument(
        "--all",
        help="The complete analysis of all methods"
    )
    parser.add_argument(
        "-breakdown", "--measure-breakdown",
        help="Adds breakdown data to results"
    )
    parser.add_argument(
        "-throughput", "--measure-throughput",
        help="Adds throughput data to results"
    )
    parser.add_argument(
        "-predict", "--habitat-predict",
        help="Adds habitat data prediction to results"
    )
    parser.add_argument(
        "-utilization", "--measure-utilization",
        help="Adds utilization data to results"
    )
    parser.add_argument(
        "-energy", "--energy-compute",
        help="Adds energy use to results"
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



def measure_breakdown(session, nvml):
    print("analysis: running measure_breakdown()")
    yield session.measure_breakdown(nvml)
    release_memory()

def measure_throughput(session):
    print("analysis: running measure_throughput()")
    yield session.measure_throughput()
    release_memory()

def habitat_predict(session):
    print("analysis: running deepview_predict()")
    yield session.habitat_predict()
    release_memory()

def measure_utilization(session):
    print("analysis: running measure_utilization()")
    yield session.measure_utilization()
    release_memory()

def energy_compute(session):
    print("analysis: running energy_compute()")
    yield session.energy_compute()
    release_memory()

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

        session = AnalysisSession.new_from(project_root, args.entry_point) 
        release_memory()
        message = None

        if args.measure_breakdown is not None or args.all is not None: 
            with NVML() as nvml:
                data['analysisState']['breakdown'] = MessageToDict(next(measure_breakdown(session, nvml)))

        if args.measure_throughput is not None or args.all is not None:
            data['analysisState']['throughput'] = MessageToDict(next(measure_throughput(session)))

        if args.habitat_predict is not None or args.all is not None:
            data['analysisState']['habitat'] = MessageToDict(next(habitat_predict(session)))

        if args.measure_utilization is not None or args.all is not None:
            data['analysisState']['utilization'] = MessageToDict(next(measure_utilization(session)))

        if args.energy_compute is not None or args.all is not None:
            data['analysisState']['energy'] = MessageToDict(next(energy_compute(session)))

        with open(args.output, "w") as json_file:
            json.dump(data, json_file, indent=4)

    except AnalysisError as ex: 
        print_analysis_error(ex)
        sys.exit(1)


def main(args): 
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)