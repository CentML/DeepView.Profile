import logging 
import os
import sys
import json
import platform

from deepview_profile.analysis.runner import analyze_project
from deepview_profile.nvml import NVML
from deepview_profile.utils import release_memory, next_message_to_dict, files_encoded_unique

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
        action="store_true",
        help="The complete analysis of all methods"
    )
    parser.add_argument(
        "-breakdown", "--measure-breakdown",
        action="store_true",
        help="Adds breakdown data to results"
    )
    parser.add_argument(
        "-throughput", "--measure-throughput",
        action="store_true",
        help="Adds throughput data to results"
    )
    parser.add_argument(
        "-predict", "--habitat-predict",
        action="store_true",
        help="Adds habitat data prediction to results"
    )
    parser.add_argument(
        "-utilization", "--measure-utilization",
        action="store_true",
        help="Adds utilization data to results"
    )
    parser.add_argument(
        "-energy", "--energy-compute",
        action="store_true",
        help="Adds energy use to results"
    )
    parser.add_argument(
        "--include-ddp",
        action="store_true",
        help="Adds ddp analysis to results"
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
    parser.add_argument(
        "--exclude-source",
        action="store_true",
        help="Allows not adding encodedFiles section"
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

def ddp_analysis(session):
    print("analysis: running ddp_computation()")
    yield session.ddp_computation()
    release_memory()

def hardware_information(nvml):
    
    hardware_info = { 
        'hostname': platform.node(),
        'os': " ".join(list(platform.uname())),
        'gpus': nvml.get_device_names()
    }
    return hardware_info

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
                "utilization": {},
                "ddp": {}
            },
            "epochs": 50,
            "iterPerEpoch": 1000,
            "encodedFiles": []
        }

        session = AnalysisSession.new_from(project_root, args.entry_point)
        release_memory()

        is_return_all = args.all

        with NVML() as nvml:
            data['analysisState']['hardware_info'] = hardware_information(nvml)
            if args.measure_breakdown or is_return_all:
                data['analysisState']['breakdown'] = next_message_to_dict(measure_breakdown(session, nvml))

                operation_tree = data['analysisState']['breakdown']['operationTree']
                if not args.exclude_source and operation_tree is not None:
                    data['encodedFiles'] = files_encoded_unique(operation_tree)

        if args.measure_throughput or is_return_all:
            data['analysisState']['throughput'] = next_message_to_dict(measure_throughput(session))

        if args.habitat_predict or is_return_all:
            data['analysisState']['habitat'] = next_message_to_dict(habitat_predict(session))

        if args.measure_utilization or is_return_all:
            data['analysisState']['utilization'] = next_message_to_dict(measure_utilization(session))

        if args.energy_compute or is_return_all:
            data['analysisState']['energy'] = next_message_to_dict(energy_compute(session))

        if args.include_ddp:
            data['analysisState']['ddp'] = next_message_to_dict(ddp_analysis(session))

        with open(args.output, "w") as json_file:
            json.dump(data, json_file, indent=4)

    except AnalysisError as ex: 
        print_analysis_error(ex)
        sys.exit(1)

def main(args): 
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)