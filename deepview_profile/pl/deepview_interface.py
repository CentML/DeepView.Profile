import sys
from typing import Callable
import platform

from deepview_profile.analysis.session import AnalysisSession
from deepview_profile.exceptions import AnalysisError
from deepview_profile.nvml import NVML

# from deepview_profile.utils import release_memory, next_message_to_dict, files_encoded_unique
from deepview_profile.utils import release_memory, files_encoded_unique
from deepview_profile.error_printing import print_analysis_error

from google.protobuf.json_format import MessageToDict


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
        "hostname": platform.node(),
        "os": " ".join(list(platform.uname())),
        "gpus": nvml.get_device_names(),
    }
    return hardware_info


class DummyStaticAnalyzer:
    def batch_size_location(self):
        return None


def next_message_to_dict(a):
    message = next(a)
    return MessageToDict(message, preserving_proto_field_name=True)


def trigger_profiling(
    project_root: str,
    entry_point: str,
    initial_batch_size: int,
    input_provider: Callable,
    model_provider: Callable,
    iteration_provider: Callable,
):
    try:
        data = {
            "analysis": {
                "message_type": "analysis",
                "project_root": project_root,
                "project_entry_point": entry_point,
                "hardware_info": {},
                "throughput": {},
                "breakdown": {},
                "habitat": {},
                "additionalProviders": "",
                "energy": {},
                "utilization": {},
                "ddp": {},
            },
            "epochs": 50,
            "iterations": 1000,
            "encodedFiles": [],
        }

        session = AnalysisSession(
            project_root,
            entry_point,
            project_root,
            model_provider,
            input_provider,
            iteration_provider,
            initial_batch_size,
            DummyStaticAnalyzer(),
        )
        release_memory()

        exclude_source = False

        with NVML() as nvml:
            data["analysis"]["hardware_info"] = hardware_information(nvml)
            data["analysis"]["breakdown"] = next_message_to_dict(
                measure_breakdown(session, nvml)
            )

            operation_tree = data["analysis"]["breakdown"]["operation_tree"]
            if not exclude_source and operation_tree is not None:
                data["encodedFiles"] = files_encoded_unique(operation_tree)

        data["analysis"]["throughput"] = next_message_to_dict(
            measure_throughput(session)
        )
        data["analysis"]["habitat"] = next_message_to_dict(habitat_predict(session))
        data["analysis"]["utilization"] = next_message_to_dict(
            measure_utilization(session)
        )
        data["analysis"]["energy"] = next_message_to_dict(energy_compute(session))
        # data['analysis']['ddp'] = next_message_to_dict(ddp_analysis(session))

        from deepview_profile.export_converter import convert

        data["analysis"] = convert(data["analysis"])

        return data

    except AnalysisError as ex:
        print_analysis_error(ex)
        sys.exit(1)
