import argparse
import os

import innpv.protocol_gen.innpv_pb2 as pm
from innpv.exceptions import AnalysisError
from innpv.nvml import NVML
from innpv.tracking.memory import track_memory_usage
from innpv.tracking.report import MiscSizeType

MODEL_PROVIDER_NAME = "innpv_model_provider"
INPUT_PROVIDER_NAME = "innpv_input_provider"


# The code in this file should be "standalone" (i.e. it should not rely on any
# state kept in the innpv server). This is because eventually we might need to
# run it in a separate process for performance purposes.


def analyze_project(project_root, entry_point, nvml):
    model_provider, input_provider = _get_providers(project_root, entry_point)

    report = track_memory_usage(model_provider, input_provider)

    memory_usage = pm.MemoryUsageResponse()
    memory_usage.peak_usage_bytes = report.get_misc_entry(
        MiscSizeType.PeakUsageBytes)
    memory_usage.memory_capacity_bytes = nvml.get_memory_capacity().total

    for weight_entry in report.get_weight_entries(path_prefix=project_root):
        entry = memory_usage.weight_entries.add()
        entry.weight_name = weight_entry.weight_name
        entry.size_bytes = weight_entry.size_bytes
        entry.grad_size_bytes = weight_entry.grad_size_bytes

    for activation_entry in report.get_activation_entries(
            path_prefix=project_root):
        entry = memory_usage.activation_entries.add()
        entry.operation_name = activation_entry.operation_name
        entry.size_bytes = activation_entry.size_bytes

    return memory_usage


def _get_providers(project_root, entry_point):
    # 1. Run the entry point file to "load" the model
    try:
        scope = _run_entry_point(project_root, entry_point)
    except SyntaxError as ex:
        raise AnalysisError(
            "Syntax error on line {} column {} in {}.".format(
                ex.lineno,
                ex.offset,
                os.path.relpath(ex.filename, start=project_root),
            ),
        ) from ex

    # 2. Check that the model provider and input provider functions exist
    if MODEL_PROVIDER_NAME not in scope:
        raise AnalysisError(
            "The project entry point file is missing a model provider "
            "function. Please add a model provider function named "
            "\"{}\".".format(MODEL_PROVIDER_NAME)
        )

    if INPUT_PROVIDER_NAME not in scope:
        raise AnalysisError(
            "The project entry point file is missing an input provider "
            "function. Please add an input provider function named "
            "\"{}\".".format(INPUT_PROVIDER_NAME)
        )

    return (scope[MODEL_PROVIDER_NAME], scope[INPUT_PROVIDER_NAME])


def _run_entry_point(project_root, entry_point):
    file_name = os.path.join(project_root, entry_point)
    with open(file_name) as file:
        code_str = file.read()
    code = compile(code_str, file_name, mode="exec")
    scope = {}
    exec(code, scope, scope)
    return scope


def main():
    # This is used for development and debugging purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("entry_point", type=str)
    args = parser.parse_args()

    project_root = os.getcwd()
    with NVML() as nvml:
        result = analyze_project(project_root, args.entry_point, nvml)
    print('Peak usage:   ', result.peak_usage_bytes, 'bytes')
    print('Max. capacity:', result.memory_capacity_bytes, 'bytes')
    print('No. of weight entries:', len(result.weight_entries))
    print('No. of activ. entries:', len(result.activation_entries))


if __name__ == "__main__":
    main()
