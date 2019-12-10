import argparse
import os

from innpv.tracking.memory import track_memory_usage
from innpv.exceptions import AnalysisError

MODEL_PROVIDER_NAME = "innpv_model_provider"
INPUT_PROVIDER_NAME = "innpv_input_provider"


def analyze_project(project_root, entry_point):
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

    model_provider = scope[MODEL_PROVIDER_NAME]
    input_provider = scope[INPUT_PROVIDER_NAME]

    # 3. Run the analysis - right now this is just a memory analysis
    report = track_memory_usage(model_provider, input_provider)


def _run_entry_point(project_root, entry_point):
    file_name = os.path.join(project_root, entry_point)
    with open(file_name) as file:
        code_str = file.read()
    code = compile(code_str, file_name, mode="exec")
    scope = {}
    exec(code, scope, scope)
    return scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entry_point", type=str)
    args = parser.parse_args()

    # TODO:
    #  - Define protobufs for the memory usage report
    #  - Serialize and write out results or errors
    project_root = os.getcwd()
    analyze_project(project_root, args.entry_point)


if __name__ == "__main__":
    main()
