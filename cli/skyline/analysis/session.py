import inspect
import logging
import math
import os

import numpy as np

import skyline.protocol_gen.innpv_pb2 as pm
from skyline.exceptions import AnalysisError, exceptions_as_analysis_errors
from skyline.profiler.iteration import IterationProfiler
from skyline.tracking.memory import track_memory_usage
from skyline.tracking.report import MiscSizeType

logger = logging.getLogger(__name__)

MODEL_PROVIDER_NAME = "skyline_model_provider"
INPUT_PROVIDER_NAME = "skyline_input_provider"
ITERATION_PROVIDER_NAME = "skyline_iteration_provider"
BATCH_SIZE_ARG = "batch_size"


class AnalysisSession:
    def __init__(
        self,
        project_root,
        model_provider,
        input_provider,
        iteration_provider,
        batch_size
    ):
        self._project_root = project_root
        self._model_provider = model_provider
        self._input_provider = input_provider
        self._iteration_provider = iteration_provider
        self._batch_size = batch_size
        self._memory_usage_percentage = None

    @classmethod
    def new_from(cls, project_root, entry_point):
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

        if ITERATION_PROVIDER_NAME not in scope:
            raise AnalysisError(
                "The project entry point file is missing an iteration "
                "provider function. Please add an iteration provider function "
                "named \"{}\".".format(ITERATION_PROVIDER_NAME)
            )

        model_provider = scope[MODEL_PROVIDER_NAME]
        input_provider = scope[INPUT_PROVIDER_NAME]
        iteration_provider = scope[ITERATION_PROVIDER_NAME]

        batch_size = _validate_providers(
            model_provider, input_provider, iteration_provider)

        return cls(
            project_root,
            model_provider,
            input_provider,
            iteration_provider,
            batch_size,
        )

    def measure_memory_usage(self, nvml):
        report = track_memory_usage(
            self._model_provider,
            self._input_provider,
            self._iteration_provider,
        )

        memory_usage = pm.MemoryUsageResponse()
        memory_usage.peak_usage_bytes = report.get_misc_entry(
            MiscSizeType.PeakUsageBytes)
        memory_usage.memory_capacity_bytes = nvml.get_memory_capacity().total

        for weight_entry in report.get_weight_entries(
                path_prefix=self._project_root):
            entry = memory_usage.weight_entries.add()
            entry.weight_name = weight_entry.weight_name
            entry.size_bytes = weight_entry.size_bytes
            entry.grad_size_bytes = weight_entry.grad_size_bytes
            _set_file_context(entry, self._project_root, weight_entry)

        for activation_entry in report.get_activation_entries(
                path_prefix=self._project_root):
            entry = memory_usage.activation_entries.add()
            entry.operation_name = activation_entry.operation_name
            entry.size_bytes = activation_entry.size_bytes
            _set_file_context(entry, self._project_root, activation_entry)

        self._memory_usage_percentage = (
            memory_usage.peak_usage_bytes / memory_usage.memory_capacity_bytes
        )

        return memory_usage

    def measure_throughput(self):
        profiler = IterationProfiler.new_from(
            self._model_provider,
            self._input_provider,
            self._iteration_provider,
        )
        num_samples = 3
        samples = profiler.sample_run_time_ms_by_batch_size(
            start_batch_size=self._batch_size,
            memory_usage_percentage=self._memory_usage_percentage,
            num_samples=num_samples,
        )
        if len(samples) == 0 or samples[0].batch_size != self._batch_size:
            raise AnalysisError(
                "Something went wrong with Skyline when measuring your "
                "model's throughput. Please file a bug."
            )

        measured_throughput = (
            samples[0].batch_size / samples[0].run_time_ms * 1000
        )
        throughput = pm.ThroughputResponse()
        throughput.samples_per_second = measured_throughput
        throughput.predicted_max_samples_per_second = math.nan

        if len(samples) != num_samples:
            return throughput

        run_times = np.array(
            list(map(lambda sample: sample.run_time_ms, samples)))
        batches = np.array(
            list(map(lambda sample: sample.batch_size, samples)))
        stacked = np.vstack([batches, np.ones(len(batches))]).T
        slope, coefficient = np.linalg.lstsq(stacked, run_times, rcond=None)[0]
        logger.debug(
            "Run time model - Slope: %f, Coefficient: %f", slope, coefficient)

        predicted_max_throughput = 1000.0 / slope

        # Our prediction can be inaccurate due to sampling error or incorrect
        # assumptions. In these cases, we ignore our prediction. At the very
        # minimum, a good linear model has a positive slope and coefficient.
        if (slope < 1e-3 or coefficient < 1e-3 or
                measured_throughput > predicted_max_throughput):
            return throughput

        throughput.predicted_max_samples_per_second = predicted_max_throughput

        return throughput


def _run_entry_point(project_root, entry_point):
    file_name = os.path.join(project_root, entry_point)
    with open(file_name) as file:
        code_str = file.read()
    code = compile(code_str, file_name, mode="exec")
    scope = {}
    exec(code, scope, scope)
    return scope


def _set_file_context(message, project_root, entry):
    if entry.file_path is None or entry.line_number is None:
        return

    message.context.line_number = entry.line_number
    relative_file_path = os.path.relpath(entry.file_path, start=project_root)
    message.context.file_path.components.extend(
        relative_file_path.split(os.sep))


def _validate_providers(model_provider, input_provider, iteration_provider):
    model_sig = inspect.signature(model_provider)
    if len(model_sig.parameters) != 0:
        raise AnalysisError(
            "The model provider function cannot have any parameters."
        )

    input_sig = inspect.signature(input_provider)
    if (len(input_sig.parameters) != 1 or
            BATCH_SIZE_ARG not in input_sig.parameters or
            type(input_sig.parameters[BATCH_SIZE_ARG].default) is not int):
        raise AnalysisError(
            "The input provider function must have exactly one '{}' "
            "parameter with an integral default "
            "value.".format(BATCH_SIZE_ARG)
        )
    batch_size = input_sig.parameters[BATCH_SIZE_ARG].default

    iteration_sig = inspect.signature(iteration_provider)
    if len(iteration_sig.parameters) != 1:
        raise AnalysisError(
            "The iteration provider function must have exactly one "
            "parameter (the model being profiled)."
        )

    err = _validate_provider_return_values(
        model_provider, input_provider, iteration_provider)
    if err is not None:
        raise err

    return batch_size


def _validate_provider_return_values(
        model_provider, input_provider, iteration_provider):
    with exceptions_as_analysis_errors():
        # We return exceptions instead of raising them here to prevent
        # them from being caught by the exception context manager.
        model = model_provider()
        if not callable(model):
            return AnalysisError(
                "The model provider function must return a callable (i.e. "
                "return something that can be called like a PyTorch "
                "module or function)."
            )

        inputs = input_provider()
        try:
            input_iter = iter(inputs)
        except TypeError as ex:
            return AnalysisError(
                "The input provider function must return an iterable that "
                "contains the inputs for the model."
            )

        iteration = iteration_provider(model)
        if not callable(iteration):
            return AnalysisError(
                "The iteration provider function must return a callable "
                "(i.e. return something that can be called like a "
                "function)."
            )

        return None
