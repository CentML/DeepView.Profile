import inspect
import logging
import math
import os

import numpy as np

import innpv.protocol_gen.innpv_pb2 as pm
from innpv.exceptions import AnalysisError
from innpv.profiler.iteration import IterationProfiler
from innpv.tracking.memory import track_memory_usage
from innpv.tracking.report import MiscSizeType

logger = logging.getLogger(__name__)

MODEL_PROVIDER_NAME = "innpv_model_provider"
INPUT_PROVIDER_NAME = "innpv_input_provider"
ITERATION_PROVIDER_NAME = "innpv_iteration_provider"
BATCH_SIZE_ARG = "batch_size"


class AnalysisSession:
    def __init__(
        self,
        project_root,
        model_provider,
        input_provider,
        iteration_provider
    ):
        self._project_root = project_root
        self._model_provider = model_provider
        self._input_provider = input_provider
        self._iteration_provider = iteration_provider
        self._batch_size = self._validate_providers()
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

        return cls(
            project_root,
            scope[MODEL_PROVIDER_NAME],
            scope[INPUT_PROVIDER_NAME],
            scope[ITERATION_PROVIDER_NAME],
        )

    def _validate_providers(self):
        model_sig = inspect.signature(self._model_provider)
        if len(model_sig.parameters) != 0:
            raise AnalysisError(
                "The model provider function cannot have any parameters."
            )

        input_sig = inspect.signature(self._input_provider)
        if (len(input_sig.parameters) != 1 or
                BATCH_SIZE_ARG not in input_sig.parameters or
                type(input_sig.parameters[BATCH_SIZE_ARG].default) is not int):
            raise AnalysisError(
                "The input provider function must have exactly one '{}' "
                "parameter with an integral default "
                "value.".format(BATCH_SIZE_ARG)
            )
        batch_size = input_sig.parameters[BATCH_SIZE_ARG].default

        iteration_sig = inspect.signature(self._iteration_provider)
        if len(iteration_sig.parameters) != 1:
            raise AnalysisError(
                "The iteration provider function must have exactly one "
                "parameter (the model being profiled)."
            )

        return batch_size

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
        samples = profiler.sample_run_time_ms_by_batch_size(
            self._batch_size)
        if len(samples) == 0 or samples[0].batch_size != self._batch_size:
            raise AnalysisError(
                "Something went wrong with INNPV when measuring your model's "
                "throughput. Please file a bug."
            )

        throughput = pm.ThroughputResponse()
        throughput.samples_per_second = (
            samples[0].batch_size / samples[0].run_time_ms * 1000
        )

        run_times = np.array(
            list(map(lambda sample: sample.run_time_ms, samples)))
        batches = np.array(
            list(map(lambda sample: sample.batch_size, samples)))
        stacked = np.vstack([batches, np.ones(len(batches))]).T
        slope, coefficient = np.linalg.lstsq(stacked, run_times, rcond=None)[0]
        logger.debug(
            "Run time model - Slope: %f, Coefficient: %f", slope, coefficient)

        if slope < 1e-3 or coefficient < 1e-3:
            # We expect the slope and coefficient to be positive. If they are
            # not, we ignore our prediction.
            throughput.predicted_max_samples_per_second = math.nan
            return throughput

        throughput.predicted_max_samples_per_second = 1000.0 / slope

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
