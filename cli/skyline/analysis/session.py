import inspect
import logging
import math
import os

import numpy as np

import skyline.protocol_gen.innpv_pb2 as pm
from skyline.exceptions import AnalysisError
from skyline.profiler.iteration import IterationProfiler
from skyline.tracking.memory.tracker import track_memory_usage
from skyline.tracking.memory.report import MiscSizeType
from skyline.tracking.time.tracker import track_operation_run_time
from skyline.user_code_utils import user_code_environment

logger = logging.getLogger(__name__)

MODEL_PROVIDER_NAME = "skyline_model_provider"
INPUT_PROVIDER_NAME = "skyline_input_provider"
ITERATION_PROVIDER_NAME = "skyline_iteration_provider"
BATCH_SIZE_ARG = "batch_size"


class AnalysisSession:
    def __init__(
        self,
        project_root,
        path_to_entry_point_dir,
        model_provider,
        input_provider,
        iteration_provider,
        batch_size
    ):
        self._project_root = project_root
        self._path_to_entry_point_dir = path_to_entry_point_dir
        self._model_provider = model_provider
        self._input_provider = input_provider
        self._iteration_provider = iteration_provider
        self._batch_size = batch_size
        self._profiler = None
        self._memory_usage_percentage = None
        self._batch_size_iteration_run_time_ms = None

    @classmethod
    def new_from(cls, project_root, entry_point):
        path_to_entry_point = os.path.join(project_root, entry_point)
        # Note: This is not necessarily the same as project_root because the
        #       entry_point could be in a subdirectory.
        path_to_entry_point_dir = os.path.dirname(path_to_entry_point)

        # 1. Run the entry point file to "load" the model
        try:
            scope = _run_entry_point(
                path_to_entry_point,
                path_to_entry_point_dir,
            )
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
            model_provider,
            input_provider,
            iteration_provider,
            path_to_entry_point_dir,
        )

        return cls(
            project_root,
            path_to_entry_point_dir,
            model_provider,
            input_provider,
            iteration_provider,
            batch_size,
        )

    def measure_memory_usage(self, nvml):
        self._prepare_for_memory_profiling()

        report = track_memory_usage(
            self._model_provider,
            self._input_provider,
            self._iteration_provider,
            self._path_to_entry_point_dir,
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
        if self._profiler is None:
            self._initialize_iteration_profiler()

        num_samples = 3
        samples = self._profiler.sample_run_time_ms_by_batch_size(
            start_batch_size=self._batch_size,
            memory_usage_percentage=self._memory_usage_percentage,
            start_batch_size_run_time_ms=self._batch_size_iteration_run_time_ms,
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

    def measure_run_time_breakdown(self):
        run_time_report = track_operation_run_time(
            self._model_provider,
            self._input_provider,
            self._path_to_entry_point_dir,
        )
        if self._batch_size_iteration_run_time_ms is None:
            if self._profiler is None:
                self._initialize_iteration_profiler()

            self._batch_size_iteration_run_time_ms, _ = \
                self._profiler.measure_run_time_ms(self._batch_size)

        run_time = pm.RunTimeResponse()
        run_time.iteration_run_time_ms = self._batch_size_iteration_run_time_ms

        for run_time_entry in run_time_report.get_run_time_entries(
                path_prefix=self._project_root):
            entry = run_time.run_time_entries.add()
            entry.operation_name = run_time_entry.operation_name
            entry.forward_ms = run_time_entry.forward_ms
            if run_time_entry.backward_ms is not None:
                entry.backward_ms = run_time_entry.backward_ms
            else:
                entry.backward_ms = math.nan
            _set_file_context(entry, self._project_root, run_time_entry)

        return run_time

    def generate_memory_usage_report(self, save_report_to):
        self._prepare_for_memory_profiling()

        track_memory_usage(
            self._model_provider,
            self._input_provider,
            self._iteration_provider,
            self._path_to_entry_point_dir,
            report_file=save_report_to,
        )

    def generate_run_time_breakdown_report(self, save_report_to):
        track_operation_run_time(
            self._model_provider,
            self._input_provider,
            self._path_to_entry_point_dir,
            report_file=save_report_to,
        )

    def _initialize_iteration_profiler(self):
        self._profiler = IterationProfiler.new_from(
            self._model_provider,
            self._input_provider,
            self._iteration_provider,
            self._path_to_entry_point_dir,
        )

    def _prepare_for_memory_profiling(self):
        if self._profiler is not None:
            # It's important that the IterationProfiler is uninitialized here
            # because it stores a copy of the model, which takes up GPU memory.
            # This would skew our memory profiling results.
            del self._profiler
            self._profiler = None


def _run_entry_point(path_to_entry_point, path_to_entry_point_dir):
    with open(path_to_entry_point) as file:
        code_str = file.read()
    code = compile(code_str, path_to_entry_point, mode="exec")
    with user_code_environment(path_to_entry_point_dir):
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


def _validate_providers(
    model_provider,
    input_provider,
    iteration_provider,
    path_to_entry_point_dir,
):
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
        model_provider,
        input_provider,
        iteration_provider,
        path_to_entry_point_dir,
    )
    if err is not None:
        raise err

    return batch_size


def _validate_provider_return_values(
    model_provider,
    input_provider,
    iteration_provider,
    path_to_entry_point_dir,
):
    with user_code_environment(path_to_entry_point_dir):
        # We return exceptions instead of raising them here to prevent
        # them from being caught by the code environment context manager.
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
