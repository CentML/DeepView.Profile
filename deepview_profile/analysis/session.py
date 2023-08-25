import ast
import inspect
import logging
import math
import os
import pynvml
import re
import torch
import numpy as np
import deepview_profile.protocol_gen.innpv_pb2 as pm
from deepview_profile.analysis.static import StaticAnalyzer
from deepview_profile.db.database import DatabaseInterface, EnergyTableInterface
from deepview_profile.energy.measurer import EnergyMeasurer
from deepview_profile.exceptions import AnalysisError, exceptions_as_analysis_errors
from deepview_profile.profiler.iteration import IterationProfiler
from deepview_profile.tracking.tracker import Tracker
from deepview_profile.user_code_utils import user_code_environment
from deepview_profile.profiler.utilization import utilization_analysis, serialize_response
from typing import List
import dill
import torch.multiprocessing as mp
import sys

logger = logging.getLogger(__name__)

# habitat imports
try:
    import collections
    import habitat
    from habitat.analysis import SPECIAL_OPERATIONS
    from habitat.profiling.run_time import RunTimeProfiler

    habitat_found = True
except ImportError:
    logger.debug("DeepView.Predict not found, GPU predictions not available")
    habitat_found = False


MODEL_PROVIDER_NAME = "deepview_model_provider"
INPUT_PROVIDER_NAME = "deepview_input_provider"
ITERATION_PROVIDER_NAME = "deepview_iteration_provider"
BATCH_SIZE_ARG = "batch_size"


# Habitat variables
Context = collections.namedtuple(
    "Context",
    ["origin_device", "profiler", "percentile"],
)


class AnalysisSession:
    def __init__(
        self,
        project_root,
        entry_point,
        path_to_entry_point_dir,
        model_provider,
        input_provider,
        iteration_provider,
        batch_size,
        entry_point_static_analyzer,
    ):
        self._project_root = project_root
        self._entry_point = entry_point
        self._path_to_entry_point_dir = path_to_entry_point_dir
        self._model_provider = model_provider
        self._input_provider = input_provider
        self._iteration_provider = iteration_provider
        self._batch_size = batch_size
        self._entry_point_static_analyzer = entry_point_static_analyzer
        self._profiler = None
        self._memory_usage_percentage = None
        self._batch_size_iteration_run_time_ms = None
        self._batch_size_peak_usage_bytes = None
        self._utilization = None
        self._energy_table_interface = EnergyTableInterface(
            DatabaseInterface().connection
        )

    @classmethod
    def new_from(cls, project_root, entry_point):
        path_to_entry_point = os.path.join(project_root, entry_point)
        # Note: This is not necessarily the same as project_root because the
        #       entry_point could be in a subdirectory.
        path_to_entry_point_dir = os.path.dirname(path_to_entry_point)

        # 1. Run the entry point file to "load" the model
        entry_point_code, entry_point_ast, scope = _run_entry_point(
            path_to_entry_point,
            path_to_entry_point_dir,
            project_root,
        )

        # 2. Check that the model provider and input provider functions exist
        if MODEL_PROVIDER_NAME not in scope:
            raise AnalysisError(
                "The project entry point file is missing a model provider "
                "function. Please add a model provider function named "
                '"{}".'.format(MODEL_PROVIDER_NAME)
            ).with_file_context(entry_point)

        if INPUT_PROVIDER_NAME not in scope:
            raise AnalysisError(
                "The project entry point file is missing an input provider "
                "function. Please add an input provider function named "
                '"{}".'.format(INPUT_PROVIDER_NAME)
            ).with_file_context(entry_point)

        if ITERATION_PROVIDER_NAME not in scope:
            raise AnalysisError(
                "The project entry point file is missing an iteration "
                "provider function. Please add an iteration provider function "
                'named "{}".'.format(ITERATION_PROVIDER_NAME)
            ).with_file_context(entry_point)

        batch_size = _validate_providers_signatures(
            scope[MODEL_PROVIDER_NAME],
            scope[INPUT_PROVIDER_NAME],
            scope[ITERATION_PROVIDER_NAME],
            entry_point,
        )

        (
            model_provider,
            input_provider,
            iteration_provider,
        ) = _wrap_providers_with_validators(
            scope[MODEL_PROVIDER_NAME],
            scope[INPUT_PROVIDER_NAME],
            scope[ITERATION_PROVIDER_NAME],
            entry_point,
        )

        return cls(
            project_root,
            entry_point,
            path_to_entry_point_dir,
            model_provider,
            input_provider,
            iteration_provider,
            batch_size,
            StaticAnalyzer(entry_point_code, entry_point_ast),
        )

    def measure_utilization(self):
        resp = pm.UtilizationResponse()
        payload = dill.dumps((self._model_provider, self._input_provider,
                             self._iteration_provider, logging.root.level))
        analysis_results = dispatch_process(utilization_analysis, payload,
                                            self._path_to_entry_point_dir)
        if analysis_results.get("error"):
            resp.analysis_error.error_message = analysis_results["error"]
            return resp
        serialize_response(resp.rootNode, analysis_results["root_node"])
        resp.tensor_utilization = float(analysis_results["tensor_core_perc"])

        return resp

    def energy_compute(self) -> pm.EnergyResponse:
        energy_measurer = EnergyMeasurer()

        model = self._model_provider()
        inputs = self._input_provider()
        iteration = self._iteration_provider(model)
        resp = pm.EnergyResponse()

        try:
            energy_measurer.begin_measurement()
            iterations = 20
            for _ in range(iterations):
                iteration(*inputs)
            energy_measurer.end_measurement()
            resp.total_consumption = energy_measurer.total_energy() / float(iterations)
            resp.batch_size = self._batch_size

            components = []
            components_joules = []

            if energy_measurer.cpu_energy() is not None:
                cpu_component = pm.EnergyConsumptionComponent()
                cpu_component.component_type = pm.ENERGY_CPU_DRAM
                cpu_component.consumption_joules = energy_measurer.cpu_energy() / float(
                    iterations
                )
                components.append(cpu_component)
                components_joules.append(cpu_component.consumption_joules)
            else:
                cpu_component = pm.EnergyConsumptionComponent()
                cpu_component.component_type = pm.ENERGY_CPU_DRAM
                cpu_component.consumption_joules = 0.0
                components.append(cpu_component)
                components_joules.append(cpu_component.consumption_joules)

            gpu_component = pm.EnergyConsumptionComponent()
            gpu_component.component_type = pm.ENERGY_NVIDIA
            gpu_component.consumption_joules = energy_measurer.gpu_energy() / float(
                iterations
            )
            components.append(gpu_component)
            components_joules.append(gpu_component.consumption_joules)

            resp.components.extend(components)

            # get last 10 runs if they exist
            path_to_entry_point = os.path.join(
                self._project_root, self._entry_point)
            past_runs = (
                self._energy_table_interface.get_latest_n_entries_of_entry_point(
                    10, path_to_entry_point
                )
            )
            resp.past_measurements.extend(
                _convert_to_energy_responses(past_runs))

            # add current run to database
            current_entry = [path_to_entry_point] + components_joules
            current_entry.append(self._batch_size)
            self._energy_table_interface.add_entry(current_entry)
        except AnalysisError as ex:
            message = str(ex)
            logger.error(message)
            resp.analysis_error.error_message = message
        except Exception as ex:
            message = str(ex)
            logger.error(message)
            resp.analysis_error.error_message = (
                f"There was an error obtaining energy measurements: {message}"
            )
        finally:
            return resp

    def habitat_predict(self):
        resp = pm.HabitatResponse()

        payload = dill.dumps((self._model_provider, self._input_provider,
                             self._iteration_provider, logging.root.level))
        analysis_results = dispatch_process(habitat_workload, payload,
                                            self._path_to_entry_point_dir)
        if analysis_results.get("error"):
            resp.analysis_error.error_message = analysis_results["error"]
            return resp
        for res in analysis_results["data"]:
            pred = pm.HabitatDevicePrediction()
            pred.device_name = res["device_name"]
            pred.runtime_ms = res["runtime_ms"]
            resp.predictions.append(pred)

        return resp

    def measure_breakdown(self, nvml):
        # 1. Measure the breakdown entries
        self._prepare_for_memory_profiling()
        tracker = self._get_tracker_instance()
        tracker.track_memory()
        tracker.track_run_time()
        breakdown = tracker.get_hierarchical_breakdown()
        del tracker

        # 2. Measure the overall iteration run time
        if self._batch_size_iteration_run_time_ms is None:
            if self._profiler is None:
                self._initialize_iteration_profiler()

            (
                self._batch_size_iteration_run_time_ms,
                self._batch_size_peak_usage_bytes,
                _,
            ) = self._profiler.measure_run_time_ms(self._batch_size)

        # 3. Serialize the measured data
        bm = pm.BreakdownResponse()
        bm.batch_size = self._batch_size
        bm.peak_usage_bytes = breakdown.peak_usage_bytes
        bm.memory_capacity_bytes = nvml.get_memory_capacity().total
        bm.iteration_run_time_ms = self._batch_size_iteration_run_time_ms
        breakdown.operations.serialize_to_protobuf(bm.operation_tree)
        breakdown.weights.serialize_to_protobuf(bm.weight_tree)

        # 4. Bookkeeping for the throughput measurements
        self._memory_usage_percentage = bm.peak_usage_bytes / bm.memory_capacity_bytes

        return bm

    def measure_throughput(self):
        if self._profiler is None:
            self._initialize_iteration_profiler()

        # 1. Measure the throughput at several spots to be able to build a
        #    prediction model
        num_samples = 3
        samples = self._profiler.sample_run_time_ms_by_batch_size(
            start_batch_size=self._batch_size,
            memory_usage_percentage=self._memory_usage_percentage,
            start_batch_size_run_time_ms=self._batch_size_iteration_run_time_ms,
            num_samples=num_samples,
        )
        if len(samples) == 0 or samples[0].batch_size != self._batch_size:
            raise AnalysisError(
                "Something went wrong with DeepView.Profile when measuring your "
                "model's throughput. Please file a bug."
            )

        # 2. Begin filling in the throughput response
        logger.debug("sampling results \n %r" % str(samples))
        measured_throughput = samples[0].batch_size / \
            samples[0].run_time_ms * 1000
        throughput = pm.ThroughputResponse()
        throughput.samples_per_second = measured_throughput
        throughput.predicted_max_samples_per_second = math.nan
        throughput.can_manipulate_batch_size = False

        # 3. Determine whether we have information about the batch size
        #    location in the code
        batch_info = self._entry_point_static_analyzer.batch_size_location()
        if batch_info is not None:
            throughput.batch_size_context.line_number = batch_info[0]
            throughput.can_manipulate_batch_size = batch_info[1]
            throughput.batch_size_context.file_path.components.extend(
                self._entry_point.split(os.sep)
            )

        # 4. If we do not have enough throughput samples, we cannot build any
        #    prediction models so just return the message as-is
        if len(samples) != num_samples:
            return throughput

        # 5. Build and validate the prediction models for run time (throughput)
        #    and memory
        batches = list(map(lambda sample: sample.batch_size, samples))
        run_times = list(map(lambda sample: sample.run_time_ms, samples))
        usages = list(map(lambda sample: sample.peak_usage_bytes, samples))

        run_time_model = _fit_linear_model(batches, run_times)
        peak_usage_model = _fit_linear_model(batches, usages)

        logger.debug(
            "Run time model - Slope: %f, Bias: %f (ms)",
            *run_time_model,
        )
        logger.debug(
            "Peak usage model - Slope: %f, Bias: %f (bytes)",
            *peak_usage_model,
        )

        throughput.peak_usage_bytes.slope = peak_usage_model[0]
        throughput.peak_usage_bytes.bias = peak_usage_model[1]

        predicted_max_throughput = 1000.0 / run_time_model[0]

        # Our prediction can be inaccurate due to sampling error or incorrect
        # assumptions. In these cases, we ignore our prediction. At the very
        # minimum, a good linear model has a positive slope and bias.
        # if (run_time_model[0] < 1e-3 or run_time_model[1] < 1e-3 or
        if run_time_model[0] < 1e-3 or measured_throughput > predicted_max_throughput:
            return throughput

        throughput.predicted_max_samples_per_second = predicted_max_throughput
        throughput.run_time_ms.slope = run_time_model[0]
        throughput.run_time_ms.bias = run_time_model[1]

        return throughput

    def measure_peak_usage_bytes(self):
        self._prepare_for_memory_profiling()
        # Run one iteration to initialize the gradients
        with user_code_environment(self._path_to_entry_point_dir, self._project_root):
            model = self._model_provider()
            iteration = self._iteration_provider(model)
            iteration(*self._input_provider(batch_size=self._batch_size))

        torch.cuda.reset_max_memory_allocated()
        with user_code_environment(self._path_to_entry_point_dir, self._project_root):
            # NOTE: It's important to run at least 2 iterations here. It turns
            #       out that >= 2 iterations is the number of iterations needed
            #       to get a stable measurement of the total memory
            #       consumption. When using Adam, if you run one iteration, the
            #       memory usage ends up being too low by a constant factor.
            for _ in range(2):
                iteration(*(self._input_provider(batch_size=self._batch_size)))
        return torch.cuda.max_memory_allocated()

    def generate_memory_usage_report(self, save_report_to):
        self._prepare_for_memory_profiling()
        tracker = self._get_tracker_instance()
        tracker.track_memory()
        tracker.get_memory_report(report_file=save_report_to)

    def generate_run_time_breakdown_report(self, save_report_to):
        tracker = self._get_tracker_instance()
        tracker.track_run_time()
        tracker.get_run_time_report(report_file=save_report_to)

    def _initialize_iteration_profiler(self):
        self._profiler = IterationProfiler.new_from(
            self._model_provider,
            self._input_provider,
            self._iteration_provider,
            self._path_to_entry_point_dir,
            self._project_root,
        )

    def _prepare_for_memory_profiling(self):
        if self._profiler is not None:
            # It's important that the IterationProfiler is uninitialized here
            # because it stores a copy of the model, which takes up GPU memory.
            # This would skew our memory profiling results.
            del self._profiler
            self._profiler = None

    def _get_tracker_instance(self):
        return Tracker(
            model_provider=self._model_provider,
            iteration_provider=self._iteration_provider,
            input_provider=self._input_provider,
            project_root=self._project_root,
            user_code_path=self._path_to_entry_point_dir,
        )


def _run_entry_point(path_to_entry_point, path_to_entry_point_dir, project_root):
    with open(path_to_entry_point) as file:
        code_str = file.read()
    with exceptions_as_analysis_errors(project_root):
        tree = ast.parse(code_str, filename=path_to_entry_point)
        code = compile(tree, path_to_entry_point, mode="exec")
    with user_code_environment(path_to_entry_point_dir, project_root):
        scope = {}
        exec(code, scope, scope)
    return code_str, tree, scope


def _validate_providers_signatures(
    model_provider,
    input_provider,
    iteration_provider,
    entry_point,
):
    model_sig = inspect.signature(model_provider)
    if len(model_sig.parameters) != 0:
        raise AnalysisError(
            "The model provider function cannot have any parameters."
        ).with_file_context(entry_point)

    input_sig = inspect.signature(input_provider)
    if (
        len(input_sig.parameters) != 1
        or BATCH_SIZE_ARG not in input_sig.parameters
        or type(input_sig.parameters[BATCH_SIZE_ARG].default) is not int
    ):
        raise AnalysisError(
            "The input provider function must have exactly one '{}' "
            "parameter with an integral default "
            "value.".format(BATCH_SIZE_ARG)
        ).with_file_context(entry_point)
    batch_size = input_sig.parameters[BATCH_SIZE_ARG].default

    iteration_sig = inspect.signature(iteration_provider)
    if len(iteration_sig.parameters) != 1:
        raise AnalysisError(
            "The iteration provider function must have exactly one "
            "parameter (the model being profiled)."
        ).with_file_context(entry_point)

    return batch_size


def _wrap_providers_with_validators(
    model_provider,
    input_provider,
    iteration_provider,
    entry_point,
):
    def model_provider_wrapped():
        model = model_provider()
        if not callable(model):
            raise AnalysisError(
                "The model provider function must return a callable (i.e. "
                "return something that can be called like a PyTorch "
                "module or function)."
            ).with_file_context(entry_point)
        return model

    def input_provider_wrapped(batch_size=None):
        if batch_size is None:
            inputs = input_provider()
        else:
            inputs = input_provider(batch_size=batch_size)

        if isinstance(inputs, torch.Tensor):
            raise AnalysisError(
                "The input provider function must return an iterable that "
                "contains the inputs for the model. If your model only takes "
                "a single tensor as input, return a single element tuple or "
                "list in your input provider (e.g., 'return [the_input]')."
            ).with_file_context(entry_point)

        try:
            iter(inputs)
            return inputs
        except TypeError:
            raise AnalysisError(
                "The input provider function must return an iterable that "
                "contains the inputs for the model."
            ).with_file_context(entry_point)

    def iteration_provider_wrapped(model):
        iteration = iteration_provider(model)
        if not callable(iteration):
            raise AnalysisError(
                "The iteration provider function must return a callable "
                "(i.e. return something that can be called like a "
                "function)."
            ).with_file_context(entry_point)
        return iteration

    return (
        model_provider_wrapped,
        input_provider_wrapped,
        iteration_provider_wrapped,
    )


def _fit_linear_model(x, y):
    y_np = np.array(y)
    x_np = np.array(x)
    stacked = np.vstack([x_np, np.ones(len(x_np))]).T
    slope, bias = np.linalg.lstsq(stacked, y_np, rcond=None)[0]
    # Linear model: y = slope * x + bias
    return slope, bias


def _convert_to_energy_responses(entries: list) -> List[pm.EnergyResponse]:
    energy_response_list = []
    for entry in entries:
        if EnergyTableInterface.is_valid_entry_with_timestamp(entry):
            energy_response = pm.EnergyResponse()
            cpu_component = pm.EnergyConsumptionComponent()
            cpu_component.component_type = pm.ENERGY_CPU_DRAM
            cpu_component.consumption_joules = entry[1]

            gpu_component = pm.EnergyConsumptionComponent()
            gpu_component.component_type = pm.ENERGY_NVIDIA
            gpu_component.consumption_joules = entry[2]

            energy_response.total_consumption = (
                gpu_component.consumption_joules + cpu_component.consumption_joules
            )
            energy_response.components.extend([cpu_component, gpu_component])

            energy_response.batch_size = entry[3]
            energy_response_list.append(energy_response)

    return energy_response_list


def dispatch_process(func, payload, path_to_entry_point_dir):
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=func, args=(
        queue, payload, path_to_entry_point_dir), daemon=True)
    p.start()
    results = queue.get()
    p.join()
    return results


def habitat_workload(queue, payload, path_to_entry_point_dir):
    sys.path.append(path_to_entry_point_dir)

    def habitat_compute_threshold(runnable, context):
        tracker = habitat.OperationTracker(context.origin_device)
        with tracker.track():
            runnable()

        run_times = []
        trace = tracker.get_tracked_trace()
        for op in trace.operations:
            if op.name in SPECIAL_OPERATIONS:
                continue
            run_times.append(op.forward.run_time_ms)
            if op.backward is not None:
                run_times.append(op.backward.run_time_ms)

        return np.percentile(run_times, context.percentile)

    try:
        model_provider, input_provider, iteration_provider, logging_level = dill.loads(
            payload)
        resp = []

        if not habitat_found:
            logger.debug(
                "Skipping deepview predictions, returning empty response.")
            queue.put("habitat not found")
            return

        print("deepview_predict: begin")
        DEVICES = [
            habitat.Device.P100,
            habitat.Device.P4000,
            habitat.Device.RTX2070,
            habitat.Device.RTX2080Ti,
            habitat.Device.T4,
            habitat.Device.V100,
            habitat.Device.A100,
            habitat.Device.RTX3090,
            habitat.Device.A40,
            habitat.Device.A4000,
            habitat.Device.RTX4000
        ]

        # Detect source GPU
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() == 0:
            raise Exception("NVML failed to find a GPU. Please ensure that you \
            have a NVIDIA GPU installed and that the drivers are functioning \
            correctly.")

        # TODO: Consider profiling on not only the first detected GPU
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        source_device_name = pynvml.nvmlDeviceGetName(
            nvml_handle).decode("utf-8")
        split_source_device_name = re.split(r"-|\s|_|\\|/", source_device_name)
        source_device = (
            None if logging_level > logging.DEBUG else habitat.Device.T4
        )
        for device in DEVICES:
            if device.name in "".join(split_source_device_name):
                source_device = device
        pynvml.nvmlShutdown()
        if not source_device:
            logger.debug(
                "Skipping DeepView predictions,\
                source not in list of supported GPUs."
            )
            src = {}
            src["device_name"] = "unavailable"
            src["runtime_ms"] = -1
            resp.append(src)
            print(
                f"\ndevice_name:{src['device_name']}\ttime:{src['runtime_ms']}")
            queue.put({"data": resp})
            return

        print("deepview_predict: detected source device", source_device.name)

        # get model
        model = model_provider()
        inputs = input_provider()
        iteration = iteration_provider(model)

        def runnable():
            iteration(*inputs)

        profiler = RunTimeProfiler()

        context = Context(
            origin_device=source_device, profiler=profiler, percentile=99.5
        )

        threshold = habitat_compute_threshold(runnable, context)

        tracker = habitat.OperationTracker(
            device=context.origin_device,
            metrics=[
                habitat.Metric.SinglePrecisionFLOPEfficiency,
                habitat.Metric.DRAMReadBytes,
                habitat.Metric.DRAMWriteBytes,
            ],
            metrics_threshold_ms=threshold,
        )

        with tracker.track():
            iteration(*inputs)

        print("deepview_predict: tracing on origin device")
        trace = tracker.get_tracked_trace()

        src = {}
        src["device_name"] = "source"
        src["runtime_ms"] = trace.run_time_ms
        resp.append(src)
        print(f"\ndevice_name:{src['device_name']}\ttime:{src['runtime_ms']}")
        for device in DEVICES:
            print("deepview_predict: predicting for", device)
            predicted_trace = trace.to_device(device)

            pred = {}
            pred["device_name"] = device.name
            pred["runtime_ms"] = predicted_trace.run_time_ms
            resp.append(pred)
            print(
                f"\ndevice_name:{pred['device_name']}\ttime:{pred['runtime_ms']}")

        queue.put({"data": resp})
    except AnalysisError as ex:
        message = str(ex)
        logger.error(message)
        queue.put({"error": message})
    except Exception as ex:
        message = str(ex)
        logger.error(message)
        queue.put({"error": message})
