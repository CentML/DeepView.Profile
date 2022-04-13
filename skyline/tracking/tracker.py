import enum
import logging

import torch

from skyline.tracking.backward_interceptor import BackwardInterceptor
from skyline.tracking.breakdown import HierarchicalBreakdownBuilder
from skyline.tracking.memory.activations import ActivationsTracker
from skyline.tracking.memory.report import MemoryReportBuilder, MiscSizeType
from skyline.tracking.memory.weights import WeightsTracker
from skyline.tracking.time.operation import OperationRunTimeTracker
from skyline.tracking.time.report import OperationRunTimeReportBuilder
from skyline.user_code_utils import user_code_environment

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(
        self,
        model_provider,
        iteration_provider,
        input_provider,
        project_root,
        user_code_path,
    ):
        self._model_provider = model_provider
        self._iteration_provider = iteration_provider
        self._input_provider = input_provider
        self._project_root = project_root
        self._user_code_path = user_code_path

        # Overall tracker state - used for correctness. Essentially, *if* the
        # caller wants to perform memory tracking, they must do so first before
        # run time tracking.
        self._tracker_state = _TrackerState.CREATED

        # Shared state
        self._model = None

        # Memory tracking state
        self._weight_tracker = None
        self._activations_tracker = None
        self._peak_usage_bytes = None

        # Run time tracking state
        self._operation_tracker = None

    def track_memory(self):
        if self._tracker_state != _TrackerState.CREATED:
            raise RuntimeError(
                'Memory tracking must be the first operation performed '
                'on a new Tracker.'
            )

        self._tracker_state = _TrackerState.MEMORY_TRACKED

        # Sanity check: For accurate profiling numbers, the initial memory
        # usage should be zero bytes.
        initial_memory_bytes = torch.cuda.memory_allocated()
        if initial_memory_bytes != 0:
            logger.debug(
                'Non-zero initial memory usage during memory tracking: '
                '%d bytes',
                initial_memory_bytes,
            )

        # 1. Track and record memory usage associated with model creation
        self._weight_tracker = WeightsTracker(self._project_root)
        with user_code_environment(self._user_code_path, self._project_root):
            with self._weight_tracker.track():
                self._model = self._model_provider()
            # Run one iteration to initialize the gradients
            iteration = self._iteration_provider(self._model)
            iteration(*self._input_provider())

        # 2. Track and record memory usage associated with stored activations
        self._activations_tracker = ActivationsTracker(self._project_root)
        self._activations_tracker.track_memory_usage(
            iteration, self._input_provider, self._user_code_path)

        # 3. Record peak memory usage
        torch.cuda.reset_max_memory_allocated()
        with user_code_environment(self._user_code_path, self._project_root):
            iteration(*(self._input_provider()))
        self._peak_usage_bytes = torch.cuda.max_memory_allocated()

    def track_run_time(self):
        if self._tracker_state == _TrackerState.CREATED:
            with user_code_environment(
                    self._user_code_path, self._project_root):
                self._model = self._model_provider()
        elif self._tracker_state != _TrackerState.MEMORY_TRACKED:
            raise RuntimeError('Run time tracking has already been performed.')

        self._tracker_state = _TrackerState.RUN_TIME_TRACKED

        # 2. Perform operation run time profiling
        with user_code_environment(self._user_code_path, self._project_root):
            inputs = self._input_provider()
            iteration = self._iteration_provider(self._model)

        self._operation_tracker = OperationRunTimeTracker(self._project_root)
        backward_interceptor = BackwardInterceptor()
        with self._operation_tracker.track():
            with backward_interceptor.intercept():
                with user_code_environment(
                        self._user_code_path, self._project_root):
                    iteration(*inputs)

    def get_memory_report(self, report_file=None):
        if (self._weight_tracker is None or
                self._activations_tracker is None or
                self._peak_usage_bytes is None):
            raise RuntimeError('Memory tracking has not been performed yet.')

        return (MemoryReportBuilder(report_file)
                .process_tracker(self._weight_tracker)
                .process_tracker(self._activations_tracker)
                .add_misc_entry(
                    MiscSizeType.PeakUsageBytes, self._peak_usage_bytes)
                .build())

    def get_run_time_report(self, report_file=None):
        if self._operation_tracker is None:
            raise RuntimeError('Run time tracking has not been performed yet.')

        return (OperationRunTimeReportBuilder(report_file)
                .process_tracker(self._operation_tracker)
                .build())

    def get_hierarchical_breakdown(self):
        if (self._weight_tracker is None or
                self._activations_tracker is None or
                self._peak_usage_bytes is None or
                self._operation_tracker is None):
            raise RuntimeError(
                'Memory tracking and run time tracking have not both been '
                'performed yet.'
            )

        return (HierarchicalBreakdownBuilder()
                .for_model(self._model)
                .set_peak_usage_bytes(self._peak_usage_bytes)
                .process_tracker(self._operation_tracker)
                .process_tracker(self._activations_tracker)
                .process_tracker(self._weight_tracker)
                .build())


class _TrackerState(enum.Enum):
    CREATED = 0
    MEMORY_TRACKED = 1
    RUN_TIME_TRACKED = 2
