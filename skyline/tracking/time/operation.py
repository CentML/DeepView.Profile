import collections

import torch

from skyline.tracking.call_stack import CallStack
from skyline.tracking.base import TrackerBase
from skyline.tracking.callable_tracker import CallableTracker
from skyline.tracking.utils import remove_dunder
from skyline.profiler.operation import OperationProfiler

OperationInfo = collections.namedtuple(
    'OperationInfo', ['operation_name', 'stack', 'forward_ms', 'backward_ms'])


class OperationRunTimeTracker(TrackerBase):
    def __init__(self, project_root):
        super().__init__()
        self._callable_tracker = CallableTracker(self._hook_creator)
        self._profiler = OperationProfiler()
        self._project_root = project_root
        self._processing_hook = False

        self.operations = []

    def start_tracking(self):
        super().start_tracking()
        self._callable_tracker.start_tracking()

    def stop_tracking(self):
        super().stop_tracking()
        self._callable_tracker.stop_tracking()

    def populate_report(self, builder):
        for op_info in self.operations:
            builder.add_run_time_entry(
                operation_name=remove_dunder(op_info.operation_name),
                forward_ms=op_info.forward_ms,
                backward_ms=op_info.backward_ms,
                stack_context=op_info.stack,
            )

    def populate_breakdown(self, builder):
        # The HierarchicalBreakdownBuilder uses the same run time entry API as
        # the OperationRunTimeReportBuilder.
        self.populate_report(builder)

    def _hook_creator(self, func):
        def hook(*args, **kwargs):
            # NOTE: We use self._processing_hook to handle cases where we have
            #       hooks on nested function calls.
            if self._processing_hook:
                return func(*args, **kwargs)

            self._processing_hook = True
            try:
                stack = CallStack.from_here(self._project_root, start_from=2)
                if len(stack.frames) == 0:
                    return func(*args, **kwargs)

                forward_ms, backward_ms = self._profiler.measure_operation_ms(
                    func, args, kwargs)
                self.operations.append(OperationInfo(
                    operation_name=func.__name__,
                    stack=stack,
                    forward_ms=forward_ms,
                    backward_ms=backward_ms,
                ))

                # Actually run the hooked function
                return func(*args, **kwargs)
            finally:
                self._processing_hook = False

        return hook
