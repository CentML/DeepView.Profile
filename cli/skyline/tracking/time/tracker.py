from skyline.tracking.time.operation import OperationRunTimeTracker
from skyline.tracking.time.report import OperationRunTimeReportBuilder
from skyline.user_code_utils import user_code_environment


def track_operation_run_time(
    model_provider,
    input_provider,
    user_code_path,
    report_file=None,
):
    with user_code_environment(user_code_path):
        inputs = input_provider()
        model = model_provider()

    operation_tracker = OperationRunTimeTracker()
    with operation_tracker.track(), user_code_environment(user_code_path):
        out = model(*inputs)

    return (OperationRunTimeReportBuilder(report_file)
            .process_tracker(operation_tracker)
            .build())
