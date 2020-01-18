from skyline.tracking.time.iteration import IterationTracker
from skyline.user_code_utils import user_code_environment


def track_iteration_run_time(
    model_provider,
    input_provider,
    user_code_path,
    report_file=None,
):
    with user_code_environment(user_code_path):
        inputs = input_provider()
        model = model_provider()

    iteration_tracker = IterationTracker()
    with iteration_tracker.track(), user_code_environment(user_code_path):
        out = model(*inputs)

    return iteration_tracker.operations
