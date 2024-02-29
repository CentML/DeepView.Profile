import pytest
import pickle
from utils import DeepviewSession, BackendContext
from google.protobuf.json_format import MessageToDict
from config_params import TestConfig
import os

REPS = 2
NUM_EXPECTED_MESSAGES = 6


def get_config_name():
    import pkg_resources

    package_versions = {p.key: p.version for p in pkg_resources.working_set}
    return package_versions


config = TestConfig()

tests = list()
for model_name in config["model_names_from_examples"]:
    dir_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples",
        model_name,
    )
    tests.append((model_name, dir_path))


@pytest.mark.parametrize("test_name, entry_point", tests)
def test_entry_point(test_name, entry_point):
    print(f"Testing {entry_point}")

    # create new folder
    folder = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/tests_results"
    )
    os.makedirs(folder, exist_ok=True)

    stdout_fd = open(os.path.join(folder, f"{test_name}_interactive_output.log"), "w")
    stderr_fd = open(os.path.join(folder, f"{test_name}_interactive_w_debug_output.log"), "w")
    context = BackendContext(entry_point, stdout_fd=stdout_fd, stderr_fd=stderr_fd)
    context.spawn_process()

    analysis_messages = list()

    for reps in range(REPS):
        sess = DeepviewSession()
        while context.state == 0:
            pass
        sess.connect("localhost", 60120)
        sess.send_initialize_request(entry_point)
        sess.send_analysis_request()
        while (
            context.alive()
            and sess.alive()
            and len(sess.received_messages) < NUM_EXPECTED_MESSAGES
        ):
            pass

        sess.cleanup()
        analysis_messages.extend(sess.received_messages)

        assert len(sess.received_messages) == NUM_EXPECTED_MESSAGES, (
            f"Run {reps}: Expected to receive {NUM_EXPECTED_MESSAGES} got "
            f"{len(sess.received_messages)} (did the process terminate prematurely?)"
        )

    context.terminate()
    # create folder to store files
    # flush contents to files
    with open(os.path.join(folder, f"{test_name}_analysis.pkl"), "wb") as fp:
        pickle.dump(list(map(MessageToDict, analysis_messages)), fp)
    # write package versions
    package_dict = get_config_name()
    with open(os.path.join(folder, "package-list.txt"), "w") as f:
        for k, v in package_dict.items():
            f.write(f"{k}={v}\n")
    stdout_fd.close()
    stderr_fd.close()
