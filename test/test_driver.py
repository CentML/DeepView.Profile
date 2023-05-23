
import pytest
import json
from utils import SkylineSession, BackendContext

with open("config.json", "r") as fp:
    config = json.load(fp)

tests = list()
for entry_point in config["entry_points"]:
    tests.append((config["skyline_bin"], entry_point))

@pytest.mark.parametrize("skyline_bin,entry_point", tests)
def test_entry_point(skyline_bin, entry_point):
    print(f"Testing {entry_point}")
    context = BackendContext(skyline_bin, entry_point)
    context.spawn_process()

    sess = SkylineSession()
    while context.state == 0:
        pass
    sess.connect("localhost", 60120)
    sess.send_initialize_request()
    sess.send_analysis_request()
    while len(sess.received_messages) < 4:
        pass

    sess.cleanup()
    context.terminate()

    assert(len(sess.received_messages) == 4)
