from datetime import date, datetime
import e2e_utils
import json
from jsonschema import validate
from math import nan
import numpy as np
import os
import pickle
import torch
from typing import List, Tuple
import subprocess
import sys
sys.path.append('./..')
from utils import BackendContext, SkylineSession


class _ModelMetrics:
    def __init__(self, name, batch_sizes=[], baseline_metrics=[], skyline_metrics=[]):
        assert(len(batch_sizes) == len(baseline_metrics) == len(skyline_metrics))
        self.name = name
        self.batchSizes = batch_sizes
        self.baselineThroughputs = [metric.samples_per_second for metric in baseline_metrics]
        self.baselinePeakUsages = [metric.peak_usage_bytes for metric in baseline_metrics]
        self.skylineThroughputs = [metric.samples_per_second for metric in skyline_metrics]
        self.skylinePeakUsages = [metric.peak_usage_bytes for metric in skyline_metrics]
        self.throughputPercentageDiffs = (np.abs(np.array(self.baselineThroughputs) - np.array(self.skylineThroughputs))/np.array(self.baselineThroughputs)).tolist()
        self.peakUsagePercentageDiffs = (np.abs(np.array(self.baselinePeakUsages) - np.array(self.skylinePeakUsages))/np.array(self.baselinePeakUsages)).tolist()


def _update_entry_point_batch_size(file_name, original_batch_size, target_batch_size):
    if original_batch_size == target_batch_size:
        pass
    
    with open(file_name, 'r') as file :
        filedata = file.read()

    # ensure that batch_size is a parameter
    assert(filedata.find("batch_size=") != -1)
    
    original_string = "batch_size=" + str(original_batch_size)
    replacement_string = "batch_size=" + str(target_batch_size)

    filedata = filedata.replace(original_string, replacement_string)

    with open(file_name, 'w') as file:
        file.write(filedata)

class ModelMetricsRunner:
    def __init__(self, model_name, skyline_bin, entry_point):
        self.model_name = model_name
        self._skyline_bin = skyline_bin
        self._entry_point = entry_point
        self.model_metrics = self._get_model_metrics()

    def _get_baseline_values(self) -> Tuple[int, List[e2e_utils.ProfilingMetrics]]:
        # Note that we run this in a separate process to make sure there is no overhead GPU memory remaining when we run
        # skyline profiling
        baseline_test_runner_cmd = [sys.executable, "baseline_test_runner.py", self.model_name, self._entry_point]
        result = subprocess.run(baseline_test_runner_cmd)
        if type(result) == subprocess.CompletedProcess:
            file_name = self.model_name + '.pkl'
            baseline_profiling_metrics_list = []
            given_batch_size = nan
            with open(file_name, 'rb') as f:
                baseline_metrics = pickle.load(f)
                # some sanity checks
                assert(type(baseline_metrics) == e2e_utils.BaselineMetrics)
                assert(baseline_metrics.model_name == self.model_name)
                assert(baseline_metrics.entry_point == self._entry_point)
                given_batch_size = baseline_metrics.given_batch_size
                baseline_profiling_metrics_list = baseline_metrics.baseline_metric_list
            # clean up the temp file
            if os.path.exists(file_name):
                os.remove(file_name)
            return given_batch_size, baseline_profiling_metrics_list
        else:
            raise RuntimeError("Cannot get baseline test results!")
        

    def _get_skyline_metrics(self) -> e2e_utils.ProfilingMetrics:
        context = BackendContext(self._skyline_bin, self._entry_point)
        context.spawn_process()

        sess = SkylineSession()
        while context.state == 0: pass
        sess.connect("localhost", 60120)
        sess.send_initialize_request()
        sess.send_analysis_request()
        while len(sess.received_messages) < 4: pass

        sess.cleanup()
        context.terminate()

        assert(len(sess.received_messages) == 4)
        
        metrics = e2e_utils.ProfilingMetrics()

        for msg in sess.received_messages:
            if msg.HasField("breakdown"):
                breakdown_response = msg.breakdown
                metrics.batch_size = breakdown_response.batch_size
            if msg.HasField("throughput"):
                throughput_response = msg.throughput
                metrics.samples_per_second = throughput_response.samples_per_second
                metrics.peak_usage_bytes = metrics.batch_size*throughput_response.peak_usage_bytes.slope + throughput_response.peak_usage_bytes.bias
        return metrics
    
    def _get_model_metrics(self) -> _ModelMetrics:
        given_batch_size, baseline_metrics = self._get_baseline_values()
        skyline_metrics = []
        batch_sizes = [baseline_metric.batch_size for baseline_metric in baseline_metrics]
        for iter_batch_size in batch_sizes:
            _update_entry_point_batch_size(self._entry_point, given_batch_size, iter_batch_size)
            skyline_metrics.append(self._get_skyline_metrics())
            _update_entry_point_batch_size(self._entry_point, iter_batch_size, given_batch_size)
        return _ModelMetrics(self.model_name, batch_sizes, baseline_metrics, skyline_metrics)

def get_formatted_gpu_name() -> str:
    return torch.cuda.get_device_name().replace(" ", "_")

def get_metrics_file_name() -> str:
    return "results/" + date.today().isoformat() + "_({})".format(get_formatted_gpu_name())

def test_model_metrics_json_dump():
    with open("config.json", "r") as fp:
        config = json.load(fp)
    skyline_bin = config["skyline_bin"]
    entry_points = config["entry_points"]
    model_names = config["model_names"]
    assert len(entry_points) == len(model_names)

    gpu_metrics = {"gpuId": get_formatted_gpu_name(), "timeStamp": datetime.now().isoformat(), "modelMetrics": [] }
    for i in range(len(entry_points)):
        model_metrics_runner = ModelMetricsRunner(model_names[i], skyline_bin, entry_points[i])
        gpu_metrics["modelMetrics"].append(model_metrics_runner.model_metrics.__dict__)
    if not os.path.exists("results"):
        os.mkdir("results")
    file_name = get_metrics_file_name()
    with open(file_name, "w") as outfile:
        json.dump(gpu_metrics, outfile)
    test_json_validity

def test_json_validity():
    # validate against the schema
    with open("skyline_test_report.schema.json", "r") as fp:
        schema = json.load(fp)
    with open(get_metrics_file_name(), "r") as fp:
        metrics = json.load(fp)
    assert validate(metrics, schema)


if __name__ == "__main__":
    test_model_metrics_json_dump()