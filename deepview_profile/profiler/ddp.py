from scipy.stats import gaussian_kde
import numpy as np
import os
import logging
from deepview_profile.user_code_utils import user_code_environment
from deepview_profile.exceptions import AnalysisError
from deepview_profile.pytorch_profiler_log_reader import (
    get_first_last_step,
    get_bucket_sizes,
    get_single_gpu_backward_info,
)
import dill
import time
from torch.profiler import profile, schedule, ProfilerActivity
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import subprocess

logger = logging.getLogger("ddp")
logger.setLevel(logging.DEBUG)
FILENAME = "pytorch_profiler.json"
RANK = 0
WORLD_SIZE = 1
DEFAULT_BUCKET_SIZE = 25


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _bucket_estimate_max_expected(bucket_times, ngpu):
    m = 1000

    np_samples = np.array(bucket_times)

    kde_samples = gaussian_kde(np_samples)

    z_arr = []
    for _ in range(m):
        num_resamples = kde_samples.resample(ngpu)

        z_arr.append(np.max(num_resamples))

    expected_max = np.mean(z_arr)

    return expected_max


def _bucket_comp_times(path_to_file):
    data_matrix = []
    first_step, last_step = get_first_last_step(path_to_file)
    NUM_STEPS = 25
    forward_time_acc = 0
    for step in range(first_step + 1, first_step + NUM_STEPS + 1):
        fw_time, bucket_comp_times = get_single_gpu_backward_info(path_to_file, step)
        forward_time_acc += fw_time
        """
        storing as:
        [bucket_0 time1, bucket_1 time1, ... , bucket_n time1]
        [bucket_0 time2, bucket_1 time2, ... , bucket_n time2]
        ...
        """
        data_matrix.append(bucket_comp_times)
    # convert to numpy and transpose
    data_numpy = np.array(data_matrix)
    """
    store as :
    [bucket_0 time1, bucket_0 time2, ...., bucket_0 time n]
    [bucket_1 time1, bucket_1 time2, ...., bucket_1 time n]
    """
    data_transpose = np.transpose(data_numpy)

    return forward_time_acc / NUM_STEPS, data_transpose


def _bucket_expected_max(bucket_times, ngpus):
    expected_max_arr = []
    for samples in bucket_times:
        expected_max = _bucket_estimate_max_expected(samples, ngpus)
        expected_max_arr.append(expected_max)

    return expected_max_arr


def _trace_handler(p):
    p.export_chrome_trace(FILENAME)


def run_profiler(model_provider, input_provider, iteration_provider):
    setup(RANK, WORLD_SIZE)

    model = model_provider()
    inputs = input_provider()
    ddp_model = DDP(model, device_ids=[RANK], bucket_cap_mb=DEFAULT_BUCKET_SIZE)
    iteration = iteration_provider(ddp_model)
    # warmup for 30 secs
    start = time.time()
    elapsed = 0

    while elapsed < 30:
        for _ in range(100):
            iteration(*inputs)
        elapsed = time.time() - start

    skip_first = 10
    wait = 5
    warmup = 10
    active = 30
    totalIterations = skip_first + wait + warmup + active
    deepviewSchedule = schedule(
        skip_first=skip_first, wait=wait, warmup=warmup, active=active, repeat=1
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=deepviewSchedule,
        on_trace_ready=_trace_handler,
    ) as p:
        for _ in range(totalIterations):
            iteration(*inputs)
            p.step()


def ddp_analysis(queue, payload, path_to_entry_point_dir):
    sys.path.append(path_to_entry_point_dir)
    try:
        model_provider, input_provider, iteration_provider, logging_level = dill.loads(
            payload
        )

        run_profiler(model_provider, input_provider, iteration_provider)

        path_to_file = os.path.join(os.getcwd(), FILENAME)

        fw_avg_msec, bucket_comp_times = _bucket_comp_times(path_to_file)
        bucket_sizes_arr = get_bucket_sizes(model_provider(), DEFAULT_BUCKET_SIZE)

        expected_max_2gpus = _bucket_expected_max(bucket_comp_times, 2)
        expected_max_4gpus = _bucket_expected_max(bucket_comp_times, 4)

        jsonFormat = {
            "forward_time_ms": fw_avg_msec,
            "bucket_sizes": bucket_sizes_arr,
            "expected_max_2gpus": expected_max_2gpus,
            "expected_max_4gpus": expected_max_4gpus,
        }
        queue.put(jsonFormat)

    except AnalysisError as ex:
        message = str(ex)
        logger.error(message)
        queue.put({"error": message})
    except Exception as ex:
        message = str(ex)
        logger.error(message)
        queue.put({"error": message})

    finally:
        # delete pytorch logs
        subprocess.run(["rm", "-f", os.path.join(os.getcwd(), FILENAME)])
