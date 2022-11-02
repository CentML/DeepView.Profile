import argparse
import e2e_utils
import os
import pickle
import torch.cuda as cuda
import torch.utils.benchmark as benchmark
from typing import List, Tuple, Callable
import skyline.analysis.session as session
import sys


class BaselineMetricsRunner:
    def __init__(self, model_name, entry_point):
        self.model_name = model_name
        self._entry_point = entry_point
        # get the original directory so we can restore it after get
        original_dir = os.getcwd()
        self._model_root_dir = os.path.dirname(entry_point)
        self._entry_filename = os.path.basename(entry_point)
        # Change to the model's root directory since this runner is only for this model
        # This must be done before _process_and_validate_entry_point to be able to process the entry point properly
        os.chdir(self._model_root_dir)
        self._given_batch_size, self._model_provider, self._input_provider, self._iteration_provider = self._process_and_validate_entry_point()
        self._batch_sizes = self._get_batch_sizes()
        self._baseline_metric_list = self._get_baseline_metrics()
        os.chdir(original_dir)

    def _process_and_validate_entry_point(self) -> Tuple[int, Callable, Callable, Callable]:
        # Note: This is not necessarily the same as project_root because the
        # entry_point could be in a subdirectory.
        path_to_entry_point = os.path.join(self._model_root_dir, self._entry_filename)
        path_to_entry_point_dir = os.path.dirname(path_to_entry_point)

        _, __, scope = session._run_entry_point(path_to_entry_point, path_to_entry_point_dir, self._model_root_dir)
        
        # Check that the model provider and input provider functions exist
        if session.MODEL_PROVIDER_NAME not in scope:
            raise session.AnalysisError(
                "The project entry point file is missing a model provider "
                "function. Please add a model provider function named "
                "\"{}\".".format(session.MODEL_PROVIDER_NAME)
            ).with_file_context(self._entry_point)

        if session.INPUT_PROVIDER_NAME not in scope:
            raise session.AnalysisError(
                "The project entry point file is missing an input provider "
                "function. Please add an input provider function named "
                "\"{}\".".format(session.INPUT_PROVIDER_NAME)
            ).with_file_context(self._entry_point)

        if session.ITERATION_PROVIDER_NAME not in scope:
            raise session.AnalysisError(
                "The project entry point file is missing an iteration "
                "provider function. Please add an iteration provider function "
                "named \"{}\".".format(session.ITERATION_PROVIDER_NAME)
            ).with_file_context(self._entry_point)

        given_batch_size = session._validate_providers_signatures(
            scope[session.MODEL_PROVIDER_NAME],
            scope[session.INPUT_PROVIDER_NAME],
            scope[session.ITERATION_PROVIDER_NAME],
            self._entry_point,
        )

        model_provider, input_provider, iteration_provider = (
            session._wrap_providers_with_validators(
                scope[session.MODEL_PROVIDER_NAME],
                scope[session.INPUT_PROVIDER_NAME],
                scope[session.ITERATION_PROVIDER_NAME],
                self._entry_point,
            )
        )
        return (given_batch_size, model_provider, input_provider, iteration_provider)

    def _get_batch_sizes(self) -> List[int]:
        cuda.empty_cache()
        cuda.reset_max_memory_allocated()
        timer = benchmark.Timer(
            stmt='iterator(*inputs)',
            globals={
                'inputs': self._input_provider(batch_size=self._given_batch_size),
                'iterator': self._iteration_provider(self._model_provider())
            }
        )
        timer.timeit(100)
        peak_memory_usage = cuda.max_memory_allocated()
        cuda.empty_cache()
        max_memory = cuda.mem_get_info()[1]
        max_batch_size = self._given_batch_size*max_memory/peak_memory_usage
        # TODO mshin: figure out a way to dynamically modify the divisors of the values
        return [int(max_batch_size/12), int(max_batch_size/6), int(max_batch_size/3), int(max_batch_size/2)]

    def _get_baseline_metrics(self) -> List[e2e_utils.ProfilingMetrics]:
        profiling_metrics_list = []
        for iter_batchsize in self._batch_sizes:
            cuda.empty_cache()
            cuda.reset_max_memory_allocated()

            timer = benchmark.Timer(
                stmt='iterator(*inputs)',
                globals={
                    'inputs': self._input_provider(batch_size=iter_batchsize),
                    'iterator': self._iteration_provider(self._model_provider())
                }
            )
            measurement = timer.timeit(100)
            throughput = float(iter_batchsize)/measurement.mean
            peak_memory_usage = cuda.max_memory_allocated()
            profiling_metrics_list.append(e2e_utils.ProfilingMetrics(batch_size=iter_batchsize, samples_per_second=throughput, peak_usage_bytes=peak_memory_usage))
        cuda.empty_cache()
        return profiling_metrics_list
    
    def save_to_file(self):
        file_name = self.model_name + '.pkl'
        with open(file_name, 'wb') as f:
            baseline_metrics = e2e_utils.BaselineMetrics(self.model_name, self._entry_point, self._given_batch_size, self._baseline_metric_list)    
            pickle.dump(baseline_metrics, f, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(
        prog="baseline_test_runner",
        description="Baseline test runner: Gets profiling metrics for a given metric."
                    "Used to compare Skyline's profiling results",
    )
    parser.add_argument(
        "name",
        help="Name of the model that we are profiling. Used for naming the export file",
    )
    parser.add_argument(
        "entry_point",
        
        help="The entry point file of the given model. This must be given in absolute file path"
    )
    try:
        args = parser.parse_args()
        baseline_runner = BaselineMetricsRunner(args.name, args.entry_point)
        baseline_runner.save_to_file()
    except SystemError as s:
        print(s)
        sys.exit(2)


if __name__ == "__main__":
    main()