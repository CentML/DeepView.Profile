import argparse
import logging
import os

from skyline.analysis.session import AnalysisSession
from skyline.nvml import NVML


def analyze_project(project_root, entry_point, nvml):
    session = AnalysisSession.new_from(project_root, entry_point)
    yield session.measure_run_time_breakdown()
    yield session.measure_memory_usage(nvml)
    yield session.measure_throughput()


def main():
    # This is used for development and debugging purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("entry_point", type=str)
    args = parser.parse_args()

    project_root = os.getcwd()
    with NVML() as nvml:
        analyzer = analyze_project(project_root, args.entry_point, nvml)
        run_time = next(analyzer)
        memory = next(analyzer)
        throughput = next(analyzer)

    print('Peak usage:   ', memory.peak_usage_bytes, 'bytes')
    print('Max. capacity:', memory.memory_capacity_bytes, 'bytes')
    print('No. of weight entries:', len(memory.weight_entries))
    print('No. of activ. entries:', len(memory.activation_entries))
    print('Throughput:', throughput.samples_per_second, 'samples/s')


if __name__ == "__main__":
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.DEBUG,
    }
    logging.basicConfig(**kwargs)
    main()
