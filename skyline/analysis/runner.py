import argparse
import logging
import os

from skyline.analysis.session import AnalysisSession
from skyline.nvml import NVML


def analyze_project(project_root, entry_point, nvml):
    session = AnalysisSession.new_from(project_root, entry_point)
    yield session.measure_breakdown(nvml)
    yield session.measure_throughput()


def main():
    # This is used for development and debugging purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("entry_point", type=str)
    args = parser.parse_args()

    project_root = os.getcwd()
    with NVML() as nvml:
        analyzer = analyze_project(project_root, args.entry_point, nvml)
        breakdown = next(analyzer)
        throughput = next(analyzer)

    print('Peak usage:   ', breakdown.peak_usage_bytes, 'bytes')
    print('Max. capacity:', breakdown.memory_capacity_bytes, 'bytes')
    print('No. of weight breakdown nodes:   ', len(breakdown.operation_tree))
    print('No. of operation breakdown nodes:', len(breakdown.weight_tree))
    print('Throughput:', throughput.samples_per_second, 'samples/s')


if __name__ == "__main__":
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.DEBUG,
    }
    logging.basicConfig(**kwargs)
    main()
