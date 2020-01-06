import torch
import logging
from scipy import stats

from skyline.exceptions import AnalysisError
from skyline.models.analysis import MemoryInfo, LinearModel
from skyline.profiler.memory_subprocess import measure_memory_usage

logger = logging.getLogger(__name__)


def get_memory_info(source_code, class_name, annotation_info, nvml):
    try:
        input_size = annotation_info.input_size

        capacity_mb = _to_mb(nvml.get_memory_capacity().total)
        current_usage_mb = _to_mb(measure_memory_usage(
            source_code, class_name, input_size, [input_size[0]])[0])
        usage_model_mb = _get_memory_model(source_code, class_name, input_size)

        return MemoryInfo(usage_model_mb, current_usage_mb, capacity_mb)

    except Exception as ex:
        raise AnalysisError(str(ex), type(ex))


def _get_memory_model(source_code, class_name, input_size):
    # TODO: Select batch sizes for this more intelligently
    batches = [8, 16, 32]
    memory_usage = list(map(
        lambda usage_bytes: _to_mb(usage_bytes),
        measure_memory_usage(source_code, class_name, input_size, batches),
    ))
    slope, intercept, r_value, _, _ = stats.linregress(
        batches, memory_usage)
    logger.debug(
        'Calculated memory usage model - '
        'coefficient: %.4f, intercept: %.4f, r_value = %.4f',
        slope,
        intercept,
        r_value,
    )
    return LinearModel(slope, intercept)


def _get_memory_usage_local(model, batch_size, input_size):
    """
    Returns the current memory usage for a given batch size, in megabytes.

    NOTE: This amount can be less than the peak memory usage.
    """
    mock_input = torch.randn(
        (batch_size, *input_size[1:]),
        device=torch.device('cuda'),
    )
    output = model(mock_input)
    return _to_mb(torch.cuda.memory_allocated())


def _to_mb(num_bytes):
    return num_bytes / 1024 / 1024


def main():
    import argparse
    import code
    from skyline.legacy_analysis.parser import parse_source_code, analyze_code
    from skyline.config import Config
    from skyline.profiler import to_trainable_model
    from skyline.nvml import NVML

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hints-file",
        default="hints.yml",
        help="Path to the performance hints configuration YAML file.",
    )
    parser.add_argument('file')
    args = parser.parse_args()
    Config.initialize_hints_config(args.hints_file)

    with open(args.file, 'r') as file:
        lines = [line for line in file]

    source_code = ''.join(lines)
    tree, source_map = parse_source_code(source_code)
    class_name, annotation_info, _ = analyze_code(tree, source_map)
    model = to_trainable_model(tree, class_name)

    with NVML() as nvml:
        memory_info = get_memory_info(
            source_code, class_name, annotation_info, nvml)
        print(memory_info)
        code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()
