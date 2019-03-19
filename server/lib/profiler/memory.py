import torch
import logging
from scipy import stats

from lib.models.analysis import MemoryInfo, LinearModel
from lib.exceptions import AnalysisError

logger = logging.getLogger(__name__)


def get_memory_info(model, annotation_info, nvml):
    try:
        input_size = annotation_info.input_size

        capacity_mb = _to_mb(nvml.get_memory_capacity().total)
        current_usage_mb = _get_memory_usage(model, input_size[0], input_size)
        usage_model_mb = _get_memory_model(model, input_size)

        return MemoryInfo(usage_model_mb, current_usage_mb, capacity_mb)

    except Exception as ex:
        raise AnalysisError(str(ex), type(ex))


def _get_memory_model(model, input_size):
    # TODO: Select batch sizes for this more intelligently
    batches = [32, 64, 128]
    memory_usage = list(map(
        lambda batch_size: _get_memory_usage(model, batch_size, input_size),
        batches,
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


def _get_memory_usage(model, batch_size, input_size):
    """
    Returns the peak memory usage for a given batch size, in megabytes.

    NOTE: The peak memory usage occurs at the end of the forward pass
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
    from lib.analysis.parser import parse_source_code, analyze_code
    from lib.config import Config
    from lib.profiler import to_trainable_model
    from lib.nvml import NVML

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

    tree, source_map = parse_source_code(''.join(lines))
    class_name, annotation_info, _ = analyze_code(tree, source_map)
    model = to_trainable_model(tree, class_name)

    with NVML() as nvml:
        memory_info = get_memory_info(model, annotation_info, nvml)
        print(memory_info)
        code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()
