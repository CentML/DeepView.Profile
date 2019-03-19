import torch
import logging
from scipy import stats

from lib.config import Config
from lib.models.analysis import ThroughputInfo, LinearModel
from lib.exceptions import AnalysisError

logger = logging.getLogger(__name__)


def get_throughput_info(model, annotation_info, memory_info):
    try:
        torch.backends.cudnn.benchmark = True
        logger.debug('cuDNN benchmarking enabled.')

        input_size = annotation_info.input_size
        runtime_ms = _measure_runtime(model, input_size[0], input_size)
        runtime_model_ms = _get_runtime_model(model, input_size)

        throughput = input_size[0] / runtime_ms * 1000
        # NOTE: Reduce the theoretical max by 10% to account for potential
        #       modelling errors
        max_throughput = (
            1.0 / runtime_model_ms.coefficient * 1000 * 0.9
        )

        max_capacity_batch_size = memory_info.usage_model_mb.inverse(
            memory_info.max_capacity_mb)
        max_capacity_throughput = (
            max_capacity_batch_size /
            runtime_model_ms.evaluate(max_capacity_batch_size) *
            1000
        )
        # Indicates whether the maximum attainable throughput is limited by
        # memory capacity or not
        throughput_limit = min(max_throughput, max_capacity_throughput)

        return ThroughputInfo(
            throughput, max_throughput, throughput_limit, runtime_model_ms)

    except Exception as ex:
        raise AnalysisError(str(ex), type(ex))
    finally:
        torch.backends.cudnn.benchmark = False
        logger.debug('cuDNN benchmarking disabled.')


def _get_runtime_model(model, input_size):
    # TODO: Select batch sizes for this more intelligently
    batches = [32, 64, 128]
    runtimes_ms = list(map(
        lambda batch_size: _measure_runtime(model, batch_size, input_size),
        batches,
    ))
    slope, intercept, r_value, _, _ = stats.linregress(
        batches, runtimes_ms)
    logger.debug(
        'Calculated runtime model (ms) - '
        'coefficient: %.4f, intercept: %.4f, r_value = %.4f',
        slope,
        intercept,
        r_value,
    )
    return LinearModel(slope, intercept)


def _measure_runtime(model, batch_size, input_size):
    """
    Returns the training iteration runtime for the given model in milliseconds.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    mock_input = torch.randn(
        (batch_size, *input_size[1:]),
        device=torch.device('cuda'),
    )
    output = model(mock_input)
    fake_grad = torch.ones_like(output)

    def iteration():
        output = model(mock_input)
        output.backward(fake_grad)

    # Warmup
    for _ in range(Config.warm_up):
        iteration()

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(Config.measure_for):
        iteration()

    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / Config.warm_up


def main():
    import argparse
    import code
    from lib.analysis.parser import parse_source_code, analyze_code
    from lib.profiler import to_trainable_model
    from lib.profiler.memory import get_memory_info
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
    throughput_info = get_throughput_info(model, annotation_info, memory_info)
    print(memory_info)
    print(throughput_info)
    code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()
