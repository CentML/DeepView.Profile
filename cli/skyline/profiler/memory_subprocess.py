import pickle

from skyline.exceptions import AnalysisError


def measure_memory_usage(source_code, class_name, input_size, batch_sizes):
    """
    Uses a child process to measure the model's peak memory usage, in bytes.

    NOTE: Batch sizes must be in increasing order!
    """
    import subprocess
    result = None

    try:
        process = subprocess.Popen(
            ['python3', '-m', 'lib.profiler.memory_subprocess', class_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        pickle.dump(source_code, process.stdin)
        pickle.dump(input_size, process.stdin)
        pickle.dump(batch_sizes, process.stdin)
        process.stdin.flush()
        result = pickle.load(process.stdout)

    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()

    if result is None:
        raise RuntimeError('Child process memory measurement failed.')
    elif isinstance(result, AnalysisError):
        raise result

    return result


def measure_memory_main():
    # Invoked as a child process. Do not call directly!
    import torch
    import sys

    try:
        class_name = sys.argv[1]
        source_code = pickle.load(sys.stdin.buffer)
        input_size = pickle.load(sys.stdin.buffer)
        batch_sizes = pickle.load(sys.stdin.buffer)

        code = compile(source_code, '<string>', 'exec')
        scope = {}
        exec(code, scope, scope)

        usages_bytes = []
        torch.backends.cudnn.benchmark = True
        model = scope[class_name]().cuda()
        model.train()

        for batch_size in batch_sizes:
            mock_input = torch.randn(
                (batch_size, *input_size[1:]), device=torch.device('cuda'))
            output = model(mock_input)
            fake_grads = torch.ones_like(output)
            output.backward(fake_grads)
            usages_bytes.append(torch.cuda.max_memory_allocated())

        pickle.dump(usages_bytes, sys.stdout.buffer)
        sys.stdout.flush()

    except Exception as ex:
        error = AnalysisError(str(ex), type(ex))
        pickle.dump(error, sys.stdout.buffer)

    finally:
        sys.stdin.close()
        sys.stdout.close()


if __name__ == '__main__':
    measure_memory_main()
