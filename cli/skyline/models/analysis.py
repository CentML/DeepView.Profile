from skyline.models.source_map import Position


class OperationInfo:
    def __init__(self, bound_name, op_name, ast_node, position, perf_hints):
        self.bound_name = bound_name
        self.op_name = op_name
        self.ast_node = ast_node
        self.position = position
        self.perf_hints = perf_hints
        self.usages = []
        self.runtime_us = 0

    def set_usages(self, usages):
        self.usages = usages

    def add_to_runtime_us(self, runtime_us):
        self.runtime_us += runtime_us


class OperationInfoMap:
    def __init__(self):
        self.operations = {}

    def add_operation_info(self, operation):
        self.operations[operation.bound_name] = operation

    def get_operation_info_by_bound_name(self, bound_name):
        if bound_name not in self.operations:
            return None
        return self.operations[bound_name]

    def get_operations(self):
        return self.operations.values()

    def set_runtimes_from_cache(self, cached_info_map):
        """
        Used to set the runtimes from cache for when the parsed code has not
        changed.
        """
        for bound_name, op_info in self.operations.items():
            cached_op_info = cached_info_map.get_operation_info_by_bound_name(
                bound_name)
            op_info.runtime_us = cached_op_info.runtime_us


class AnnotationInfo:
    def __init__(self, input_size, start_position, end_position):
        self.input_size = input_size
        self.start_position = start_position
        self.end_position = end_position


class PerformanceHint:
    def __init__(self, keyword, position, effectiveness, natural_direction):
        self.keyword = keyword
        self.position = position
        self.effectiveness = effectiveness
        self.natural_direction = natural_direction


class LinearModel:
    def __init__(self, coefficient, bias):
        self.coefficient = coefficient
        self.bias = bias

    def __repr__(self):
        return 'LinearModel(coefficient={:.4f}, bias={:.4f})'.format(
            self.coefficient, self.bias)

    def evaluate(self, x):
        return self.coefficient * x + self.bias

    def inverse(self, y):
        return (y - self.bias) / self.coefficient


class MemoryInfo:
    def __init__(self, usage_model_mb, usage_mb, max_capacity_mb):
        self.usage_model_mb = usage_model_mb
        self.usage_mb = usage_mb
        self.max_capacity_mb = max_capacity_mb

    def __repr__(self):
        return 'MemoryInfo(model={}, usage_mb={}, capacity_mb={})'.format(
            self.usage_model_mb, self.usage_mb, self.max_capacity_mb)


class ThroughputInfo:
    def __init__(
        self,
        throughput,
        max_throughput,
        runtime_model_ms
    ):
        self.throughput = throughput
        self.max_throughput = max_throughput
        self.runtime_model_ms = runtime_model_ms

    def __repr__(self):
        return (
            'ThroughputInfo(thpt={}, max_thpt={}, model={})'
            .format(
                self.throughput,
                self.max_throughput,
                self.runtime_model_ms,
            )
        )

    def batch_from_throughput(self, throughput):
        # Thpt = batch / runtime_model
        throughput_ms = throughput / 1000
        return (
            (throughput_ms * self.runtime_model_ms.bias) /
            (1 - throughput_ms * self.runtime_model_ms.coefficient)
        )


class PerformanceLimits:
    def __init__(self, max_batch_size, throughput_limit):
        self.max_batch_size = max_batch_size
        self.throughput_limit = throughput_limit

    def __repr__(self):
        return (
            'PerformanceLimits(max_batch={:.2f}, thpt_limit={:.2f})'.format(
                self.max_batch_size,
                self.throughput_limit,
            )
        )
