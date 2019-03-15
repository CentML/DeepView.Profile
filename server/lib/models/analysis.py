import lib.models_gen.messages_pb2 as m
from lib.models.source_map import Position


class OperationSourceMap:
    def __init__(self, bound_name, op_name, ast_node, position, perf_hints):
        self.bound_name = bound_name
        self.op_name = op_name
        self.ast_node = ast_node
        self.position = position
        self.perf_hints = perf_hints

    def fill_protobuf(self, info_pb):
        info_pb.bound_name = self.bound_name
        info_pb.op_name = self.op_name
        self.position.fill_protobuf(info_pb.location)
        for hint in self.perf_hints:
            hint_pb = info_pb.hints.add()
            hint.fill_protobuf(hint_pb)


class AnnotationInfo:
    def __init__(self, input_size, start_position, end_position):
        self.input_size = input_size
        self.start_position = start_position
        self.end_position = end_position

    def fill_protobuf(self, annotation_pb):
        for integer in self.input_size:
            annotation_pb.input_size.values.append(integer)
        self.start_position.fill_protobuf(annotation_pb.annotation_start)
        self.end_position.fill_protobuf(annotation_pb.annotation_end)


class PerformanceHint:
    def __init__(self, keyword, position, effectiveness, natural_direction):
        self.keyword = keyword
        self.position = position
        self.effectiveness = effectiveness
        self.natural_direction = natural_direction

    def fill_protobuf(self, perf_hint_pb):
        if self.effectiveness == 'high':
            perf_hint_pb.effectiveness = m.PerformanceHint.HIGH
        else:
            perf_hint_pb.effectiveness = m.PerformanceHint.LOW
        perf_hint_pb.natural_direction = self.natural_direction
        self.position.fill_protobuf(perf_hint_pb.location)
