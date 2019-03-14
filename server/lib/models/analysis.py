import lib.models_gen.messages_pb2 as m

class Position:
    def __init__(self, line, column):
        self.line = line
        self.column = column

    def fill_protobuf(self, location_pb):
        location_pb.line = self.line
        location_pb.column = self.column


class OperationSourceMap:
    def __init__(self, bound_name, op_name, ast_node, position):
        self.bound_name = bound_name
        self.op_name = op_name
        self.ast_node = ast_node
        self.position = position

    def fill_protobuf(self, info_pb):
        info_pb.bound_name = self.bound_name
        info_pb.op_name = self.op_name
        self.position.fill_protobuf(info_pb.location)


class AnnotationInfo:
    def __init__(self, input_size, start_position, end_position):
        self.input_size = input_size
        self.start_position = start_position
        self.end_position = end_position

    def fill_protobuf(self, annotation_pb):
        for integer in self.input_size:
            annotation_pb.input_size.values.append(integer)
        self.start_position.fill_protobuf(annotation_pb.annotation_start)
        self.end_position.fill_protobuf(annotation_pb.annotation_start)
