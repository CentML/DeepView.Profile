import lib.models_gen.messages_pb2 as m


class OperationSourceMap:
    def __init__(self, bound_name, op_name, ast_node, line, column):
        self.bound_name = bound_name
        self.op_name = op_name
        self.ast_node = ast_node
        self.line = line
        self.column = column

    def fill_protobuf(self, info_pb):
        info_pb.bound_name = self.bound_name
        info_pb.op_name = self.op_name
        info_pb.location.line = self.line
        info_pb.location.column = self.column


class AnnotationInfo:
    def __init__(self, input_size, line, column):
        self.input_size = input_size
        self.line = line
        self.column = column
