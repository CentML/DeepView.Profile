

class OperationSourceMap:
    def __init__(self, bound_name, op_name, ast_node, line, column):
        self.bound_name = bound_name
        self.op_name = op_name
        self.ast_node = ast_node
        self.line = line
        self.column = column
