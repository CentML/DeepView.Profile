import ast

from lib.exceptions import AnalysisError
from lib.models.analysis import OperationSourceMap, Position


class PyTorchModuleExtractorVisitor(ast.NodeVisitor):
    def __init__(self):
        self.class_nodes = []

    def visit_ClassDef(self, node):
        if len(node.bases) == 0:
            return
        # TODO: Actually check that the class is a subclass of torch.nn.Module
        #       The check should be a runtime check, not a static one.
        self.class_nodes.append(node)

    def get_class_node(self):
        if len(self.class_nodes) == 0:
            raise AnalysisError('A PyTorch module was not found.')
        if len(self.class_nodes) > 1:
            raise AnalysisError(
                'Multiple PyTorch modules were found. INNPV only supports '
                'source files with a single PyTorch module.'
            )
        return self.class_nodes[0]


class PyTorchFunctionExtractor(ast.NodeVisitor):
    def __init__(self):
        self.relevant_functions = {}

    def visit_FunctionDef(self, node):
        if node.name != '__init__' and node.name != 'forward':
            return
        self.relevant_functions[node.name] = node

    def get_functions(self):
        return self.relevant_functions


class PyTorchStatementProcessor(ast.NodeVisitor):
    def __init__(self):
        self.model_operations = []
        self.module_param_extractor = PyTorchModuleParameterExtractor()

    def get_model_operations(self):
        return self.model_operations

    def visit_Assign(self, node):
        if not self._is_instance_assignment(node.targets):
            return

        op_name, op_node = self.module_param_extractor.visit(node.value)
        if op_name is None or op_node is None:
            return

        # We subtract 1 from the line numbers to make them 0-based
        self.model_operations.append(OperationSourceMap(
            bound_name=node.targets[0].attr,
            op_name=op_name,
            ast_node=node,
            position=Position(node.value.lineno - 1, node.value.col_offset),
        ))

    def _is_instance_assignment(self, targets):
        # Check to make sure the model parameter is being assigned to an
        # instance variable (i.e. it is of the form "self.<var>")
        if len(targets) != 1:
            return False
        node = targets[0]
        if not isinstance(node, ast.Attribute):
            return False
        if not isinstance(node.value, ast.Name):
            return False
        return node.value.id == 'self'


class PyTorchModuleParameterExtractor(ast.NodeVisitor):
    def visit_Call(self, node):
        # We look for module instantiations prefixed with "torch.nn." or "nn."
        # to allow us to disambiguate between user defined names and PyTorch
        # modules.
        if not isinstance(node.func, ast.Attribute):
            return (None, None)
        if not self._is_properly_prefixed(node.func.value):
            return (None, None)
        return (node.func.attr, node)

    def _is_properly_prefixed(self, node):
        path = self.visit(node)
        if len(path) > 2:
            return False
        if len(path) == 2 and path[0] != 'torch':
            return False
        return path[-1] == 'nn'

    def visit_Attribute(self, node):
        path = self.visit(node.value)
        path.append(node.attr)
        return path

    def visit_Name(self, node):
        return [node.id]
