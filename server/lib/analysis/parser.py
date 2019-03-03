import ast
import re

from lib.analysis.ast_visitors import (
    PyTorchModuleExtractorVisitor,
    PyTorchFunctionExtractor,
    PyTorchStatementProcessor,
)
from lib.exceptions import AnalysisError
import lib.models_gen.messages_pb2 as m

INPUT_SIZE_REGEX = re.compile(
    '.*@innpv[ \t]+size[ \t]+\((?P<sizes>[0-9]+(,[ \t]*[0-9]+)*)\).*',
)


def analyze_source_code(source_code):
    # 1. Generate the AST associated with the user's code
    try:
        tree = ast.parse(source_code)
    except SyntaxError as ex:
        raise AnalysisError(
            'Syntax error on line {} column {}.'
            .format(ex.lineno, ex.offset)
        ) from ex

    # 2. Find the class definition for the PyTorch module
    extractor_visitor = PyTorchModuleExtractorVisitor()
    extractor_visitor.visit(tree)
    class_node = extractor_visitor.get_class_node()

    # 3. Find the relevant functions
    function_visitor = PyTorchFunctionExtractor()
    function_visitor.visit(class_node)
    functions = function_visitor.get_functions()
    if not '__init__' in functions:
        raise AnalysisError('Missing __init__() function in PyTorch module.')
    if not 'forward' in functions:
        raise AnalysisError('Missing forward() function in PyTorch module.')

    # 4. Extract the input tensor dimension
    input_size = _parse_input_size(ast.get_docstring(functions['forward']))

    # 5. Extract the line numbers of the module parameters
    statement_visitor = PyTorchStatementProcessor()
    statement_visitor.visit(functions['__init__'])
    model_operations = statement_visitor.get_model_operations()

    return (input_size, model_operations)


def _parse_input_size(docstring):
    if docstring is None:
        raise AnalysisError(
            'The forward() function is missing an input size @innpv '
            'annotation.',
        )

    for line in docstring.split('\n'):
        result = INPUT_SIZE_REGEX.match(line)
        if result is None:
            continue

        sizes = result.groupdict()['sizes'].split(',')
        return tuple(int(dimension) for dimension in sizes)

    raise AnalysisError(
        'The forward() function is missing an input size @innpv '
        'annotation or there is a syntax error in the @innpv size '
        'annotation.',
    )


def main():
    import argparse
    import code
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    with open(args.file, 'r') as file:
        lines = [line for line in file]
        input_size, model_operations = analyze_source_code(''.join(lines))

    code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()
