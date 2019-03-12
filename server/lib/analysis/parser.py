import ast
import re

from lib.analysis.ast_visitors import (
    PyTorchModuleExtractorVisitor,
    PyTorchFunctionExtractor,
    PyTorchStatementProcessor,
)
from lib.exceptions import AnalysisError
from lib.models.analysis import AnnotationInfo
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

    # 4. Parse the annotation, extract the input size and other source metadata
    input_size, annotation = _parse_annotation(
        ast.get_docstring(functions['forward']),
    )
    line, column = _get_annotation_source_location(
        source_code,
        annotation,
        functions['forward'],
    )
    annotation_info = AnnotationInfo(input_size, line, column)

    # 5. Extract the line numbers of the module parameters
    statement_visitor = PyTorchStatementProcessor()
    statement_visitor.visit(functions['__init__'])
    model_operations = statement_visitor.get_model_operations()

    return (annotation_info, model_operations)


def _parse_annotation(docstring):
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
        return tuple(int(dimension) for dimension in sizes), line

    raise AnalysisError(
        'The forward() function is missing an input size @innpv '
        'annotation or there is a syntax error in the @innpv size '
        'annotation.',
    )


def _get_annotation_source_location(source_code, annotation, function_node):
    # TODO: Make this work cross-platform
    code_by_line = source_code.split('\n')
    function_lineno = function_node.lineno

    # NOTE: AST line numbers are 1-based
    for offset, line in enumerate(code_by_line[function_lineno:]):
        index = line.find(annotation)
        if index == -1:
            continue
        # Add 1 to use the same convention as AST line numbers (1-based)
        return (function_lineno + offset + 1, index)

    raise AssertionError('Could not find the INNPV annotation\'s line number.')


def main():
    import argparse
    import code
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    with open(args.file, 'r') as file:
        lines = [line for line in file]
        annotation_info, model_operations = analyze_source_code(''.join(lines))

    code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()
