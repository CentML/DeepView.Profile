import ast
import re

from skyline.legacy_analysis.ast_visitors import (
    PyTorchModuleExtractorVisitor,
    PyTorchFunctionExtractor,
    PyTorchStatementProcessor,
    PyTorchModuleUsagesExtractor,
)
from skyline.exceptions import AnalysisError
from skyline.models.analysis import AnnotationInfo, Position
from skyline.models.source_map import SourceMap

INPUT_SIZE_REGEX = re.compile(
    '.*@innpv[ \t]+size[ \t]+\((?P<sizes>[0-9]+(,[ \t]*[0-9]+)*)\).*',
)


def parse_source_code(source_code):
    try:
        tree = ast.parse(source_code)
    except SyntaxError as ex:
        raise AnalysisError(
            'Syntax error on line {} column {}.'
            .format(ex.lineno, ex.offset)
        ) from ex
    source_map = SourceMap(source_code)

    return (tree, source_map)


def analyze_code(tree, source_map):
    # 1. Find the class definition for the PyTorch module
    extractor_visitor = PyTorchModuleExtractorVisitor()
    extractor_visitor.visit(tree)
    class_node = extractor_visitor.get_class_node()

    # 2. Find the relevant functions
    function_visitor = PyTorchFunctionExtractor()
    function_visitor.visit(class_node)
    functions = function_visitor.get_functions()
    if not '__init__' in functions:
        raise AnalysisError('Missing __init__() function in PyTorch module.')
    if not 'forward' in functions:
        raise AnalysisError('Missing forward() function in PyTorch module.')

    # 3. Parse the annotation, extract the input size and other source metadata
    input_size, annotation = _parse_annotation(
        ast.get_docstring(functions['forward']),
    )
    annotation_start = source_map.find_position(
        annotation, functions['forward'].lineno)

    if annotation_start is None:
        raise AssertionError(
            'Could not find the INNPV annotation\'s line number.')

    # NOTE: Ranges in Atom are end-exclusive
    annotation_info = AnnotationInfo(
        input_size,
        annotation_start,
        annotation_start.offset(len(annotation)),
    )

    # 4. Extract the line numbers of the module parameters
    statement_visitor = PyTorchStatementProcessor(source_map)
    statement_visitor.visit(functions['__init__'])
    model_operations = statement_visitor.get_model_operations()

    # 5. Extract module usages from the forward() method
    module_names = set(map(
        lambda op_info: op_info.bound_name,
        model_operations.get_operations(),
    ))
    usages_extractor = PyTorchModuleUsagesExtractor(module_names)
    usages_extractor.visit(functions['forward'])
    module_usages = usages_extractor.get_usages()

    for model_op in model_operations.get_operations():
        if model_op.bound_name not in module_usages:
            continue
        model_op.set_usages(module_usages[model_op.bound_name])

    return (class_node.name, annotation_info, model_operations)


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


def main():
    import argparse
    import code
    from skyline.config import Config

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
    class_name, annotation_info, model_operations = analyze_code(
        *parse_source_code(''.join(lines)))

    code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()
