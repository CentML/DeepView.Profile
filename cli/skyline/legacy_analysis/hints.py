import ast
from skyline.config import Config
from skyline.models.analysis import PerformanceHint


def extract_performance_hints(op_name, call_node, source_map):
    """
    Extracts performance hints associated with the kwargs in PyTorch module
    instantiations.

      e.g. self.linear = torch.nn.Linear(in_features=500, out_features=500)

    """
    if op_name not in Config.Hints:
        return []

    perf_hints = []

    for keyword_node in call_node.keywords:
        if keyword_node.arg not in Config.Hints[op_name]:
            continue

        perf_hint = _process_keyword(
            op_name, call_node, source_map, keyword_node)

        # Skip the keyword if we run into a problem processing it
        if perf_hint is None:
            continue

        perf_hints.append(perf_hint)

    return perf_hints


def _process_keyword(op_name, call_node, source_map, keyword_node):
    # NOTE: Subtract 1 here because AST line numbers are 1-based
    keyword_position = source_map.find_position(
        keyword_node.arg, call_node.lineno - 1)

    if keyword_position is None:
        return None

    # Select a "nice" line:col offset (middle of the keyword)
    refined_position = keyword_position.offset(len(keyword_node.arg) // 2)

    properties = Config.Hints[op_name][keyword_node.arg]
    return PerformanceHint(
        keyword=keyword_node.arg,
        position=refined_position,
        effectiveness=properties['effectiveness'],
        natural_direction=properties['natural_direction'],
    )
