import ast


class SourceCache:
    """
    Used to cache analysis results at the source-level. This helps prevent
    re-analysis for insignificant changes (e.g. whitespace changes, comments).
    """
    def __init__(self):
        self._cached_source = None
        self._cached_contents = None

    def query(self, parse_tree):
        if self._cached_source is None:
            return None
        if ast.dump(parse_tree) != self._cached_source:
            return None
        return self._cached_contents

    def store(self, parse_tree, contents):
        self._cached_source = ast.dump(parse_tree)
        self._cached_contents = contents


class RuntimeCache:
    """
    Used to cache the runtime profiling results for individual modules.
    """
    def __init__(self):
        self._cache = {}

    def query(self, module, inputs):
        cache_key = self._get_cache_key(module, inputs)
        if cache_key not in self._cache:
            return None
        return self._cache[cache_key]

    def store(self, module, inputs, runtime_us):
        cache_key = self._get_cache_key(module, inputs)
        self._cache[cache_key] = runtime_us

    def _get_cache_key(self, module, inputs):
        return '{}_{}'.format(repr(module), self._inputs_to_key(inputs))

    def _inputs_to_key(self, inputs):
        tuples_as_strings = [
            str(tuple(input_tensor.size())) for input_tensor in inputs
        ]
        return '_'.join(tuples_as_strings)
