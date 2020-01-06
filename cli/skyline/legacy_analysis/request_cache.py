import ast
from skyline.lru_cache import LRUCache


class SourceCache:
    """
    Used to cache analysis results at the source-level. This helps prevent
    re-analysis for insignificant changes (e.g. whitespace changes, comments).
    """
    def __init__(self):
        self._cache = LRUCache(max_size=128)

    def query(self, parse_tree):
        return self._cache.query(ast.dump(parse_tree))

    def store(self, parse_tree, contents):
        self._cache.add(ast.dump(parse_tree), contents)


class RuntimeCache:
    """
    Used to cache the runtime profiling results for individual modules.
    """
    def __init__(self):
        self._cache = LRUCache(max_size=1024)

    def query(self, module, inputs):
        cache_key = self._get_cache_key(module, inputs)
        return self._cache.query(cache_key)

    def store(self, module, inputs, runtime_us):
        cache_key = self._get_cache_key(module, inputs)
        self._cache.add(cache_key, runtime_us)

    def _get_cache_key(self, module, inputs):
        return '{}_{}'.format(repr(module), self._inputs_to_key(inputs))

    def _inputs_to_key(self, inputs):
        tuples_as_strings = [
            str(tuple(input_tensor.size())) for input_tensor in inputs
        ]
        return '_'.join(tuples_as_strings)
