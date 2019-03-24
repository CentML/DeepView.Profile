

class LRUCache:
    def __init__(self, max_size=128):
        self._max_size = max_size
        self._cache_by_key = {}
        self._cache_by_use = _LRUCacheList()

    def query(self, key):
        if key not in self._cache_by_key:
            return None
        node = self._cache_by_key[key]
        self._cache_by_use.move_to_front(node)
        return node.value

    def add(self, key, value):
        if self._cache_by_use.size >= self._max_size:
            removed = self._cache_by_use.remove_back()
            del self._cache_by_key[removed.key]
        node = self._cache_by_use.add_to_front(key, value)
        self._cache_by_key[key] = node


class _LRUCacheList:
    def __init__(self):
        # Front of the list: most recently used
        self.front = None
        self.back = None
        self.size = 0

    def add_to_front(self, key, value):
        node = _LRUCacheNode(key, value)
        self._add_to_front(node)
        self.size += 1
        return node

    def _add_to_front(self, node):
        if self.size == 0:
            self.front = node
            self.back = node
        else:
            node.next = self.front
            self.front.prev = node
            self.front = node

    def move_to_front(self, node):
        if self.front == node:
            # Nothing needs to be done if the node is already at the front of
            # the list
            return

        if node.next is None:
            # Back of the list
            node.prev.next = None
            self.back = node.prev
            node.prev = None
        else:
            # Middle of the list
            node.prev.next = node.next
            node.next.prev = node.prev
            node.next = None
            node.prev = None

        self._add_to_front(node)

    def remove_back(self):
        if self.size == 0:
            return None

        node = self.back

        if self.size == 1:
            self.front = None
            self.back = None
        else:
            node.prev.next = None
            self.back = node.prev

        self.size -= 1
        return node


class _LRUCacheNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
