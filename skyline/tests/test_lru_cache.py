import unittest

import skyline.lru_cache as lru


class LRUCacheTests(unittest.TestCase):
    def test_cache_empty(self):
        cache = lru.LRUCache()
        self.assertIsNone(cache.query(123))
        self.assertEqual(len(cache._cache_by_key), 0)
        self.assertEqual(cache._cache_by_use.size, 0)

    def test_cache_one(self):
        cache = lru.LRUCache(max_size=1)
        self.assertIsNone(cache.query(1))
        cache.add(1, 1)
        self.assertEqual(cache.query(1), 1)
        cache.add(2, 2)
        self.assertIsNone(cache.query(1))
        self.assertEqual(cache.query(2), 2)
        self.assertEqual(len(cache._cache_by_key), 1)
        self.assertEqual(cache._cache_by_use.size, 1)

    def test_cache_two(self):
        cache = lru.LRUCache(max_size=2)
        cache.add(1, 1)
        cache.add(2, 2)
        self.assertEqual(cache.query(2), 2)
        cache.add(3, 3)
        self.assertEqual(cache.query(2), 2)
        cache.add(5, 5)
        self.assertIsNone(cache.query(3))
        self.assertEqual(len(cache._cache_by_key), 2)
        self.assertEqual(cache._cache_by_use.size, 2)

    def test_cache_three(self):
        cache = lru.LRUCache(max_size=3)
        self.assertIsNone(cache.query(1))
        cache.add(1, 1)
        cache.add(2, 2)
        cache.add(3, 3)
        self.assertEqual(cache.query(1), 1)
        self.assertEqual(cache.query(3), 3)
        self.assertEqual(cache.query(2), 2)
        cache.add(4, 4)
        self.assertIsNone(cache.query(1))
        self.assertEqual(cache.query(4), 4)
        self.assertEqual(cache.query(3), 3)
        self.assertEqual(cache.query(2), 2)
        cache.add(5, 5)
        self.assertIsNone(cache.query(4))
        self.assertEqual(len(cache._cache_by_key), 3)
        self.assertEqual(cache._cache_by_use.size, 3)

    def test_cache_cyclical(self):
        cache = lru.LRUCache(max_size=3)
        cache.add(1, 1)
        cache.add(2, 2)
        cache.add(3, 3)
        cache.add(4, 4)
        cache.add(5, 5)
        cache.add(6, 6)
        self.assertIsNone(cache.query(1))
        self.assertIsNone(cache.query(2))
        self.assertIsNone(cache.query(3))
        self.assertEqual(cache.query(4), 4)
        self.assertEqual(cache.query(5), 5)
        self.assertEqual(cache.query(6), 6)
        self.assertEqual(len(cache._cache_by_key), 3)
        self.assertEqual(cache._cache_by_use.size, 3)


class LRUCacheListTests(unittest.TestCase):
    def setUp(self):
        self.list = lru._LRUCacheList()

    def list_to_array(self, backward=False):
        items = []
        ptr = self.list.front if not backward else self.list.back
        while ptr is not None:
            items.append((ptr.key, ptr.value))
            ptr = ptr.next if not backward else ptr.prev
        return items

    def test_list_none(self):
        self.assertIsNone(self.list.front)
        self.assertIsNone(self.list.back)
        self.assertEqual(self.list.size, 0)
        node = self.list.add_to_front(1, 1)
        self.assertIsNone(node.next)
        self.assertIsNone(node.prev)
        self.assertEqual(self.list.size, 1)

    def test_list_add(self):
        self.list.add_to_front('hello', 'world')
        self.assertEqual(self.list.size, 1)
        self.assertEqual(self.list.front, self.list.back)
        self.assertIsNone(self.list.front.next)
        self.assertIsNone(self.list.front.prev)
        self.assertEqual(self.list.front.key, 'hello')
        self.assertEqual(self.list.front.value, 'world')

        self.list.add_to_front('hello2', 'world2')
        self.assertEqual(self.list.size, 2)
        self.assertEqual(
            self.list_to_array(backward=True),
            [('hello', 'world'), ('hello2', 'world2')],
        )

    def test_list_add_several(self):
        self.list.add_to_front(1, 1)
        self.list.add_to_front(2, 2)
        self.list.add_to_front(3, 3)
        self.assertEqual(self.list.size, 3)
        self.assertNotEqual(self.list.front, self.list.back)
        self.assertEqual(self.list_to_array(), [(3, 3), (2, 2), (1, 1)])

    def test_list_move_three(self):
        n1 = self.list.add_to_front(1, 1)
        n2 = self.list.add_to_front(2, 2)
        n3 = self.list.add_to_front(3, 3)

        self.list.move_to_front(n1)
        self.assertEqual(self.list_to_array(), [(1, 1), (3, 3), (2, 2)])
        self.assertEqual(
            self.list_to_array(backward=True), [(2, 2), (3, 3), (1, 1)])
        self.assertEqual(self.list.size, 3)

        self.list.move_to_front(n2)
        self.assertEqual(self.list_to_array(), [(2, 2), (1, 1), (3, 3)])
        self.assertEqual(
            self.list_to_array(backward=True), [(3, 3), (1, 1), (2, 2)])
        self.assertEqual(self.list.size, 3)

        self.list.move_to_front(n2)
        self.assertEqual(self.list_to_array(), [(2, 2), (1, 1), (3, 3)])
        self.assertEqual(
            self.list_to_array(backward=True), [(3, 3), (1, 1), (2, 2)])
        self.assertEqual(self.list.size, 3)

    def test_list_move_two(self):
        n1 = self.list.add_to_front(1, 1)
        n2 = self.list.add_to_front(2, 2)
        self.assertEqual(self.list_to_array(), [(2, 2), (1, 1)])

        self.list.move_to_front(n1)
        self.assertEqual(self.list_to_array(), [(1, 1), (2, 2)])

    def test_list_move_one(self):
        n1 = self.list.add_to_front(1, 1)
        self.assertEqual(self.list_to_array(), [(1, 1)])
        self.list.move_to_front(n1)
        self.assertEqual(self.list_to_array(), [(1, 1)])
        self.assertEqual(self.list_to_array(backward=True), [(1, 1)])
        self.assertEqual(self.list.size, 1)

    def test_list_remove_back_empty(self):
        removed = self.list.remove_back()
        self.assertEqual(self.list.size, 0)
        self.assertIsNone(removed)
        self.assertEqual(self.list_to_array(), [])

    def test_list_remove_back_one(self):
        n1 = self.list.add_to_front(1, 1)
        self.assertEqual(self.list_to_array(), [(1, 1)])
        removed = self.list.remove_back()
        self.assertEqual(n1, removed)
        self.assertEqual(self.list.size, 0)
        self.assertEqual(self.list_to_array(), [])

    def test_list_remove_back_two(self):
        n1 = self.list.add_to_front(1, 1)
        n2 = self.list.add_to_front(2, 2)
        self.assertEqual(self.list.size, 2)
        self.assertEqual(self.list_to_array(), [(2, 2), (1, 1)])
        removed = self.list.remove_back()
        self.assertEqual(removed, n1)
        self.assertEqual(self.list_to_array(), [(2, 2)])
        self.assertEqual(self.list_to_array(backward=True), [(2, 2)])
        self.assertEqual(self.list.size, 1)

    def test_list_remove_back_three(self):
        n1 = self.list.add_to_front(1, 1)
        n2 = self.list.add_to_front(2, 2)
        n3 = self.list.add_to_front(3, 3)
        self.assertEqual(self.list.size, 3)
        self.assertEqual(self.list_to_array(), [(3, 3), (2, 2), (1, 1)])
        removed = self.list.remove_back()
        self.assertEqual(removed, n1)
        self.assertEqual(self.list_to_array(), [(3, 3), (2, 2)])
        self.assertEqual(self.list_to_array(backward=True), [(2, 2), (3, 3)])
        self.assertEqual(self.list.size, 2)
