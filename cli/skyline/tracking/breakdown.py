

class HierarchicalBreakdownBuilder:
    def __init__(self):
        self._module_names_by_id = {}
        self._root = BreakdownEntry('root', 0)

    def for_model(self, model):
        for name, module in model.named_modules():
            self._module_names_by_id[id(module)] = name
        return self

    def process_tracker(self, tracker):
        tracker.populate_breakdown(self)
        return self

    def add_run_time_entry(
            self, operation_name, forward_ms, backward_ms, stack_context):
        if len(stack_context) == 0:
            raise ValueError(
                'Adding run time entry with no context to the breakdown.')

        for entry in self._traverse_and_insert(operation_name, stack_context):
            entry.add_run_time(forward_ms, backward_ms)

        return self

    def add_activation_entry(self, operation_name, size_bytes, stack_context):
        if len(stack_context) == 0:
            raise ValueError(
                'Adding activation entry with no context to the breakdown.')

        for entry in self._traverse_and_insert(operation_name, stack_context):
            entry.add_activation_size(size_bytes)

        return self

    def build(self):
        self._prune_tree()
        return self._root

    def _traverse_and_insert(self, operation_name, stack_context):
        """
        A generator that, given a list of relevant stack frames, traverses (and
        inserts entries, if needed) the hierarchical breakdown tree, yielding
        each node along its path.
        """
        parent = self._root

        for idx, frame in enumerate(reversed(stack_context)):
            is_last_frame = idx == len(stack_context) - 1
            context = (frame.file_path, frame.line_number)

            if context not in parent.children:
                name = (
                    operation_name if is_last_frame
                    else self._module_names_by_id[frame.module_id]
                )
                new_entry = BreakdownEntry(name, frame.module_id)
                new_entry.add_context(context)
                parent.children[context] = new_entry

            yield parent
            parent = parent.children[context]

        yield parent

    def _prune_tree(self):
        # Current node, key to node from parent, parent node
        stack = [(self._root, None, None)]

        # Depth first traversal. Prune interior nodes with only one child:
        #   e.g. ... -> parent -> node -> child -> ... becomes
        #        ... -> parent -> child -> ...
        while len(stack) != 0:
            node, key, parent = stack.pop()

            if len(node.children) == 1 and parent is not None:
                # Remove "node" from the tree and have the parent
                # point directly to node's only child
                child = next(node.children.values())
                child.add_context(key)
                parent.children[key] = child
                node.children.clear()
                stack.append((child, key, parent))
            else:
                for key_to_child, child in node.children.items():
                    stack.append((child, key_to_child, node))


class BreakdownEntry:
    def __init__(self, name, module_id):
        self._name = name
        self._module_id = module_id

        self._children = {}
        self._contexts = set()

        self._forward_ms = 0.
        self._backward_ms = None
        self._size_bytes = 0

    def add_run_time(self, forward_ms, backward_ms):
        self._forward_ms += forward_ms

        if backward_ms is None:
            return
        if self._backward_ms is None:
            self._backward_ms = 0.
        self._backward_ms += backward_ms

    def add_activation_size(self, size_bytes):
        self._size_bytes += size_bytes

    def add_context(self, context):
        self._contexts.add(context)

    @property
    def name(self):
        return self._name

    @property
    def forward_ms(self):
        return self._forward_ms

    @property
    def backward_ms(self):
        return self._backward_ms

    @property
    def size_bytes(self):
        return self._size_bytes

    @property
    def children(self):
        return self._children
