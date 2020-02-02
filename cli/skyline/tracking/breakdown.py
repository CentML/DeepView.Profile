import collections
import math
import os

HierarchicalBreakdown = collections.namedtuple(
    'HierarchicalBreakdown', ['operations', 'weights', 'peak_usage_bytes'])


class HierarchicalBreakdownBuilder:
    def __init__(self):
        self._module_names_by_id = {}
        self._operation_root = OperationNode('root', -1)
        self._weight_root = WeightNode('root', -1)
        self._peak_usage_bytes = None

    def for_model(self, model):
        for name, module in model.named_modules():
            self._module_names_by_id[id(module)] = name
        return self

    def process_tracker(self, tracker):
        tracker.populate_breakdown(self)
        return self

    def add_run_time_entry(
            self, operation_name, forward_ms, backward_ms, stack_context):
        if len(stack_context.frames) == 0:
            raise ValueError(
                'Adding run time entry with no context to the breakdown.')

        for entry in self._traverse_and_insert(
                self._operation_root, operation_name, stack_context):
            entry.add_run_time(forward_ms, backward_ms)

        return self

    def add_activation_entry(self, operation_name, size_bytes, stack_context):
        if len(stack_context.frames) == 0:
            raise ValueError(
                'Adding activation entry with no context to the breakdown.')

        for entry in self._traverse_and_insert(
                self._operation_root, operation_name, stack_context):
            entry.add_activation_size(size_bytes)

        return self

    def add_weight_entry(
            self, weight_name, size_bytes, grad_size_bytes, stack_context):
        if len(stack_context.frames) == 0:
            raise ValueError(
                'Adding weight entry with no context to the breakdown.')

        for entry in self._traverse_and_insert(
                self._weight_root, weight_name, stack_context):
            entry.add_weight_size(size_bytes, grad_size_bytes)

        return self

    def set_peak_usage_bytes(self, peak_usage_bytes):
        self._peak_usage_bytes = peak_usage_bytes
        return self

    def build(self):
        if self._peak_usage_bytes is None:
            raise RuntimeError(
                'Missing peak usage when constructing the breakdown.')

        self._prune_tree(self._operation_root)
        self._prune_tree(self._weight_root)
        return HierarchicalBreakdown(
            operations=self._operation_root,
            weights=self._weight_root,
            peak_usage_bytes=self._peak_usage_bytes,
        )

    def _traverse_and_insert(self, root, leaf_name, stack_context):
        """
        A generator that, given a list of relevant stack frames, traverses (and
        inserts entries, if needed) the hierarchical breakdown tree, yielding
        each node along its path.
        """
        parent = root
        node_constructor = type(root)
        stack_frames = stack_context.frames

        for idx, frame in enumerate(reversed(stack_frames)):
            is_last_frame = idx == len(stack_frames) - 1
            context = (frame.file_path, frame.line_number)

            if context not in parent.children:
                name = (
                    leaf_name if is_last_frame
                    else self._module_names_by_id[frame.module_id]
                )
                new_entry = node_constructor(name, frame.module_id)
                new_entry.add_context(context)
                parent.children[context] = new_entry

            yield parent
            parent = parent.children[context]

        yield parent

    def _prune_tree(self, root):
        # Current node, key to node from parent, parent node
        stack = [(root, None, None)]

        # Depth first traversal. Prune interior nodes with only one child:
        #   e.g. ... -> parent -> node -> child -> ... becomes
        #        ... -> parent -> child -> ...
        while len(stack) != 0:
            node, key, parent = stack.pop()

            if len(node.children) == 1 and parent is not None:
                # Remove "node" from the tree and have the parent
                # point directly to node's only child
                child = next(iter(node.children.values()))
                child.add_context(key)
                parent.children[key] = child
                node.children.clear()
                stack.append((child, key, parent))
            else:
                for key_to_child, child in node.children.items():
                    stack.append((child, key_to_child, node))


class BreakdownNode:
    def __init__(self, name, module_id):
        self._name = name
        self._module_id = module_id

        self._children = {}
        self._contexts = set()

    def add_context(self, context):
        self._contexts.add(context)

    @property
    def name(self):
        return self._name

    @property
    def children(self):
        return self._children

    def serialize_data_to_protobuf(self, entry):
        # Template method, must be implemented by subclasses
        raise NotImplementedError

    def serialize_to_protobuf(self, array):
        # Serialize using a preorder traversal
        stack = [self]
        while len(stack) != 0:
            node = stack.pop()

            entry = array.add()
            entry.name = self.name
            entry.num_children = len(self.children)
            for file_path, line_number in self._contexts:
                file_ref = entry.contexts.add()
                file_ref.line_number = line_number
                file_ref.file_path.components.extend(file_path.split(os.sep))
            self.serialize_data_to_protobuf(entry)

            for child in node.children.values():
                stack.append(child)


class OperationNode(BreakdownNode):
    def __init__(self, name, module_id):
        super().__init__(name, module_id)
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

    @property
    def forward_ms(self):
        return self._forward_ms

    @property
    def backward_ms(self):
        return self._backward_ms

    @property
    def size_bytes(self):
        return self._size_bytes

    def serialize_data_to_protobuf(self, entry):
        entry.operation.forward_ms = self.forward_ms
        entry.operation.backward_ms = (
            self.backward_ms if self.backward_ms is not None
            else math.nan
        )
        entry.operation.size_bytes = self.size_bytes


class WeightNode(BreakdownNode):
    def __init__(self, name, module_id):
        super().__init__(name, module_id)
        self._size_bytes = 0
        self._grad_size_bytes = 0

    def add_weight_size(self, size_bytes, grad_size_bytes):
        self._size_bytes += size_bytes
        self._grad_size_bytes += grad_size_bytes

    @property
    def size_bytes(self):
        return self._size_bytes

    @property
    def grad_size_bytes(self):
        return self._grad_size_bytes

    def serialize_data_to_protobuf(self, entry):
        entry.weight.size_bytes = self.size_bytes
        entry.weight.grad_size_bytes = self.grad_size_bytes
