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
        self._operation_root.build_context_info_map()
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
                    else self._module_names_by_id[
                        stack_frames[-idx-2].module_id]
                )
                if name == '':
                    name = 'Model'
                new_entry = node_constructor(name, frame.module_id)
                new_entry.add_context(context)
                parent.children[context] = new_entry

            yield parent
            parent = parent.children[context]

        parent.add_name(leaf_name)
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
        self._names = [name]
        self._module_id = module_id

        self._children = {}
        self._contexts = []

    def add_context(self, context):
        self._contexts.append(context)

    def add_name(self, name):
        if name in self._names:
            return
        self._names.append(name)

    @property
    def name(self):
        return ', '.join(self._names)

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
            entry.name = node.name
            entry.num_children = len(node.children)
            for context in reversed(node._contexts):
                file_ref = entry.contexts.add()
                _serialize_file_ref(file_ref, context)
            node.serialize_data_to_protobuf(entry)

            for child in node.children.values():
                stack.append(child)


class OperationNode(BreakdownNode):
    def __init__(self, name, module_id):
        super().__init__(name, module_id)
        self._forward_ms = 0.
        self._backward_ms = None
        self._size_bytes = 0
        self._context_info_map = None

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
        for context, context_info in self._context_info_map.items():
            map_entry = entry.operation.context_info_map.add()
            _serialize_file_ref(map_entry.context, context)
            context_info.serialize_to_protobuf(map_entry)

    def build_context_info_map(self):
        """
        Builds aggregate memory/run time usage information for each tracked
        line of code.
        """
        stack = [(self, 0)]
        while len(stack) > 0:
            node, visit_count = stack.pop()

            if visit_count > 0:
                for child in node.children.values():
                    context_info = ContextInfo(
                        size_bytes=child.size_bytes,
                        run_time_ms=(
                            child.forward_ms if child.backward_ms is None
                            else child.forward_ms + child.backward_ms
                        ),
                    )
                    for context in child._contexts:
                        if context in node._context_info_map:
                            node._context_info_map[context] += context_info
                        else:
                            node._context_info_map[context] = context_info

                    ContextInfo.merge_map(
                        node._context_info_map,
                        child._context_info_map,
                    )
            else:
                node._context_info_map = {}
                stack.append((node, 1))
                for child in node.children.values():
                    stack.append((child, 0))


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


class ContextInfo:
    def __init__(self, size_bytes, run_time_ms, invocations=1):
        self._size_bytes = size_bytes
        self._run_time_ms = run_time_ms
        self._invocations = invocations

    def __add__(self, other):
        return ContextInfo(
            size_bytes=self.size_bytes + other.size_bytes,
            run_time_ms=self.run_time_ms + other.run_time_ms,
            invocations=self.invocations + other.invocations,
        )

    def __iadd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return (
            'ContextInfo(invocations={:d}, size_bytes={:d}, '
            'run_time_ms={:.4f})'.format(
                self.invocations, self.size_bytes, self.run_time_ms)
        )

    @property
    def size_bytes(self):
        return self._size_bytes

    @property
    def run_time_ms(self):
        return self._run_time_ms

    @property
    def invocations(self):
        return self._invocations

    @staticmethod
    def merge_map(destination, to_merge):
        for key, value in to_merge.items():
            if key in destination:
                destination[key] += value
            else:
                destination[key] = value

    def serialize_to_protobuf(self, entry):
        entry.run_time_ms = self.run_time_ms
        entry.size_bytes = self.size_bytes
        entry.invocations = self.invocations


def _serialize_file_ref(file_ref, context):
    file_path, line_number = context
    file_ref.line_number = line_number
    file_ref.file_path.components.extend(file_path.split(os.sep))
