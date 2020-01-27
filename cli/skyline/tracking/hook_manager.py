

class HookManager:
    def __init__(self):
        self._original_callables = {}

    def attach_hooks_on_module(self, module, predicate, hook_creator):
        self.attach_hooks_on_module_using(
            module, module, predicate, hook_creator)

    def attach_hooks_on_module_using(
            self, module, using_module, predicate, hook_creator):
        """
        Attach hooks onto functions in the provided module. Use the
        `using_module` to discover the existing functions.
        """
        for prop in dir(using_module):
            if not predicate(getattr(module, prop)):
                continue
            self.attach_hook(module, prop, hook_creator)

    def attach_hook(self, module, prop, hook_creator):
        target = getattr(module, prop)
        self._maybe_store_callable(module, prop, target)
        setattr(module, prop, hook_creator(target))

    def remove_hooks(self):
        for module, callable_pairs in self._original_callables.items():
            for prop, original_callable in callable_pairs.items():
                setattr(module, prop, original_callable)
        self._original_callables.clear()

    def _maybe_store_callable(self, module, prop, original_callable):
        """
        Store the original callable (to be able to restore it) only when it is
        the first time we are encountering the given callable.
        """
        if module not in self._original_callables:
            self._original_callables[module] = {}

        if prop in self._original_callables[module]:
            return

        self._original_callables[module][prop] = original_callable
