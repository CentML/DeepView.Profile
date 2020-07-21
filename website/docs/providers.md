---
id: providers
title: Providers in Detail
---
### Model Provider

```python
def skyline_model_provider() -> torch.nn.Module:
    pass
```

The model provider must take no arguments and return an instance of your model
(a `torch.nn.Module`) that is on the GPU (i.e. you need to call `.cuda()` on
the module before returning it).


### Input Provider

```python
def skyline_input_provider(batch_size: int = 32) -> Tuple:
    pass
```

The input provider must take a single `batch_size` argument that has a default
value (the batch size you want to profile with). It must return an iterable
(does not *have* to be a `tuple`) that contains the arguments that you would
normally pass to your model's `forward` method. Any `Tensor`s in the returned
iterable must be on the GPU (i.e. you need to call `.cuda()` on them before
returning them).


### Iteration Provider

```python
def skyline_iteration_provider(model: torch.nn.Module) -> Callable:
    pass
```

The iteration provider must take a single `model` argument, which will be an
instance of your model. This provider must return a callable (e.g., a function)
that, when invoked, runs a single training iteration.
