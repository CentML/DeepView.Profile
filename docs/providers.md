### Model Provider

```python
def skyline_model_provider() -> torch.nn.Module:
    pass
```

The model provider must take no arguments and return an instance of your model (a `torch.nn.Module`) that is on the GPU (i.e. you need to call `.cuda()` on the module before returning it).

### Input Provider

```python
def skyline_input_provider(batch_size: int = 32) -> Tuple:
    pass
```

The input provider must take a single `batch_size` argument that has a default value (the batch size you want to profile with). It must return an iterable (does not *have* to be a `tuple`) that contains the arguments that you would normally pass to your model's `forward` method. Any `Tensor`s in the returned iterable must be on the GPU (i.e. you need to call `.cuda()` on them before returning them).


### Iteration Provider

```python
def skyline_iteration_provider(model: torch.nn.Module) -> Callable:
    pass
```

The iteration provider must take a single `model` argument, which will be an instance of your model. This provider must return a callable (e.g., a function) that, when invoked, runs a single training iteration.

### Example

Suppose that your project code is kept under a `my_project` directory:

```zsh
my_project
├── __init__.py
└── model.py
```
and your model is defined in `model.py`:

```python
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.linear = nn.Linear(in_features=387096, out_features=10)

    def forward(self, input):
        out = self.conv(input)
        return self.linear(out.view(-1, 387096))
```

One way to write the `entry_point.py` file would be:

```python
import torch
import torch.nn as nn

from my_project.model import Model


class ModelWithLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target):
        output = self.model(input)
        return self.loss_fn(output, target)


def skyline_model_provider():
    # Return a GPU-based instance of our model (that returns a loss)
    return ModelWithLoss().cuda()


def skyline_input_provider(batch_size=32):
    # Return GPU-based inputs for our model
    return (
      torch.randn((batch_size, 3, 256, 256)).cuda(),
      torch.randint(low=0, high=9, size=(batch_size,)).cuda(),
    )


def skyline_iteration_provider(model):
    # Return a function that executes one training iteration
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    def iteration(*inputs):
        optimizer.zero_grad()
        out = model(*inputs)
        out.backward()
        optimizer.step()
    return iteration
```
One important thing to highlight is our use of a wrapper `ModelWithLoss` module. Skyline only provides breakdowns for operations that run inside the module returned by the model provider. We included the loss function in this wrapper module to have Skyline include it in the breakdown. We could have also placed the loss function call in the `iteration` function.

You can place these provider functions either in a new file or directly in `model.py`. Whichever file contains the providers will be your project's entry point file. In this example, we defined the providers in a separate file called `entry_point.py` inside `my_project`.
