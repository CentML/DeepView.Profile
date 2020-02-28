![Skyline for Atom](https://raw.githubusercontent.com/geoffxy/skyline-atom/master/assets/skyline-wordmark.png)

-------------------------------------------------------------------------------

Skyline is a tool used with [Atom](https://atom.io) to profile, visualize, and
debug the training performance of [PyTorch](https://pytorch.org) neural
networks.

**Note:** Skyline is still under active development and should be considered an
"alpha" product. Its usage and system requirements are subject to change
between versions. See [Versioning](#versioning) for more details.

- [Installing Skyline](#installing-skyline)
- [Getting Started](#getting-started)
- [Providers in Detail](#providers-in-detail)
- [Versioning](#versioning)
- [Authors](#authors)

-------------------------------------------------------------------------------

<h2 id="installing-skyline">Installing Skyline</h2>

### Requirements

Skyline works with GPU-based neural networks that are implemented in PyTorch.
To run Skyline, you need:

- A system equipped with an NVIDIA GPU
- PyTorch 1.1.0+
- Python 3.6+

Skyline is currently only supported on Ubuntu 18.04. It should also work on
other Ubuntu versions that can run Atom and that have Python 3.6+.


### Installation

Skyline consists of two components: a command line tool and an Atom plugin
(this repository). Both components must be installed to use Skyline. They can
be installed using `pip` and `apm`:

```
pip install skyline-cli
apm install skyline
```

After installing Skyline, you will be able to invoke the command line tool by
running `skyline` in your shell.


<h2 id="getting-started">Getting Started</h2>

To use Skyline in your project, you need to first write an *entry point file*,
which is a regular Python file that describes how your model is created and
trained. See the [Entry Point](#entry-point) section for more information.

Once your entry point file is ready, navigate to your project's *root
directory* and run:

```
skyline interactive path/to/entry/point/file
```

Then, open up Atom, execute the `Skyline:Toggle` command in the command palette
(Ctrl-Shift-P), and hit the "Connect" button that appears on the right.

To shutdown Skyline, just execute the `Skyline:Toggle` command again in the
command palette. You can shutdown the interactive profiling session on the
command line by hitting Ctrl-C in your terminal.

You can also toggle the Skyline through the Atom menus: Packages > Skyline >
Show/Hide Skyline.

**Important:** To analyze your model, Skyline will actually run your code. This
means that when you invoke `skyline interactive`, you need to make sure that
your shell has the proper environments activated (if needed). For example if
you use `virtualenv` to manage your model's dependencies, you need to activate
your `virtualenv` before starting Skyline.

**Usage Statistics:** Skyline collects usage statistics in order to help us
make improvements to the tool. If you do not want Skyline to collect usage
statistics, you can disable this functionality through Skyline's package
settings in Atom (Atom > Settings/Preferences > Packages > Skyline > Settings).


### Projects

To use Skyline, all of the code that you want to profile interactively must be
stored under one common directory. Generally, this just means you need to keep
your own source code under one common directory. Skyline considers all the
files inside this common directory to be part of a *project*, and calls this
common directory your project's *root directory*.

When starting a Skyline interactive profiling session, you must invoke `skyline
interactive <entry point>` inside your project's *root directory*.


<h3 id="entry-point">Entry Point</h3>

Skyline uses an *entry point* file to learn how to create and train your model.
An entry point file is a regular Python file that contains three top-level
functions:

- `skyline_model_provider`
- `skyline_input_provider`
- `skyline_iteration_provider`

These three functions are called *providers* and must be defined with specific
signatures. The easiest way to understand how to write the providers is to read
through an example.


### Example

Suppose that your project code is kept under a `my_project` directory:

```
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

One way to write the *entry point* file would be:

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

One important thing to highlight is our use of a wrapper `ModelWithLoss`
module. Since Skyline needs to be able to call `.backwards()` directly on the
output tensor of our model, we need to use this wrapper module to compute and
return the loss of our model's output with respect to the targets (i.e. the
labels). We also include the targets as inputs to our wrapped module and in our
input provider.

You can place these *provider* functions either in a new file or directly in
`model.py`. Whichever file contains the providers will be your project's *entry
point file*. In this example, suppose that we defined the providers in a
separate file called `entry_point.py` inside `my_project`.

Suppose that `my_project` is in your home directory. To launch Skyline you
would run (in your shell):

```
cd ~/my_project
skyline interactive entry_point.py
```


<h2 id="providers-in-detail">Providers in Detail</h2>

### Model Provider

```python
def skyline_model_provider() -> torch.nn.Module:
    pass
```

The model provider must take no arguments and return an instance of your model
(a `torch.nn.Module`) that is on the GPU (i.e. you need to call `.cuda()` on
the module before returning it).

**Important:** Your model must return a tensor on which `.backward()` can be
called. Generally this means that the `torch.nn.Module` you return must compute
the loss with respect to the inputs passed into the model.


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


<h2 id="versioning">Versioning</h2>

Skyline uses semantic versioning. Before the 1.0.0 release, backwards
compatibility between minor versions will not be guaranteed.

The Skyline command line tool and plugin use *independent* version numbers.
However, it is very likely that minor and major versions of the command line
tool and plugin will be released together (and hence share major/minor version
numbers).

Generally speaking, the most recent version of the command line tool and plugin
will be compatible with each other.


<h2 id="authors">Authors</h2>

Geoffrey Yu <gxyu@cs.toronto.edu>
