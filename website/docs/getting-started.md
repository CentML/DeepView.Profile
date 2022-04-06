---
id: getting-started
title: Getting Started
---
To use Skyline in your project, you need to first write an *entry point file*,
which is a regular Python file that describes how your model is created and
trained. See the [Entry Point](#entry-point) section for more information.

Once your entry point file is ready, navigate to your project's *root
directory* and run:

```bash
skyline interactive path/to/entry/point/file
```

Then, open up Atom, execute the `Skyline:Toggle` command in the command palette
(Ctrl-Shift-P on Ubuntu, ⌘-Shift-P on macOS), and hit the "Connect" button that
appears on the right.

To shutdown Skyline, just execute the `Skyline:Toggle` command again in the
command palette. You can shutdown the interactive profiling session on the
command line by hitting Ctrl-C in your terminal.

You can also toggle Skyline using the Atom menus: Packages > Skyline >
Show/Hide Skyline.

:::info Virtual Environments
To analyze your model, Skyline will actually run your code. This means that
when you invoke `skyline interactive`, you need to make sure that your shell
has the proper environments activated (if needed). For example if you use
`virtualenv` to manage your model's dependencies, you need to activate your
`virtualenv` before starting Skyline.
:::

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


### Entry Point

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

```python title="model.py"
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

```python title="entry_point.py"
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
module. Skyline only provides breakdowns for operations that run inside the
module returned by the model provider. We included the loss function in this
wrapper module to have Skyline include it in the breakdown. We could have also
placed the loss function call in the `iteration` function.

You can place these *provider* functions either in a new file or directly in
`model.py`. Whichever file contains the providers will be your project's *entry
point file*. In this example, we defined the providers in a separate file
called `entry_point.py` inside `my_project`.

Suppose that `my_project` is in your home directory. To launch Skyline you
would run (in your shell):

```bash
cd ~/my_project
skyline interactive entry_point.py
```

Skyline will then start a profiling session and will launch Atom. To start
profiling, hit the Connect button that appears in the sidebar.
