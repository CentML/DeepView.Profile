---
id: install
title: Installing Skyline
---

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
