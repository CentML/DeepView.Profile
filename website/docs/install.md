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

Note that Skyline only supports profiling models that are trained using a GPU.

Skyline consists of two components: (i) the Skyline profiler, and (ii) a plugin
for Atom. The Skyline profiler has only been tested on Ubuntu 18.04, but should
also work on other versions of Ubuntu that have Python 3.6+. The Skyline plugin
has been tested on both Ubuntu 18.04 and macOS Mojave (10.14) with the latest
version of Atom.


### Installation

For interactive profiling (i.e. Skyline inside Atom), you need both components.
If you only plan to use Skyline for [Standalone Profiling](standalone.md), you
only need the Skyline profiler component.

#### Skyline Profiler

The Skyline profiler can be installed using `pip`. In your shell, run:

```bash
pip install skyline-cli
```

As with most Python packages, we recommend installing Skyline inside a
`virtualenv`. After installing the Skyline profiler, you will be able to
invoke it by running `skyline` in your shell.

#### Skyline Atom Plugin

The Skyline Atom plugin can be installed using `apm` (the Atom package
manager), which should be automatically installed after you install
[Atom](https://atom.io). In your shell, run:

```bash
apm install skyline
```

You can alternatively install Skyline by using Atom's preferences pane and
searching for `skyline` under the Install section.

After installing the Skyline plugin, the `Skyline:Toggle` command should be
available in your command palette. A Skyline sub-menu should also appear
under the "Packages" menu.
