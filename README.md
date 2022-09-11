![Skyline](https://raw.githubusercontent.com/skylineprof/skyline/master/assets/skyline-wordmark.png)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat)](https://github.com/CentML/skyline/blob/main/LICENSE)
![](https://img.shields.io/pypi/pyversions/skyline-profiler.svg)
[![](https://img.shields.io/pypi/v/skyline-profiler.svg)](https://pypi.org/project/skyline-profiler/)

Skyline is a tool to profile and debug the training performance of [PyTorch](https://pytorch.org) neural networks.

- [Installation](#installation)
- [Usage example](#getting-started)
- [Development Environment Setup](#dev-setup)
- [Release Process](#release-process)
- [Release History](#release-history)
- [Meta](#meta)
- [Contributing](#contributing)

<h2 id="installation">Installation</h2>

Skyline works with *GPU-based* neural networks that are implemented in [PyTorch](https://pytorch.org).

To run Skyline, you need:
- A system equipped with an NVIDIA GPU
- PyTorch 1.1.0+ with CUDA
  - **NOTE:** Default PyTorch installation on Linux distros might not have CUDA support, so you'll have to download the approriate Python Wheel from [PyTorch site](https://download.pytorch.org/whl/nightly/torch/).
- Python 3.6+ or Python 3.7+ on OSX
- [Poetry](https://python-poetry.org/)

### Installation from source
```zsh
git clone https://github.com/skylineprof/skyline.git
cd skyline
poetry install
poetry run skyline --help
```

### Installation from PyPi

**Note:** Not implemented yet

Installing with [Poetry](https://python-poetry.org/)
```zsh
poetry add skyline-profiler
poetry run skyline --help
```

Installing with [Pipenv](https://pipenv.pypa.io/en/latest/)
```zsh
pipenv install skyline-profiler
pipenv run skyline --help
```

Installing with [Pip](https://packaging.python.org/en/latest/tutorials/installing-packages/#use-pip-for-installing)
```zsh
python3 -m pip install skyline-profiler
python3 skyline
```

<h2 id="getting-started">Usage example</h2>

To use Skyline in your project, you need to first write an entry point file, which is a regular Python file that describes how your model is created and trained. See the [Entry Point](https://github.com/UofT-EcoSystem/skyline/blob/main/docs/providers.md) for more information.

Once your entry point file is ready, there are two ways to profile interactive profiling and standalone profiling.

### Interactive Profiling
```zsh
poetry run skyline interactive --skip-atom path/to/entry/point/file
```

### Standalone Profiling
Standalone profiling is useful when you just want access to Skyline's profiling functionality. Skyline will save the profiling results (called a "report") into a [SQLite database file](https://www.sqlite.org/) that you can then query yourself. We describe the database schema for Skyline's run time and memory reports in the [Run Time Report Format](https://github.com/UofT-EcoSystem/skyline/blob/main/docs/run-time-report.md) and [Memory Report Format](https://github.com/UofT-EcoSystem/skyline/blob/main/docs/memory-report.md) pages respectively.

To have Skyline perform run time profiling, you use the `skyline time`
subcommand. In addition to the entry point file, you also need to specify the
file where you want Skyline to save the run time profiling report using the
`--output` or `-o` flag.

```zsh
poetry run skyline time entry_point.py --output my_output_file.sqlite
```

Launching memory profiling is almost the same as launching run time profiling.
You just need to use `skyline memory` instead of `skyline time`.

```zsh
poetry run skyline memory entry_point.py --output my_output_file.sqlite
```

<h2 id="dev-setup">Development Environment Setup</h2>

From the project root, do
```zsh
poetry install
```
<h2 id="release-process">Release Process</h2>

1. Make sure you're on main branch and it is clean
1. Run [tools/prepare-release.sh](tools/prepare-release.sh) which will:
    * Increment the version
    * Create a release branch
    * Create a release PR
1. After the PR is merged [build-and-publish-new-version.yml](.github/workflows/build-and-publish-new-version.yml) GitHub action will:
    * build the Python Wheels
    * GitHub release
    * Try to publish to Test PyPI
    * Subject to approval publish to PyPI

<h2 id="release-history">Release History</h2>

See [Releases](https://github.com/UofT-EcoSystem/skyline/releases)

<h2 id="meta">Meta</h2>

Skyline began as a research project at the [University of Toronto](https://web.cs.toronto.edu) in collaboration with [Geofrey Yu](mailto:gxyu@cs.toronto.edu), [Tovi Grossman](https://www.tovigrossman.com) and [Gennady Pekhimenko](https://www.cs.toronto.edu/~pekhimenko/).

The accompanying research paper appears in the proceedings of UIST'20. If you are interested, you can read a preprint of the paper [here](https://arxiv.org/pdf/2008.06798.pdf).

If you use Skyline in your research, please consider citing our paper:

```bibtex
@inproceedings{skyline-yu20,
  title = {{Skyline: Interactive In-Editor Computational Performance Profiling
    for Deep Neural Network Training}},
  author = {Yu, Geoffrey X. and Grossman, Tovi and Pekhimenko, Gennady},
  booktitle = {{Proceedings of the 33rd ACM Symposium on User Interface
    Software and Technology (UIST'20)}},
  year = {2020},
}
```

It is distributed under Apache 2.0 license. See [LICENSE](https://github.com/UofT-EcoSystem/skyline/blob/main/LICENSE) and [NOTICE](https://github.com/UofT-EcoSystem/skyline/blob/main/NOTICE) for more information.

<h2 id="contributing">Contributing</h2>

Check out [CONTRIBUTING.md](https://github.com/UofT-EcoSystem/skyline/blob/main/CONTRIBUTING.md) for more information on how to help with Skyline.
