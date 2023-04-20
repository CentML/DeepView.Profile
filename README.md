![DeepView](https://raw.githubusercontent.com/CentML/DeepView.Profile/main/assets/deepview.png)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat)](https://github.com/CentML/DeepView.Profile/blob/main/LICENSE)
![](https://img.shields.io/pypi/pyversions/deepview-profile.svg)
[![](https://img.shields.io/pypi/v/deepview-profile.svg)](https://pypi.org/project/deepview-profile/)

DeepView.Profile is a tool to profile and debug the training performance of [PyTorch](https://pytorch.org) neural networks.

- [Installation](#installation)
- [Usage example](#getting-started)
- [Development Environment Setup](#dev-setup)
- [Release Process](#release-process)
- [Release History](#release-history)
- [Meta](#meta)
- [Contributing](#contributing)

<h2 id="installation">Installation</h2>

DeepView.Profile works with *GPU-based* neural networks that are implemented in [PyTorch](https://pytorch.org).

To run DeepView.Profile, you need:
- A system equipped with an NVIDIA GPU
- Python 3.7+
- PyTorch 1.1.0+ with CUDA
  - **NOTE:**  We assume you have the correct version of PyTorch installed for their GPU. Default PyTorch installation on Linux distros might not have CUDA support. If you see error similar to below, your PyTorch version is incompatible with your version of CUDA. You can download the appropriate version from the [PyTorch site](https://pytorch.org/get-started/locally/)
    ```NVIDIA GeForce RTX 3060 Ti with CUDA capability sm_86 is not compatible with the current PyTorch installation.
    The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
    If you want to use the NVIDIA GeForce RTX 3060 Ti GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
    ```

### Installation from PyPi

Installing with [Pip](https://packaging.python.org/en/latest/tutorials/installing-packages/#use-pip-for-installing)
```zsh
pip install deepview-profile
```

### Installation from source
```bash
git clone https://github.com/CentML/DeepView.Profile
cd DeepView.Profile
poetry install
poetry run deepview --help
```

<h2 id="getting-started">Usage example</h2>

To use DeepView.Profile in your project, you need to first write an entry point file, which is a regular Python file that describes how your model is created and trained. See the [Entry Point](docs/providers.md) for more information.

Once your entry point file is ready, there are two ways to profile interactive profiling and standalone profiling.

### Interactive Profiling
```zsh
python3 -m deepview_profile interactive
```

### Standalone Profiling
Standalone profiling is useful when you just want access to DeepView.Profile's profiling functionality. DeepView.Profile will save the profiling results (called a "report") into a [SQLite database file](https://www.sqlite.org/) that you can then query yourself. We describe the database schema for DeepView.Profile's run time and memory reports in the [Run Time Report Format](docs/run-time-report.md) and [Memory Report Format](docs/memory-report.md) pages respectively.

To have DeepView.Profile perform run time profiling, you use the `deepview time`
subcommand. In addition to the entry point file, you also need to specify the
file where you want DeepView.Profile to save the run time profiling report using the
`--output` or `-o` flag.

```zsh
python3 -m deepview_profile time entry_point.py --output my_output_file.sqlite
```

Launching memory profiling is almost the same as launching run time profiling.
You just need to use `deepview memory` instead of `deepview time`.

```zsh
python3 -m deepview_profile memory entry_point.py --output my_output_file.sqlite
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
    * Publish to Test PyPI
    * Subject to approval publish to PyPI

<h2 id="release-history">Release History</h2>

See [Releases](https://github.com/CentML/DeepView.Profile/releases)

<h2 id="meta">Meta</h2>

DeepView.Profile began as a research project at the [University of Toronto](https://web.cs.toronto.edu) in collaboration with [Geofrey Yu](mailto:gxyu@cs.toronto.edu), [Tovi Grossman](https://www.tovigrossman.com) and [Gennady Pekhimenko](https://www.cs.toronto.edu/~pekhimenko/).

The accompanying research paper appears in the proceedings of UIST'20. If you are interested, you can read a preprint of the paper [here](https://arxiv.org/pdf/2008.06798.pdf).

If you use DeepView.Profile in your research, please consider citing our paper:

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

It is distributed under Apache 2.0 license. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for more information.

<h2 id="contributing">Contributing</h2>

Check out [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to help with DeepView.Profile.
