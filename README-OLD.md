![Skyline](https://raw.githubusercontent.com/skylineprof/skyline/master/assets/skyline-wordmark.png)

-------------------------------------------------------------------------------

Skyline is a tool used with [Atom](https://atom.io) to profile, visualize, and
debug the training performance of [PyTorch](https://pytorch.org) neural
networks.

**Note:** Skyline is still under active development and should be considered a
beta product. Its usage and system requirements are subject to change
between versions. See [Versioning](#versioning) for more details.

More information about Skyline, including its documentation, can be found on
the [Skyline website](https://skylineprof.github.io).

- [Installing Skyline](#installing-skyline)
- [Getting Started](#getting-started)
- [Versioning](#versioning)
- [License](#license)
- [Research Paper](#research-paper)
- [Authors](#authors)

-------------------------------------------------------------------------------

<h2 id="installing-skyline">Installing Skyline</h2>

Skyline works with *GPU-based* neural networks that are implemented in
[PyTorch](https://pytorch.org).

To run Skyline, you need:

- A system equipped with an NVIDIA GPU
- PyTorch 1.1.0+
- Python 3.6+

Skyline is installed using `pip` and the Atom Package Manager (`apm`).

```bash
pip install skyline-cli
apm install skyline
```

Generally you need *both* packages to use Skyline. However, depending on your
use case and development setup, you may only need the `pip` package or you may
need to install the packages on different machines. **See the [installation
page on the website](https://skylineprof.github.io/docs/install) for detailed
installation instructions tailored to different use cases.**

After installing Skyline, you will be able to invoke the command line tool by
running `skyline` in your shell.


<h2 id="getting-started">Getting Started</h2>

To get started quickly, check out the [Getting Started page on the Skyline
website](https://skylineprof.github.io/docs/getting-started).

For more information about using Skyline, including standalone profiling and
setting up a remote project, please [see the Skyline
documentation](https://skylineprof.github.io/docs/).


<h2 id="versioning">Versioning</h2>

Skyline uses semantic versioning. Before the 1.0.0 release, backward
compatibility between minor versions will not be guaranteed.

The Skyline command line tool and plugin use *independent* version numbers.
However, it is very likely that minor and major versions of the command line
tool and plugin will be released together (and hence share major/minor version
numbers).

Generally speaking, the most recent version of the command line tool and plugin
will be compatible with each other.


<h2 id="license">License</h2>

Skyline is open source software that is licensed under the Apache 2.0 License.
Please see the `LICENSE` and `NOTICE` files in this repository for more
information.

Inside the `samples` directory, we include code samples from third party
developers that carry their own open source licenses. Please see the
`README.md` and `LICENSE` files inside those directories for more information.


<h2 id="research-paper">Research Paper</h2>

Skyline began as a research project at the [University of
Toronto](https://web.cs.toronto.edu); the accompanying research paper appears
in the proceedings of UIST'20. If you are interested, you can read a preprint
of the paper [here](https://arxiv.org/pdf/2008.06798.pdf).

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


<h2 id="authors">Authors</h2>

Skyline was written by and is primarily maintained by Geoffrey Yu
(gxyu@cs.toronto.edu).

Skyline began as a research project at the [University of
Toronto](https://web.cs.toronto.edu) in collaboration with [Tovi
Grossman](https://www.tovigrossman.com) and [Gennady
Pekhimenko](https://www.cs.toronto.edu/~pekhimenko/).
