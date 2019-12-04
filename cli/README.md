INNPV Command Line Interface (CLI)
==================================
This directory contains the code that implements INNPV's command line interface
(CLI). The CLI is written in Python and can be installed as an executable.

Right now, the CLI serves as the entrypoint for the interactive profiler. Users
can start a profiling session by running

```
$ innpv interactive <model entrypoint file>
```

Development Environment
-----------------------
To set up a development version of the CLI, use `pip` to install an editable
version of the `innpv` package. We recommend that you do this inside a virtual
Python environment such as `virtualenv` since this process will install other
Python packages as well (INNPV's dependencies).

```sh
# To install a development version of the CLI, run (inside this directory):
pip3 install --editable .
```

If using `virtualenv`:
```sh
# Run (inside this directory)
virtualenv -p $(which python3) env
source env/bin/activate
pip3 install --editable .
```
