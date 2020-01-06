Skyline Command Line Interface (CLI)
====================================
This directory contains the code that implements Skyline's command line
interface (CLI). The CLI is written in Python and can be installed as an
executable.

Right now, the CLI serves as the entrypoint for the interactive profiler. Users
can start a profiling session by running

```
$ skyline interactive <model entrypoint file>
```

Development Environment
-----------------------
To set up a development version of the CLI, use `pip` to install an editable
version of the `skyline` package. We recommend that you do this inside a
virtual Python environment such as `virtualenv` since this process will install
other Python packages as well (Skyline's dependencies).

For your convenience, you can run the `dev-setup.sh` script to create a
development environment inside a virtualenv:

```sh
# You only need to run this once
./dev-setup.sh

# Run this to activate your development environment
source env/bin/activate

# Test out Skyline
skyline --help
```

If you want to set up your development environment manually:

```sh
# To install a development version of the CLI, run (inside this directory):
pip3 install --editable .
```
