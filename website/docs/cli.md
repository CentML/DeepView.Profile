---
id: cli
title: Command Line Interface
---

Skyline is launched from the command line using the `skyline` command. Its
functionality is activated through its subcommands:
[`interactive`](#interactive-profiling), [`memory`](#memory-profiling), and
[`time`](#run-time-profiling).


### Command Line Flags

##### `-v` and `--version`
Use these flags to have Skyline print out its version.

##### `-h` and `--help`
Use these flags to have Skyline print out information about its command line
usage.


### Shared Optional Arguments
The following command line arguments can be used with all of Skyline's
subcommands.

##### `--log-file=<file>`
Use this argument if you would like Skyline to write its logs to a separate
file.

##### `--debug`
Set this command line flag to have Skyline print out more verbose logs. This is
useful primarily for debugging.

-------------------------------------------------------------------------------

## Subcommands

### Interactive Profiling

**Usage:** `skyline interactive path/to/entry_point.py`

To launch an interactive profiling session, you will need to use the `skyline
interactive` command. You need to specify the relative path to your project's
entry point file. This command will launch the Skyline profiling daemon and
will start Atom for you automatically.

:::note
Before running `skyline interactive` you need to navigate to your project's
root directory.
:::

#### Optional Arguments

##### `--skip-atom`
Set this command line flag

##### `--host=<host name>` and `--port=<port>`
Use these arguments to have the Skyline daemon bind to a custom host name
and/or listen on a custom port. By default Skyline will bind to all network
interfaces and will listen on port 60120.

Usually you do not need to set a custom host name nor port. These arguments are
useful if you need to run multiple Skyline daemon processes, or if, due to a
firewall, you need to have Skyline listen on a different port.


### Memory Profiling

**Usage:** `skyline memory --output results.sqlite path/to/entry_point.py`

Use the `memory` subcommand to get a [memory usage report](memory-report.md).
This allows you to get memory usage information without having to launch
Skyline's interactive profiler (i.e. the Skyline Atom plugin).

#### Required Arguments

##### `-o` or `--output`
You need to specify the file where Skyline should save the memory report. This
output file will be a SQLite database.


### Run Time Profiling

**Usage:** `skyline time --output results.sqlite path/to/entry_point.py`

Use the `time` subcommand to get a [iteration run time
report](run-time-report.md). This allows you to get run time information
without having to launch Skyline's interactive profiler (i.e. the Skyline Atom
plugin).

#### Required Arguments

##### `-o` or `--output`
You need to specify the file where Skyline should save the run time report.
This output file will be a SQLite database.
