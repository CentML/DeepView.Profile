---
id: standalone
title: Standalone Profiling
---

Skyline also supports standalone profiling (i.e. profiling from the command
line like a traditional profiler). Standalone profiling is useful when you just
want access to Skyline's profiling functionality. Skyline will save the
profiling results (called a "report") into a [SQLite database
file](https://www.sqlite.org/) that you can then query yourself. We describe
the database schema for Skyline's run time and memory reports in the [Run Time
Report Format](run-time-report.md) and [Memory Report Format](memory-report.md)
pages respectively.


## Preparing Your Code

You do not need to do anything special to prepare your code for standalone
profiling. You only need to write an *entry point* file just as if you were
going to start an interactive profiling session. See the [Getting
Started](getting-started.md) page for an example of an entry point file.


## Run Time Profiling

To have Skyline perform run time profiling, you use the `skyline time`
subcommand. In addition to the entry point file, you also need to specify the
file where you want Skyline to save the run time profiling report using the
`--output` or `-o` flag.

:::note
Just like when you use `skyline interactive`, you need to place your shell
inside the project root and you need to specify a relative path to your entry
point file.
:::

```bash
cd ~/my/project/root
skyline time entry_point.py --output my_output_file.sqlite
```


## Memory Profiling

Launching memory profiling is almost the same as launching run time profiling.
You just need to use `skyline memory` instead of `skyline time`.

```bash
cd ~/my/project/root
skyline memory entry_point.py --output my_output_file.sqlite
```
