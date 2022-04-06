---
id: memory-report
title: Memory Report
---

This page describes the database schema of the memory report that is generated
by Skyline's [`memory` subcommand](cli.md#memory-profiling). Recall
that Skyline's reports (memory and run time) are [SQLite database
files](https://www.sqlite.org/).

:::note
Skyline's memory profiling is for GPU memory only.
:::

## Overview

Skyline tracks the memory usage associated with a model's *weights* and
*activations*. Skyline will also report the peak amount of memory allocated
during a training iteration.

Just like the run time report, Skyline also includes the stack trace associated
with each activation or weight in the report. Skyline only includes the stack
frames associated with files inside your project (i.e. files under your
project's root directory).

## Tables

### `weight_entries`

```sql title="Schema"
CREATE TABLE weight_entries (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  grad_size_bytes INTEGER NOT NULL
);
```
This table holds the memory used by the model's weights. The `size_bytes`
column is the number of bytes used by the weight and `grad_size_bytes` is the
number of bytes used by the weight's gradient tensor. The `name` column holds
the weight's name, which is assigned by PyTorch.

### `activation_entries`

```sql title="Schema"
CREATE TABLE activation_entries (
  id INTEGER PRIMARY KEY,
  operation_name TEXT NOT NULL,
  size_bytes INTEGER NOT NULL
);
```
This table holds the memory used by the model's activations in one training
iteration. The `size_bytes` column is the number of bytes used by the
activation. The `operation_name` column is the name of the operation that
generated the activation.

### `entry_types`

```sql title="Schema"
CREATE TABLE entry_types (
  entry_type INTEGER PRIMARY KEY,
  name TEXT NOT NULL
);
```
This is a table that stores mappings of Skyline's memory entry types
(activations, weights) to numeric identifiers. Skyline maps weights to an entry
type of `1`, and activations to an entry type of `2`.

### `stack_correlation`

```sql title="Schema"
CREATE TABLE stack_correlation (
  correlation_id INTEGER PRIMARY KEY,
  entry_id INTEGER NOT NULL,
  entry_type INTEGER NOT NULL,
  UNIQUE (correlation_id, entry_id)
);
CREATE UNIQUE INDEX entry_type_and_id
  ON stack_correlation(entry_type, entry_id);
```
This table maps entries to a `correlation_id`, which can be used to look up a
memory entry's relevant stack frames in the `stack_frames` table. The
`entry_type` column contains either `1` or `2`, which corresponds to the
weights and activations respectively.

For all rows where `entry_type == 1`, the `entry_id` column will act as a
foreign key for the `id` column in the `weight_entries` table. Similarly for
all rows where `entry_type == 2`, the `entry_id` column will act as a foreign
key for the `id` column in the `activation_entries` table.

### `stack_frames`

```sql title="Schema"
CREATE TABLE stack_frames (
  correlation_id INTEGER NOT NULL,
  ordering INTEGER NOT NULL,
  file_path TEXT NOT NULL,
  line_number INTEGER NOT NULL,
  PRIMARY KEY (correlation_id, ordering)
);
```
This table holds the stack frames associated with a memory usage entry (both
weights and activations). The `correlation_id` column is a foreign key that
references the `correlation_id` in the `stack_correlation` table. File paths
stored in the `file_path` column will be relative to the project's root
directory and line numbers are 1-based.

:::note
Skyline does not add an explicit foreign key constraint to the `correlation_id`
column.
:::

**Ordering.**
There may be multiple stack frames associated with any given memory entry (i.e.
any given `correlation_id`). The `ordering` column is used to keep track of the
ordering among stack frames that share the same `correlation_id`. When sorted
in ascending order by the `ordering` column, the stack frames will be ordered
from most-specific (i.e. *closest* to the weight or operation responsible for
the activation) to least-specific (i.e. *farthest* from the weight or operation
responsible for the activation).

**Connecting to Entries.**
To get the stack frames for a given entry, you need to first query the
`stack_correlation` table to find the `correlation_id` associated with your
`entry_id` and `entry_type` combination. Then you can use that `correlation_id`
to look up the associated stack frames in this table.

### `misc_sizes`

```sql title="Schema"
CREATE TABLE misc_sizes (
  key TEXT PRIMARY KEY,
  size_bytes INT NOT NULL
);
```

This table holds any miscellaneous memory usage information that is reported by
Skyline. Currently, Skyline only uses this table to report the peak memory
usage during one training iteration. This memory usage is reported using the
`peak_usage_bytes` key.
