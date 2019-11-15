create_report_tables = {
    'weight_entries': """
      CREATE TABLE IF NOT EXISTS weight_entries (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        grad_size_bytes INTEGER NOT NULL
      )
    """,
    'iteration_entries': """
      CREATE TABLE IF NOT EXISTS iteration_entries (
        id INTEGER PRIMARY KEY,
        operation_name TEXT NOT NULL,
        size_bytes INTEGER NOT NULL
      )
    """,
    'correlation': """
      CREATE TABLE IF NOT EXISTS stack_correlation (
        correlation_id INTEGER PRIMARY KEY,
        entry_id INTEGER NOT NULL,
        entry_type INTEGER NOT NULL,
        UNIQUE (correlation_id, entry_id)
      )
    """,
    'stack_frames': """
      CREATE TABLE IF NOT EXISTS stack_frames (
        correlation_id INTEGER NOT NULL,
        ordering INTEGER NOT NULL,
        file_name TEXT NOT NULL,
        lineno INTEGER NOT NULL,
        PRIMARY KEY (correlation_id, ordering)
      )
    """,
    'entry_types': """
      CREATE TABLE IF NOT EXISTS entry_types (
        entry_type INTEGER PRIMARY KEY,
        name TEXT NOT NULL
      )
    """,
}

set_report_format_version = 'PRAGMA user_version = {version:d}'

add_entry_type = """
  INSERT INTO entry_types (entry_type, name) VALUES (?, ?)
"""

add_weight_entry = """
  INSERT INTO weight_entries (id, name, size_bytes, grad_size_bytes)
    VALUES (NULL, ?, ?, ?)
"""

add_iteration_entry = """
  INSERT INTO iteration_entries (id, operation_name, size_bytes)
    VALUES (NULL, ?, ?)
"""

add_correlation_entry = """
  INSERT INTO stack_correlation (correlation_id, entry_id, entry_type)
    VALUES (NULL, ?, ?)
"""

add_stack_frame = """
  INSERT INTO stack_frames (correlation_id, ordering, file_name, lineno)
    VALUES (?, ?, ?, ?)
"""
