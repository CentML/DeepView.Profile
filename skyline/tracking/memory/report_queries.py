import enum


class EntryType(enum.Enum):
    Weight = 1
    Activation = 2


create_report_tables = {
    'weight_entries': """
      CREATE TABLE IF NOT EXISTS weight_entries (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        grad_size_bytes INTEGER NOT NULL
      )
    """,
    'activation_entries': """
      CREATE TABLE IF NOT EXISTS activation_entries (
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
    'correlation_index': """
      CREATE UNIQUE INDEX IF NOT EXISTS entry_type_and_id
        ON stack_correlation(entry_type, entry_id)
    """,
    'stack_frames': """
      CREATE TABLE IF NOT EXISTS stack_frames (
        correlation_id INTEGER NOT NULL,
        ordering INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        line_number INTEGER NOT NULL,
        PRIMARY KEY (correlation_id, ordering)
      )
    """,
    'entry_types': """
      CREATE TABLE IF NOT EXISTS entry_types (
        entry_type INTEGER PRIMARY KEY,
        name TEXT NOT NULL
      )
    """,
    'misc_sizes': """
      CREATE TABLE IF NOT EXISTS misc_sizes (
        key TEXT PRIMARY KEY,
        size_bytes INT NOT NULL
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

add_activation_entry = """
  INSERT INTO activation_entries (id, operation_name, size_bytes)
    VALUES (NULL, ?, ?)
"""

add_correlation_entry = """
  INSERT INTO stack_correlation (correlation_id, entry_id, entry_type)
    VALUES (NULL, ?, ?)
"""

add_stack_frame = """
  INSERT INTO stack_frames (correlation_id, ordering, file_path, line_number)
    VALUES (?, ?, ?, ?)
"""

add_misc_entry = "INSERT INTO misc_sizes (key, size_bytes) VALUES (?, ?)"

get_misc_entry = "SELECT size_bytes FROM misc_sizes WHERE key = ?"

get_code_context_subquery = """
  WITH code_contexts AS (
    SELECT c.entry_id, s.file_path, s.line_number
      FROM stack_frames AS s JOIN stack_correlation AS c
        ON s.correlation_id == c.correlation_id
      WHERE
        c.entry_type = {:d}
      GROUP BY s.correlation_id HAVING s.ordering == MIN(s.ordering)
  )
"""

get_weight_entries_with_context = (
    get_code_context_subquery.format(EntryType.Weight.value) +
    """
      SELECT
          w.name, w.size_bytes, w.grad_size_bytes, c.file_path, c.line_number
        FROM weight_entries AS w
          LEFT JOIN code_contexts AS c
          ON w.id == c.entry_id
        WHERE w.size_bytes > 0
        ORDER BY c.file_path ASC, c.line_number ASC
    """
)

get_activation_entries_with_context = (
    get_code_context_subquery.format(EntryType.Activation.value) +
    """
      SELECT a.operation_name, a.size_bytes, c.file_path, c.line_number
        FROM activation_entries AS a
          LEFT JOIN code_contexts AS c
          ON a.id == c.entry_id
        WHERE a.size_bytes > 0
        ORDER BY c.file_path ASC, c.line_number ASC
    """
)
