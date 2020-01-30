

create_report_tables = {
    'run_time_entries': """
      CREATE TABLE IF NOT EXISTS run_time_entries (
        id INTEGER PRIMARY KEY,
        operation_name TEXT NOT NULL,
        forward_ms REAL NOT NULL,
        backward_ms REAL
      )
    """,
    'stack_frames': """
      CREATE TABLE IF NOT EXISTS stack_frames (
        ordering INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        line_number INTEGER NOT NULL,
        entry_id INTEGER NOT NULL,
        PRIMARY KEY (entry_id, ordering)
      )
    """,
}

set_report_format_version = 'PRAGMA user_version = {version:d}'

add_stack_frame = """
  INSERT INTO stack_frames (ordering, file_path, line_number, entry_id)
    VALUES (?, ?, ?, ?)
"""

add_run_time_entry = """
  INSERT INTO run_time_entries (operation_name, forward_ms, backward_ms)
    VALUES (?, ?, ?)
"""

get_run_time_entries_with_context = """
  WITH code_contexts AS (
    SELECT entry_id, file_path, line_number FROM stack_frames
    GROUP BY entry_id HAVING ordering == MIN(ordering)
  )
  SELECT
    e.operation_name,
    e.forward_ms,
    e.backward_ms,
    c.file_path,
    c.line_number
  FROM
    run_time_entries AS e LEFT JOIN code_contexts AS c
    ON e.id == c.entry_id
  ORDER BY c.file_path ASC, c.line_number ASC
"""
