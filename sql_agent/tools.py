"""Tool definitions (what the model sees) and handlers (what actually runs).

Tool names, descriptions and schemas are the agent's API docs. Most agent bugs
are fixed here, not in the system prompt.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .db import QueryError, ReadOnlyDB

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_tables",
        "description": "List every table in the database with its row count. Call this first to see what data exists.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "describe_table",
        "description": (
            "Show a table's columns (name, type, nullable, primary key), foreign keys, and 3 sample rows. "
            "Call this before writing SQL against a table so you use real column names and value formats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"table": {"type": "string", "description": "Exact table name from list_tables"}},
            "required": ["table"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "run_sql",
        "description": (
            "Run ONE read-only SQLite SELECT (CTEs allowed) and get the result as JSON: columns, rows, "
            "truncated flag. At most 100 rows are returned, so aggregate in SQL (COUNT, SUM, GROUP BY, "
            "ORDER BY ... LIMIT) instead of pulling raw rows. Writes, PRAGMA and restricted columns are "
            "refused with an error that explains why."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single SQLite SELECT statement"},
                "purpose": {"type": "string", "description": "One short sentence: what this query checks"},
            },
            "required": ["sql", "purpose"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class ToolRunner:
    """Maps tool names to handlers. Every handler returns a string for the model."""

    def __init__(self, db: ReadOnlyDB, max_result_chars: int = 20_000):
        self.db = db
        self.max_result_chars = max_result_chars
        self._handlers: dict[str, Callable[..., Any]] = {
            "list_tables": self._list_tables,
            "describe_table": self._describe_table,
            "run_sql": self._run_sql,
        }

    def run(self, name: str, args: dict[str, Any]) -> str:
        """Raises QueryError for errors the model should see and fix."""
        handler = self._handlers.get(name)
        if handler is None:
            raise QueryError(f"Unknown tool '{name}'. Available: {', '.join(self._handlers)}")
        out = json.dumps(handler(**args), default=str)
        if len(out) > self.max_result_chars:
            out = out[: self.max_result_chars] + '... [result cut off: aggregate or LIMIT your query]'
        return out

    def _list_tables(self) -> Any:
        return self.db.list_tables()

    def _describe_table(self, table: str) -> Any:
        return self.db.describe_table(table)

    def _run_sql(self, sql: str, purpose: str = "") -> Any:
        r = self.db.query(sql)
        return {"columns": r.columns, "rows": r.rows, "row_count": len(r.rows),
                "truncated": r.truncated, "elapsed_ms": r.elapsed_ms}
