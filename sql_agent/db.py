"""Read-only database access with defense in depth.

The model writes SQL, so treat every query as untrusted input. Three layers:
  1. The file is opened read-only (mode=ro) so writes fail at the OS/engine level.
  2. An SQLite *authorizer* allows only SELECT/READ/FUNCTION actions and blocks
     restricted columns (PII). This is enforced by the engine, not by string checks.
  3. A deadline (progress handler) and a row cap bound the cost of any one query.

In Postgres the same ideas are: a read-only role with column-level GRANTs,
`SET statement_timeout`, and `default_transaction_read_only = on`.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any

# Columns the agent must never read. The engine refuses the query if it touches them.
RESTRICTED_COLUMNS: set[tuple[str, str]] = {("customers", "email")}

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # WITH RECURSIVE
}


class QueryError(Exception):
    """A query failed in a way the model can learn from (bad SQL, blocked, timeout)."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    elapsed_ms: int


def _authorizer(action: int, arg1: str | None, arg2: str | None, _db: str | None, _src: str | None) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and (arg1, arg2) in RESTRICTED_COLUMNS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _jsonable(v: Any) -> Any:
    return v.hex() if isinstance(v, (bytes, bytearray)) else v


class ReadOnlyDB:
    def __init__(self, path: str, max_rows: int = 100, timeout_s: float = 3.0):
        self.path = path
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    def _connect(self, guarded: bool) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        if guarded:
            conn.set_authorizer(_authorizer)
        return conn

    # ---- untrusted path: SQL written by the model ----
    def query(self, sql: str) -> QueryResult:
        sql = sql.strip().rstrip(";")
        if not sql:
            raise QueryError("Empty query.")
        conn = self._connect(guarded=True)
        deadline = time.monotonic() + self.timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        start = time.monotonic()
        try:
            cur = conn.execute(sql)  # sqlite3 refuses multiple statements
            columns = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.DatabaseError as e:
            msg = str(e)
            if "interrupted" in msg:
                msg = f"Query exceeded the {self.timeout_s:g}s time limit. Add filters or aggregate."
            elif "prohibited" in msg or "not authorized" in msg:
                msg += (". Only SELECT queries are allowed, and restricted columns "
                        f"({', '.join(f'{t}.{c}' for t, c in sorted(RESTRICTED_COLUMNS))}) cannot be read. "
                        "Avoid SELECT * on tables with restricted columns.")
            raise QueryError(msg) from None
        except sqlite3.Warning as e:  # e.g. "You can only execute one statement at a time."
            raise QueryError(str(e)) from None
        finally:
            conn.close()
        truncated = len(rows) > self.max_rows
        return QueryResult(
            columns=columns,
            rows=[[_jsonable(v) for v in r] for r in rows[: self.max_rows]],
            truncated=truncated,
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )

    # ---- trusted path: schema introspection written by us ----
    def list_tables(self) -> list[dict[str, Any]]:
        conn = self._connect(guarded=False)
        try:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            return [{"table": n, "rows": conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]} for n in names]
        finally:
            conn.close()

    def describe_table(self, table: str) -> dict[str, Any]:
        conn = self._connect(guarded=False)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if not exists:
                tables = [t["table"] for t in self.list_tables()]
                raise QueryError(f"No table named '{table}'. Tables: {', '.join(tables)}")
            cols = [{
                "name": r[1], "type": r[2], "nullable": not r[3], "primary_key": bool(r[5]),
                **({"restricted": True} if (table, r[1]) in RESTRICTED_COLUMNS else {}),
            } for r in conn.execute(f'PRAGMA table_info("{table}")')]
            fks = [{"column": r[3], "references": f"{r[2]}.{r[4]}"}
                   for r in conn.execute(f'PRAGMA foreign_key_list("{table}")')]
            sample_cols = ", ".join(f'"{c["name"]}"' for c in cols if not c.get("restricted"))
            sample = [list(map(_jsonable, r)) for r in conn.execute(f'SELECT {sample_cols} FROM "{table}" LIMIT 3')]
            return {"table": table, "columns": cols, "foreign_keys": fks,
                    "sample_rows": {"columns": [c["name"] for c in cols if not c.get("restricted")], "rows": sample}}
        finally:
            conn.close()
