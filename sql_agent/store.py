"""Persist every run so you can replay, debug and audit it later."""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from .agent import RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  question TEXT NOT NULL,
  status TEXT NOT NULL,
  answer TEXT NOT NULL,
  model TEXT NOT NULL,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL,
  steps_json TEXT NOT NULL,
  queries_json TEXT NOT NULL
)"""


class RunStore:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def save(self, r: RunResult) -> None:
        d = r.to_dict()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, question, status, answer, model, input_tokens, output_tokens,"
                " duration_ms, steps_json, queries_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (r.run_id, r.question, r.status, r.answer, r.model, r.input_tokens, r.output_tokens,
                 r.duration_ms, json.dumps(d["steps"], default=str), json.dumps(r.queries)))
            self._conn.commit()

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            cols = [c[0] for c in cur.description]
        if row is None:
            return None
        d = dict(zip(cols, row))
        d["steps"] = json.loads(d.pop("steps_json"))
        d["queries"] = json.loads(d.pop("queries_json"))
        return d
