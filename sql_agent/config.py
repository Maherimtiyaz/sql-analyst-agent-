"""Settings, read once from environment variables (12-factor style)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("SQL_AGENT_MODEL", "claude-opus-5-5")
    db_path: str = os.getenv("SQL_AGENT_DB", "data/shop.db")
    runs_db_path: str = os.getenv("SQL_AGENT_RUNS_DB", "data/runs.db")
    max_steps: int = int(os.getenv("SQL_AGENT_MAX_STEPS", "10"))
    max_rows: int = int(os.getenv("SQL_AGENT_MAX_ROWS", "100"))
    query_timeout_s: float = float(os.getenv("SQL_AGENT_QUERY_TIMEOUT_S", "3"))
    max_result_chars: int = 20_000  # cap on one tool result, protects the context window
    max_tokens: int = 16_000


settings = Settings()
