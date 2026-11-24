"""Ask from the terminal:  python -m sql_agent.cli "What was revenue in March 2026?" [--trace]"""
from __future__ import annotations

import argparse
import logging
import os

from .agent import SQLAgent
from .config import settings
from .store import RunStore


def main() -> None:
    p = argparse.ArgumentParser(description="Ask the SQL analyst agent a question.")
    p.add_argument("question")
    p.add_argument("--trace", action="store_true", help="print each step's JSON trace")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO if a.trace else logging.WARNING, format="%(message)s")
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit("Set ANTHROPIC_API_KEY first (see .env.example).")

    result = SQLAgent().run(a.question)
    RunStore(settings.runs_db_path).save(result)

    print(result.answer)
    print(f"\n-- run {result.run_id} · {result.status} · {len(result.steps)} steps · "
          f"{result.input_tokens} in / {result.output_tokens} out tokens · {result.duration_ms} ms")


if __name__ == "__main__":
    main()
