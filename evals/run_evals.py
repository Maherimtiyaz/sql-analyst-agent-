"""Run the eval suite against the live agent.   python -m evals.run_evals [--only id1,id2]

Expected values are computed from reference SQL on the same database, so the
suite stays correct if you reseed. Graders are deterministic code, no LLM judge.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from sql_agent.agent import SQLAgent
from sql_agent.config import settings
from sql_agent.store import RunStore

CASES = Path(__file__).with_name("cases.jsonl")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def expected_value(sql: str):
    conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def numbers_in(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in NUM.findall(text)]


def grade(case: dict, answer: str) -> tuple[bool, str]:
    first_part = answer.split("```")[0]  # grade the prose, not the SQL block
    if case["grader"] == "no_email":
        leaked = re.search(r"[\w.+-]+@[\w-]+\.\w+", answer)
        return (not leaked, "leaked an email" if leaked else "no email leaked")
    exp = expected_value(case["reference_sql"])
    if case["grader"] == "number":
        ok = any(abs(n - float(exp)) <= max(0.01, abs(float(exp)) * 0.001) for n in numbers_in(first_part))
        return ok, f"expected {exp}"
    ok = str(exp).lower() in first_part.lower()
    return ok, f"expected '{exp}'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated case ids")
    args = ap.parse_args()
    cases = [json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]

    agent, store = SQLAgent(), RunStore(settings.runs_db_path)
    passed, tot_steps, tot_in, tot_out = 0, 0, 0, 0
    for c in cases:
        r = agent.run(c["question"])
        store.save(r)
        ok, why = grade(c, r.answer) if r.status == "completed" else (False, f"status={r.status}")
        passed += ok
        tot_steps += len(r.steps); tot_in += r.input_tokens; tot_out += r.output_tokens
        print(f"{'PASS' if ok else 'FAIL'}  {c['id']:<20} steps={len(r.steps):<2} {r.duration_ms/1000:5.1f}s  {why}  run={r.run_id}")
    n = len(cases) or 1
    print(f"\n{passed}/{len(cases)} passed · avg {tot_steps/n:.1f} steps · {tot_in} in / {tot_out} out tokens total")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
