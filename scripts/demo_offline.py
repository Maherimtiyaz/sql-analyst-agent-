"""Offline walkthrough: the REAL agent loop, tools and database, with a SCRIPTED model.

The model's decisions below are written by hand to show a typical run. Everything
else is real: tool execution, SQL errors, the PII block, traces, run storage.
Run:  python -m scripts.demo_offline
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time

from sql_agent.agent import SQLAgent
from sql_agent.config import Settings
from sql_agent.store import RunStore
from tests.conftest import FakeClient, response, text, tool_use

QUESTION = "Which product category made the most revenue in March 2026?"

BAD_SQL = """SELECT p.category, SUM(o.total) AS revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id
WHERE o.created_at LIKE '2026-03%' GROUP BY p.category ORDER BY revenue DESC"""

GOOD_SQL = """SELECT p.category,
       ROUND(SUM(oi.quantity * oi.unit_price_cents) / 100.0, 2) AS revenue_usd
FROM order_items oi
JOIN orders o   ON o.id = oi.order_id
JOIN products p ON p.id = oi.product_id
WHERE o.status = 'completed'
  AND o.created_at >= '2026-03-01' AND o.created_at < '2026-04-01'
GROUP BY p.category
ORDER BY revenue_usd DESC"""

SCRIPTED_MODEL = [
    response(text("I'll start by looking at what tables exist."),
             tool_use("t1", "list_tables"), stop_reason="tool_use"),
    response(tool_use("t2", "describe_table", table="orders"),
             tool_use("t3", "describe_table", table="order_items"),
             tool_use("t4", "describe_table", table="products"), stop_reason="tool_use"),
    response(tool_use("t5", "run_sql", sql=BAD_SQL, purpose="Revenue by category for March 2026"),
             stop_reason="tool_use"),
    response(text("orders has no total column; revenue comes from order_items. Fixing the query."),
             tool_use("t6", "run_sql", sql=GOOD_SQL, purpose="Revenue by category, completed orders, March 2026"),
             stop_reason="tool_use"),
    "FINAL",  # filled in below from the real query result
]


class ReplayClient(FakeClient):
    """Replays SCRIPTED_MODEL; the final answer is built from the real query result."""

    def __init__(self, delay_s: float = 0.0):
        super().__init__(SCRIPTED_MODEL)
        self.delay_s = delay_s

    def _create(self, **kwargs):
        last = kwargs["messages"][-1]
        if self._responses and self._responses[0] == "FINAL":
            rows = json.loads(last["content"][0]["content"])["rows"]
            top, second = rows[0], rows[1]
            self._responses[0] = response(text(
                f"{top[0]} made the most revenue in March 2026: ${top[1]:,.2f}.\n"
                f"Next was {second[0]} at ${second[1]:,.2f}. Revenue counts completed orders only.\n\n"
                f"```sql\n{GOOD_SQL}\n```"), stop_reason="end_turn")
        if self.delay_s:
            time.sleep(self.delay_s)  # feel like a real model call in the UI
        return super()._create(**kwargs)


class DemoClient(ReplayClient):
    """Prints each request/response so you can watch the conversation grow."""

    def _create(self, **kwargs):
        msgs = kwargs["messages"]
        last = msgs[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            for b in last["content"]:
                tag = "ERROR" if b.get("is_error") else "result"
                body = b["content"]
                print(f"   ← tool {tag} [{b['tool_use_id']}]: {body[:300]}{'…' if len(body) > 300 else ''}")
        resp = super()._create(**kwargs)
        n = len(self.requests)
        print(f"\n── model call {n}  (messages sent: {len(msgs)})  → stop_reason={resp.stop_reason}")
        for b in resp.content:
            if b.type == "text":
                print(f"   model says: {b.text}")
            elif b.type == "tool_use":
                args = json.dumps(b.input)
                print(f"   model calls: {b.name}({args[:160]}{'…' if len(args) > 160 else ''})  [{b.id}]")
        return resp


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    runs_db = os.path.join(tempfile.mkdtemp(), "runs.db")
    cfg = Settings(db_path="data/shop.db", runs_db_path=runs_db, max_steps=10)
    print(f"QUESTION: {QUESTION}\n(model decisions are SCRIPTED; tools, SQL, errors and data are REAL)")

    agent = SQLAgent(client=DemoClient(), cfg=cfg)
    result = agent.run(QUESTION)
    RunStore(runs_db).save(result)

    print("\n════ FINAL ANSWER ════")
    print(result.answer)
    print(f"\n════ RUN SUMMARY ════\nrun_id={result.run_id} status={result.status} steps={len(result.steps)} "
          f"tool_calls={sum(len(s.tool_calls) for s in result.steps)} "
          f"failed_tool_calls={sum(not c.ok for s in result.steps for c in s.tool_calls)}")
    print("\n════ STORED RUN (what GET /runs/{id} returns), trimmed ════")
    stored = RunStore(runs_db).get(result.run_id)
    for s in stored["steps"]:
        calls = ", ".join(f"{c['name']}{'✗' if not c['ok'] else '✓'}" for c in s["tool_calls"]) or "-"
        print(f"step {s['index']}: stop_reason={s['stop_reason']:<9} tools: {calls}")

    print("\n════ BONUS: the safety layer on real queries ════")
    for sql in ["SELECT email FROM customers WHERE id = 1",
                "DELETE FROM orders",
                "SELECT name, country FROM customers WHERE id = 1"]:
        try:
            out = agent.tools.run("run_sql", {"sql": sql, "purpose": "demo"})
            print(f"ALLOWED  {sql}\n         → {out[:120]}")
        except Exception as e:
            print(f"BLOCKED  {sql}\n         → {e}")


if __name__ == "__main__":
    main()
