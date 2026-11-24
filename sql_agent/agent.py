"""The agent loop: call model -> run tools -> append results -> repeat until done.

This is the same loop as section 8 of the Agent Field Guide, plus the things that
make it production-shaped: a run id, per-step traces, token accounting, explicit
handling of every stop_reason, and a hard step limit.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .config import Settings, settings as default_settings
from .db import QueryError, ReadOnlyDB
from .tools import TOOLS, ToolRunner

log = logging.getLogger("sql_agent")

SYSTEM_PROMPT = """You are a careful data analyst for an online shop. You answer business questions by querying a read-only SQLite database with your tools.

How to work:
- Start with list_tables, then describe_table for each table you need. Never guess column names.
- Write SQLite SQL. Aggregate in SQL; don't pull raw rows to count them yourself.
- If a query errors, read the error, fix the query, and try again.
- Before answering, sanity-check the numbers (e.g. a share can't exceed 100%, totals should match their parts).

Business definitions (use these unless the user says otherwise):
- Money is stored in cents. Report dollars with 2 decimals.
- Revenue = SUM(order_items.quantity * order_items.unit_price_cents) for orders with status = 'completed'.
- An order's date is orders.created_at (ISO 8601 text, UTC).

Answer format:
- First line: the direct answer, with the number(s).
- Then 1-3 short lines of context if useful.
- Then the final SQL you relied on, in a ```sql block.
- If the data cannot answer the question, or it needs restricted data such as customer emails, say so plainly. Never invent numbers."""


@dataclass
class ToolCallTrace:
    name: str
    args: dict[str, Any]
    ok: bool
    ms: int
    error: str | None = None


@dataclass
class StepTrace:
    index: int
    model_ms: int
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolCallTrace] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    question: str
    status: str  # completed | max_steps | refusal | truncated | error
    answer: str
    queries: list[str]
    steps: list[StepTrace]
    input_tokens: int
    output_tokens: int
    duration_ms: int
    model: str
    table: dict[str, Any] | None = None  # last successful run_sql result, for the UI

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SQLAgent:
    def __init__(self, client: Any = None, tools: ToolRunner | None = None, cfg: Settings = default_settings):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        self.cfg = cfg
        self.tools = tools or ToolRunner(
            ReadOnlyDB(cfg.db_path, max_rows=cfg.max_rows, timeout_s=cfg.query_timeout_s),
            max_result_chars=cfg.max_result_chars,
        )

    def _call_model(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            # If the model declines a request, the API retries it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def run(self, question: str, on_event: Callable[[dict[str, Any]], None] | None = None) -> RunResult:
        """Run the loop. `on_event` receives progress events (used for SSE streaming)."""
        emit = on_event or (lambda _e: None)
        run_id = uuid.uuid4().hex[:12]
        emit({"type": "run_start", "run_id": run_id, "question": question})
        started = time.monotonic()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        steps: list[StepTrace] = []
        queries: list[str] = []
        tokens_in = tokens_out = 0
        last_table: dict[str, Any] | None = None

        def finish(st: str, ans: str) -> RunResult:
            result = RunResult(run_id, question, st, ans, queries, steps, tokens_in, tokens_out,
                               int((time.monotonic() - started) * 1000), self.cfg.model, last_table)
            log.info(json.dumps({"event": "run_end", "run_id": run_id, "status": st, "steps": len(steps),
                                 "input_tokens": tokens_in, "output_tokens": tokens_out,
                                 "duration_ms": result.duration_ms}))
            return result

        for i in range(1, self.cfg.max_steps + 1):
            t0 = time.monotonic()
            resp = self._call_model(messages)
            usage = getattr(resp, "usage", None)
            step = StepTrace(i, int((time.monotonic() - t0) * 1000), resp.stop_reason,
                             getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)
            tokens_in += step.input_tokens
            tokens_out += step.output_tokens
            steps.append(step)
            # Append the full content (text, thinking, tool_use blocks) unchanged.
            messages.append({"role": "assistant", "content": resp.content})
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            emit({"type": "model", "step": i, "text": text, "stop_reason": resp.stop_reason,
                  "ms": step.model_ms})

            if resp.stop_reason == "end_turn":
                return finish("completed", text)
            if resp.stop_reason == "refusal":
                return finish("refusal", "The model declined to answer this request.")
            if resp.stop_reason == "max_tokens":
                return finish("truncated", text)
            if resp.stop_reason != "tool_use":
                return finish("error", f"Unexpected stop_reason: {resp.stop_reason}")

            # Run every tool call from this turn; send all results back in ONE user message.
            results = []
            for block in (b for b in resp.content if b.type == "tool_use"):
                t1 = time.monotonic()
                args = dict(block.input or {})
                if block.name == "run_sql" and "sql" in args:
                    queries.append(args["sql"])
                emit({"type": "tool_call", "step": i, "id": block.id, "name": block.name, "args": args})
                try:
                    content, is_error, err = self.tools.run(block.name, args), False, None
                except QueryError as e:
                    content, is_error, err = f"Error: {e}", True, str(e)
                except Exception as e:  # a bug in our code: tell the model, keep the trace
                    log.exception("tool crashed")
                    content, is_error, err = f"Internal tool error: {type(e).__name__}", True, repr(e)
                ms = int((time.monotonic() - t1) * 1000)
                step.tool_calls.append(ToolCallTrace(block.name, args, not is_error, ms, err))
                preview = None if is_error else _preview(block.name, content)
                if block.name == "run_sql" and preview and "columns" in preview:
                    last_table = preview
                emit({"type": "tool_result", "step": i, "id": block.id, "name": block.name, "ok": not is_error,
                      "ms": ms, "error": err, "preview": preview})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content,
                                 **({"is_error": True} if is_error else {})})
            log.info(json.dumps({"event": "step", "run_id": run_id, **asdict(step)}, default=str))
            messages.append({"role": "user", "content": results})

        return finish("max_steps", "Stopped: reached the step limit before finishing.")


def _preview(tool: str, content: str) -> dict[str, Any] | None:
    """A small, UI-friendly summary of a tool result."""
    try:
        data = json.loads(content)
    except ValueError:  # result was cut off for size
        return {"note": "large result (cut off)"}
    if tool == "list_tables":
        return {"tables": [t["table"] for t in data]}
    if tool == "describe_table":
        return {"table": data["table"], "columns": [c["name"] for c in data["columns"]]}
    if tool == "run_sql":
        return {"columns": data["columns"], "rows": data["rows"][:20], "row_count": data["row_count"],
                "truncated": data["truncated"], "elapsed_ms": data["elapsed_ms"]}
    return None
