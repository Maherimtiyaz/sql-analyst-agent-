"""The agent loop, driven by a scripted fake model. No API key, no network, deterministic."""
from __future__ import annotations

import json

from sql_agent.agent import SQLAgent
from sql_agent.tools import TOOLS
from tests.conftest import FakeClient, response, text, tool_use


def test_happy_path_runs_tool_and_answers(cfg):
    client = FakeClient([
        response(tool_use("t1", "run_sql", sql="SELECT COUNT(*) FROM customers", purpose="count"),
                 stop_reason="tool_use"),
        response(text("We have 300 customers."), stop_reason="end_turn"),
    ])
    r = SQLAgent(client=client, cfg=cfg).run("How many customers?")

    assert r.status == "completed"
    assert r.answer == "We have 300 customers."
    assert r.queries == ["SELECT COUNT(*) FROM customers"]
    assert len(r.steps) == 2 and r.input_tokens == 200 and r.output_tokens == 40

    # Second request carries the real tool result back to the model.
    result_msg = client.requests[1]["messages"][-1]
    assert result_msg["role"] == "user"
    block = result_msg["content"][0]
    assert block["tool_use_id"] == "t1" and "is_error" not in block
    assert json.loads(block["content"])["rows"] == [[300]]


def test_request_shape(cfg):
    client = FakeClient([response(text("hi"), stop_reason="end_turn")])
    SQLAgent(client=client, cfg=cfg).run("hello?")
    req = client.requests[0]
    assert req["model"] == cfg.model
    assert req["tools"] == TOOLS
    assert "Revenue =" in req["system"]
    assert req["fallbacks"] == "default"


def test_tool_error_is_returned_and_agent_recovers(cfg):
    client = FakeClient([
        response(tool_use("t1", "run_sql", sql="SELECT total FROM orders", purpose="bad column"),
                 stop_reason="tool_use"),
        response(tool_use("t2", "run_sql", sql="SELECT COUNT(*) FROM orders", purpose="fixed"),
                 stop_reason="tool_use"),
        response(text("2500 orders."), stop_reason="end_turn"),
    ])
    r = SQLAgent(client=client, cfg=cfg).run("How many orders?")
    err = client.requests[1]["messages"][-1]["content"][0]
    assert err["is_error"] is True and "no such column" in err["content"]
    assert r.status == "completed"
    assert [c.ok for s in r.steps for c in s.tool_calls] == [False, True]


def test_blocked_pii_query_is_an_error_not_a_crash(cfg):
    client = FakeClient([
        response(tool_use("t1", "run_sql", sql="SELECT email FROM customers WHERE id=1", purpose="pii"),
                 stop_reason="tool_use"),
        response(text("I can't access customer emails."), stop_reason="end_turn"),
    ])
    r = SQLAgent(client=client, cfg=cfg).run("Email of customer 1?")
    err = client.requests[1]["messages"][-1]["content"][0]
    assert err["is_error"] is True and "restricted" in err["content"]
    assert "@" not in r.answer


def test_parallel_tool_calls_return_in_one_message(cfg):
    client = FakeClient([
        response(tool_use("a", "describe_table", table="orders"),
                 tool_use("b", "describe_table", table="order_items"), stop_reason="tool_use"),
        response(text("done"), stop_reason="end_turn"),
    ])
    SQLAgent(client=client, cfg=cfg).run("schema?")
    msg = client.requests[1]["messages"][-1]
    assert [b["tool_use_id"] for b in msg["content"]] == ["a", "b"]


def test_unknown_tool_is_reported_to_model(cfg):
    client = FakeClient([
        response(tool_use("t1", "drop_everything"), stop_reason="tool_use"),
        response(text("ok"), stop_reason="end_turn"),
    ])
    SQLAgent(client=client, cfg=cfg).run("x")
    err = client.requests[1]["messages"][-1]["content"][0]
    assert err["is_error"] is True and "Unknown tool" in err["content"]


def test_step_limit_stops_runaway_loop(cfg):
    looping = [response(tool_use(f"t{i}", "list_tables"), stop_reason="tool_use") for i in range(cfg.max_steps)]
    r = SQLAgent(client=FakeClient(looping), cfg=cfg).run("loop forever")
    assert r.status == "max_steps" and len(r.steps) == cfg.max_steps


def test_refusal_and_truncation_statuses(cfg):
    r1 = SQLAgent(client=FakeClient([response(stop_reason="refusal")]), cfg=cfg).run("x")
    assert r1.status == "refusal"
    r2 = SQLAgent(client=FakeClient([response(text("partial"), stop_reason="max_tokens")]), cfg=cfg).run("x")
    assert r2.status == "truncated" and r2.answer == "partial"
