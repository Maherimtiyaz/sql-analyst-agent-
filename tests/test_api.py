from __future__ import annotations

from fastapi.testclient import TestClient

from sql_agent import api
from sql_agent.agent import SQLAgent
from sql_agent.store import RunStore
from tests.conftest import FakeClient, response, text, tool_use


def make_client(cfg, scripted):
    agent = SQLAgent(client=FakeClient(scripted), cfg=cfg)
    store = RunStore(cfg.runs_db_path)
    api.app.dependency_overrides[api.get_agent] = lambda: agent
    api.app.dependency_overrides[api.get_store] = lambda: store
    return TestClient(api.app)


def teardown_function():
    api.app.dependency_overrides.clear()


def test_ask_then_fetch_run(cfg):
    http = make_client(cfg, [
        response(tool_use("t1", "run_sql", sql="SELECT COUNT(*) FROM products", purpose="count"),
                 stop_reason="tool_use"),
        response(text("There are 20 products."), stop_reason="end_turn"),
    ])
    r = http.post("/ask", json={"question": "How many products?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and body["steps"] == 2
    assert body["queries"] == ["SELECT COUNT(*) FROM products"]

    run = http.get(f"/runs/{body['run_id']}").json()
    assert run["answer"] == "There are 20 products."
    assert run["steps"][0]["tool_calls"][0]["name"] == "run_sql"


def test_unknown_run_is_404(cfg):
    http = make_client(cfg, [])
    assert http.get("/runs/nope").status_code == 404


def test_question_is_validated(cfg):
    http = make_client(cfg, [])
    assert http.post("/ask", json={"question": "x"}).status_code == 422


def parse_sse(body: str) -> list[dict]:
    import json
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_stream_emits_steps_in_order_then_done(cfg):
    http = make_client(cfg, [
        response(text("Checking tables."), tool_use("t1", "list_tables"), stop_reason="tool_use"),
        response(tool_use("t2", "run_sql", sql="SELECT category, COUNT(*) FROM products GROUP BY category",
                          purpose="Products per category"), stop_reason="tool_use"),
        response(text("5 categories."), stop_reason="end_turn"),
    ])
    with http.stream("POST", "/ask/stream", json={"question": "How many categories?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.read().decode())

    types = [e["type"] for e in events]
    assert types == ["run_start", "model", "tool_call", "tool_result", "model", "tool_call", "tool_result",
                     "model", "done"]
    assert events[3]["preview"]["tables"] == ["customers", "order_items", "orders", "products"]
    done = events[-1]
    assert done["status"] == "completed" and done["answer"] == "5 categories."
    assert done["table"]["columns"] == ["category", "COUNT(*)"] and len(done["table"]["rows"]) == 5
    assert http.get(f"/runs/{done['run_id']}").status_code == 200  # stored


def test_stream_reports_tool_errors(cfg):
    http = make_client(cfg, [
        response(tool_use("t1", "run_sql", sql="SELECT nope FROM orders", purpose="bad"), stop_reason="tool_use"),
        response(text("ok"), stop_reason="end_turn"),
    ])
    with http.stream("POST", "/ask/stream", json={"question": "break it"}) as r:
        events = parse_sse(r.read().decode())
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["ok"] is False and "no such column" in result["error"]


def test_stream_turns_crashes_into_error_event(cfg):
    class Boom:
        def run(self, question, on_event=None):
            raise TypeError("Could not resolve authentication method")
    api.app.dependency_overrides[api.get_agent] = lambda: Boom()
    api.app.dependency_overrides[api.get_store] = lambda: RunStore(cfg.runs_db_path)
    with TestClient(api.app).stream("POST", "/ask/stream", json={"question": "anything"}) as r:
        events = parse_sse(r.read().decode())
    assert events == [{"type": "error", "message": "The model call failed: no valid ANTHROPIC_API_KEY. Set it and restart the server."}]


def test_index_page_is_served(cfg):
    http = make_client(cfg, [])
    r = http.get("/")
    assert r.status_code == 200 and "SQL Analyst" in r.text
