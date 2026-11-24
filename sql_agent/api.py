"""HTTP API + web UI.

Run:            uvicorn sql_agent.api:app --reload        then open http://localhost:8000
Demo (no key):  SQL_AGENT_DEMO=1 uvicorn sql_agent.api:app  (replays a scripted model run)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import RunResult, SQLAgent
from .config import settings
from .store import RunStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("sql_agent.api")
app = FastAPI(title="SQL Analyst Agent")

DEMO = os.getenv("SQL_AGENT_DEMO") == "1"
STATIC = Path(__file__).with_name("static")

_agent: SQLAgent | None = None
_store: RunStore | None = None


def get_agent() -> SQLAgent:
    global _agent
    if DEMO:  # fresh scripted client per request; tools and database are real
        from scripts.demo_offline import ReplayClient
        return SQLAgent(client=ReplayClient(delay_s=0.7), cfg=settings)
    if _agent is None:
        _agent = SQLAgent()
    return _agent


def get_store() -> RunStore:
    global _store
    if _store is None:
        _store = RunStore(settings.runs_db_path)
    return _store


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    run_id: str
    status: str
    answer: str
    queries: list[str]
    steps: int
    input_tokens: int
    output_tokens: int
    duration_ms: int


def _summary(r: RunResult) -> dict[str, Any]:
    return {"run_id": r.run_id, "status": r.status, "answer": r.answer, "queries": r.queries,
            "steps": len(r.steps), "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "duration_ms": r.duration_ms}


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    if "authentication" in msg.lower() or type(e).__name__ == "AuthenticationError":
        return "The model call failed: no valid ANTHROPIC_API_KEY. Set it and restart the server."
    return f"The agent failed: {type(e).__name__}: {msg[:300]}"


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, agent: SQLAgent = Depends(get_agent), store: RunStore = Depends(get_store)) -> AskResponse:
    # Sync endpoint: FastAPI runs it in a thread pool, so the blocking loop doesn't stall the event loop.
    result = agent.run(req.question)
    store.save(result)
    return AskResponse(**_summary(result))


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, agent: SQLAgent = Depends(get_agent),
                     store: RunStore = Depends(get_store)) -> StreamingResponse:
    """Same as /ask, but streams progress as Server-Sent Events.

    The agent loop is blocking, so it runs in a worker thread and pushes events into
    an asyncio.Queue; this async generator drains the queue onto the HTTP response.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def emit(event: dict[str, Any] | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def work() -> None:
        try:
            result = agent.run(req.question, on_event=emit)
            store.save(result)
            emit({"type": "done", **_summary(result), "table": result.table})
        except Exception as e:  # never leave the browser hanging
            log.exception("agent run failed")
            emit({"type": "error", "message": _friendly_error(e)})
        finally:
            emit(None)

    async def events():
        worker = loop.run_in_executor(None, work)
        while (event := await queue.get()) is not None:
            yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
        await worker

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/runs/{run_id}")
def get_run(run_id: str, store: RunStore = Depends(get_store)) -> dict:
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No run with id {run_id}")
    return run


@app.get("/health")
def health() -> dict:
    return {"ok": True, "demo": DEMO, "model": settings.model}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
