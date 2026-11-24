from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.seed import main as seed
from sql_agent.config import Settings
from sql_agent.db import ReadOnlyDB


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "shop.db")
    seed(path)
    return path


@pytest.fixture
def db(db_path) -> ReadOnlyDB:
    return ReadOnlyDB(db_path, max_rows=100, timeout_s=1.0)


@pytest.fixture
def cfg(db_path, tmp_path) -> Settings:
    return Settings(db_path=db_path, runs_db_path=str(tmp_path / "runs.db"), max_steps=4)


# ---------- a scripted stand-in for the Anthropic client ----------
def text(t: str):
    return SimpleNamespace(type="text", text=t)


def tool_use(id_: str, name: str, **inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def response(*blocks, stop_reason: str, tin: int = 100, tout: int = 20):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=tin, output_tokens=tout))


class FakeClient:
    """Returns pre-scripted responses and records every request it receives."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # snapshot messages: the agent keeps appending to the same list
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._responses.pop(0)
