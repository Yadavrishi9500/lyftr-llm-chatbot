"""API tests: TestClient + mocked `litellm.acompletion` (no network, no real keys)."""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database, llm_gateway  # noqa: E402
from app.main import app  # noqa: E402

MODEL = "gemini/gemini-2.0-flash"
KEY_1, KEY_2 = "sk-secret-one", "sk-secret-two"
PAIR = {"session_id": "s-1", "message": "What is the capital of France?"}


def make_response(text: str = "The capital of France is Paris.", prompt: int = 42, completion: int = 9):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        _hidden_params={},
    )


def rate_limit_error() -> Exception:
    return litellm.RateLimitError("quota exceeded", llm_provider="gemini", model=MODEL)


def server_error() -> Exception:
    return litellm.InternalServerError("boom", llm_provider="gemini", model=MODEL)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def mock_llm(monkeypatch):
    mock = AsyncMock(side_effect=lambda **kw: make_response())
    monkeypatch.setattr(litellm, "acompletion", mock)
    return mock


@pytest.fixture
def client(monkeypatch, db_path, mock_llm):
    monkeypatch.setenv("LLM_API_KEYS", f"{KEY_1},{KEY_2}")
    monkeypatch.setenv("LLM_MODEL", MODEL)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")  # tmp_path is absolute -> sqlite:////...
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")
    llm_gateway.reset_cache()
    with TestClient(app) as c:
        yield c
    llm_gateway.reset_cache()


def counts(db_path: str) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    try:
        s = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        m = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return s, m
    finally:
        conn.close()


def key_of(call) -> str:
    return call.kwargs["api_key"]


# 1 ---------------------------------------------------------------- health


def test_health_exact_shape(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


def test_health_503_without_keys(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEYS", " , ")
    r = client.get("/health")
    assert r.status_code == 503
    assert "detail" in r.json()


def test_health_503_when_db_unreachable(client, monkeypatch, tmp_path):
    # A directory cannot be opened as a SQLite database file.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}")
    r = client.get("/health")
    assert r.status_code == 503
    assert "detail" in r.json()


# 2 ------------------------------------------------- live then cache


def test_live_then_cache_hit_in_new_session(client, mock_llm):
    first = client.post("/chat", json=PAIR)
    assert first.status_code == 200
    body = first.json()
    assert body["session_id"] == "s-1"
    assert body["reply"] == "The capital of France is Paris."
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == MODEL
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0
    assert (meta["input_tokens"], meta["output_tokens"]) == (42, 9)
    assert isinstance(meta["latency_ms"], int)

    second = client.post("/chat", json={**PAIR, "session_id": "s-2"})
    assert second.status_code == 200
    cached = second.json()
    assert cached["reply"] == body["reply"]
    cmeta = cached["metadata"]
    assert cmeta["resolution_source"] == "LLM_CACHE"
    assert cmeta["key_id"] is None
    assert cmeta["fallback_attempts"] == 0
    assert (cmeta["input_tokens"], cmeta["output_tokens"]) == (42, 9)
    assert mock_llm.await_count == 1  # cache hit never touched the provider


def test_different_system_prompt_is_not_a_cache_hit(client, mock_llm):
    client.post("/chat", json=PAIR)
    r = client.post("/chat", json={**PAIR, "session_id": "s-2", "system_prompt": "Answer in French."})
    assert r.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert mock_llm.await_count == 2


def test_tokens_are_per_request_not_cumulative(client, mock_llm):
    mock_llm.side_effect = lambda **kw: make_response(prompt=10, completion=3)
    client.post("/chat", json=PAIR)
    mock_llm.side_effect = lambda **kw: make_response("Second.", prompt=25, completion=4)
    r = client.post("/chat", json={**PAIR, "message": "And Spain?"})
    meta = r.json()["metadata"]
    assert (meta["input_tokens"], meta["output_tokens"]) == (25, 4)


# 3 ------------------------------------------------ simulated failure


def test_simulate_key_1_429_fails_over_to_key_2(client, mock_llm):
    r = client.post("/chat", json=PAIR, headers={"X-Simulate-Failure": "key_1=429"})
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert [key_of(c) for c in mock_llm.await_args_list] == [KEY_2]


def test_simulate_two_keys_comma_separated_with_spaces(client):
    r = client.post("/chat", json=PAIR, headers={"X-Simulate-Failure": "key_1=429, key_2=500"})
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}


def test_simulate_wildcard_returns_503_and_writes_nothing(client, mock_llm, db_path):
    r = client.post("/chat", json=PAIR, headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert counts(db_path) == (0, 0)
    assert client.get("/sessions/s-1/messages").status_code == 404
    assert mock_llm.await_count == 0


def test_simulate_header_bypasses_cache(client, mock_llm):
    client.post("/chat", json=PAIR)  # warms the cache via key_1
    r = client.post(
        "/chat",
        json={**PAIR, "session_id": "s-2"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    meta = r.json()["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1


def test_simulation_applies_to_single_request_only(client):
    client.post("/chat", json=PAIR, headers={"X-Simulate-Failure": "key_1=500"})
    r = client.post("/chat", json={**PAIR, "session_id": "s-2", "message": "Different question"})
    assert r.json()["metadata"]["key_id"] == "key_1"
    assert r.json()["metadata"]["fallback_attempts"] == 0


def test_simulate_header_ignored_when_flag_disabled(client, monkeypatch, mock_llm):
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "false")
    r = client.post("/chat", json=PAIR, headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_1"
    assert mock_llm.await_count == 1


def test_parse_simulate_header():
    parse = llm_gateway.parse_simulate_header
    assert parse(None) is None
    assert parse("") is None
    assert parse("key_1=429") == {"key_1": 429}
    assert parse("key_1=429, key_2=500") == {"key_1": 429, "key_2": 500}
    assert parse("*=503") == {"*": 503}
    assert parse("key_1=418, nonsense, =500, key_2=abc") is None
    assert parse("key_1=418, key_2=502") == {"key_2": 502}


# 4 -------------------------------------------- real provider failures


@pytest.mark.parametrize("make_error", [rate_limit_error, server_error], ids=["429", "500"])
def test_real_provider_error_on_key_1_fails_over(client, mock_llm, make_error):
    async def side_effect(**kw):
        if kw["api_key"] == KEY_1:
            raise make_error()
        return make_response()

    mock_llm.side_effect = side_effect
    r = client.post("/chat", json=PAIR)
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert [key_of(c) for c in mock_llm.await_args_list] == [KEY_1, KEY_2]
    assert all(c.kwargs["num_retries"] == 0 for c in mock_llm.await_args_list)


def test_timeout_and_connection_errors_fail_over(client, mock_llm):
    async def side_effect(**kw):
        if kw["api_key"] == KEY_1:
            raise litellm.Timeout("slow", model=MODEL, llm_provider="gemini")
        return make_response()

    mock_llm.side_effect = side_effect
    assert client.post("/chat", json=PAIR).json()["metadata"]["fallback_attempts"] == 1

    async def conn_error(**kw):
        if kw["api_key"] == KEY_1:
            raise ConnectionError("refused")
        return make_response()

    mock_llm.side_effect = conn_error
    r = client.post("/chat", json={**PAIR, "session_id": "s-2", "message": "other"})
    assert r.json()["metadata"]["key_id"] == "key_2"


def test_all_real_providers_failing_returns_503_and_persists_nothing(client, mock_llm, db_path):
    async def side_effect(**kw):
        raise rate_limit_error() if kw["api_key"] == KEY_1 else server_error()

    mock_llm.side_effect = side_effect
    r = client.post("/chat", json=PAIR)
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert counts(db_path) == (0, 0)
    assert mock_llm.await_count == 2


def test_non_retryable_error_does_not_burn_second_key(client, mock_llm, db_path):
    mock_llm.side_effect = litellm.BadRequestError("bad", model=MODEL, llm_provider="gemini")
    r = client.post("/chat", json=PAIR)
    assert r.status_code == 502
    assert mock_llm.await_count == 1
    assert counts(db_path) == (0, 0)


def test_failed_turn_leaves_existing_history_untouched(client, mock_llm):
    assert client.post("/chat", json=PAIR).status_code == 200
    mock_llm.side_effect = rate_limit_error()
    r = client.post("/chat", json={**PAIR, "message": "second"}, headers={"X-Simulate-Failure": "*=500"})
    assert r.status_code == 503
    msgs = client.get("/sessions/s-1/messages").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_raw_keys_never_leak(client, mock_llm, caplog):
    caplog.set_level(logging.DEBUG)

    async def side_effect(**kw):
        if kw["api_key"] == KEY_1:
            raise litellm.RateLimitError(f"quota for {KEY_1}", llm_provider="gemini", model=MODEL)
        return make_response()

    mock_llm.side_effect = side_effect
    r = client.post("/chat", json=PAIR)
    blob = r.text + repr(dict(r.headers)) + client.get("/sessions/s-1/messages").text + caplog.text
    assert KEY_1 not in blob and KEY_2 not in blob


# 5 ---------------------------------------------------------- multi-turn


def test_multi_turn_passes_full_ordered_history_and_reads_back(client, mock_llm):
    replies = iter(["Paris.", "About 2.1 million people."])
    mock_llm.side_effect = lambda **kw: make_response(next(replies))

    sid = "s-multi"
    r1 = client.post("/chat", json={"session_id": sid, "message": "Capital of France?", "system_prompt": "Be brief."})
    assert r1.status_code == 200
    r2 = client.post("/chat", json={"session_id": sid, "message": "And its population?"})
    assert r2.status_code == 200

    first_call, second_call = mock_llm.await_args_list
    assert first_call.kwargs["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"},
    ]
    assert second_call.kwargs["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And its population?"},
    ]

    got = client.get(f"/sessions/{sid}/messages")
    assert got.status_code == 200
    data = got.json()
    assert data["session_id"] == sid
    msgs = data["messages"]
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "Capital of France?"),
        ("assistant", "Paris."),
        ("user", "And its population?"),
        ("assistant", "About 2.1 million people."),
    ]
    assert "metadata" not in msgs[0] and "metadata" not in msgs[2]
    assert msgs[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert set(msgs[1]["metadata"]) == {
        "resolution_source", "model", "latency_ms", "input_tokens",
        "output_tokens", "key_id", "fallback_attempts",
    }
    assert not any(m["role"] == "system" for m in msgs)


def test_system_prompt_override_applies_to_later_turns(client, mock_llm):
    sid = "s-sys"
    client.post("/chat", json={"session_id": sid, "message": "hi", "system_prompt": "Pirate."})
    client.post("/chat", json={"session_id": sid, "message": "again", "system_prompt": "Robot."})
    client.post("/chat", json={"session_id": sid, "message": "third"})
    third = mock_llm.await_args_list[2].kwargs["messages"]
    assert third[0] == {"role": "system", "content": "Robot."}
    assert third[-1] == {"role": "user", "content": "third"}


def test_unknown_session_404(client):
    r = client.get("/sessions/does-not-exist/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


def test_message_is_trimmed_before_storage(client):
    client.post("/chat", json={**PAIR, "message": "   padded   "})
    assert client.get("/sessions/s-1/messages").json()["messages"][0]["content"] == "padded"


# 6 ------------------------------------------------------- validation


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s", "message": "   \n\t  "},
        {"session_id": "s", "message": ""},
        {"session_id": "s", "message": "x" * 4001},
        {"session_id": "s", "message": "  " + "x" * 4001 + "  "},
        {"session_id": "s"},
        {"message": "hello"},
        {"session_id": "", "message": "hello"},
        {"session_id": "   ", "message": "hello"},
        {"session_id": "s" * 129, "message": "hello"},
        {"session_id": 123, "message": "hello"},
        {"session_id": "s", "message": 42},
        {},
    ],
)
def test_invalid_requests_return_422_and_persist_nothing(client, mock_llm, db_path, payload):
    r = client.post("/chat", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()
    assert counts(db_path) == (0, 0)
    assert mock_llm.await_count == 0


def test_malformed_json_returns_422(client, db_path):
    r = client.post("/chat", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "detail" in r.json()
    assert counts(db_path) == (0, 0)


def test_boundary_lengths_are_accepted(client):
    assert client.post("/chat", json={"session_id": "s" * 128, "message": "x" * 4000}).status_code == 200
    padded = {"session_id": "s-2", "message": " " * 50 + "y" * 4000 + " " * 50}
    assert client.post("/chat", json=padded).status_code == 200


# ----------------------------------------------------------- misc units


def test_database_url_parsing():
    assert database.parse_database_url("sqlite:////data/app.db") == "/data/app.db"
    assert database.parse_database_url("sqlite:///rel/path.db") == "rel/path.db"
    with pytest.raises(ValueError):
        database.parse_database_url("postgres://x/y")


def test_key_ids_are_deterministic_and_keys_hidden():
    from app.config import load_api_keys

    keys = load_api_keys(" a , b ,, c ")
    assert [k.key_id for k in keys] == ["key_1", "key_2", "key_3"]
    assert [k.api_key for k in keys] == ["a", "b", "c"]
    assert "a" not in repr(keys[0]).replace("ApiKey", "").replace("key_id", "")


def test_schema_and_wal_created_on_startup(client, db_path):
    conn = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"sessions", "messages"} <= tables
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()
