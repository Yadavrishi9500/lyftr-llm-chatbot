# Lyftr LLM Chatbot: key failover and response caching

A small full-stack chatbot. FastAPI + LiteLLM on the back, React + Vite on the front, SQLite for sessions.
Two things are done carefully: **ordered API-key failover** (with exact accounting of which key served a reply) and **response caching** (cache hits are labelled, free, and fast).

## 1. Live URLS

This repository is not deployed to a public host. The running stack (after `make up`) is:

| What | URL |
| --- | --- |
| Frontend (nginx) | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Health check | http://localhost:8000/health |
| Interactive API docs | http://localhost:8000/docs |

## 2. Run locally: with Docker and without

### With Docker (recommended)

```bash
cp .env.example .env        # then put two real keys in LLM_API_KEYS
make up                     # docker compose up -d --build
# open http://localhost:3000
make down                   # stop
make test                   # backend test suite (needs the local Python setup below)
```

`docker compose` waits for the backend healthcheck before starting the frontend. SQLite lives in the named volume `sqlite_data` mounted at `/data`, so history survives `make down`/`make up`.

### Without Docker

Backend (Python 3.11+):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export LLM_API_KEYS="key-one,key-two"
export DATABASE_URL="sqlite:///./app.db"               # relative path; default is sqlite:////data/app.db
export ENABLE_FAILURE_SIMULATION=true
uvicorn app.main:app --port 8000
```

Frontend (Node 20+):

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000, talks to http://localhost:8000
```

Tests (no network and no real keys needed, `litellm.acompletion` is mocked):

```bash
cd backend && pytest -v
```

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_API_KEYS` | none, required | Comma-separated keys. Ids are `key_1`, `key_2`, ... in order. |
| `LLM_MODEL` | `gemini/gemini-2.0-flash` | Any LiteLLM model string. |
| `DATABASE_URL` | `sqlite:////data/app.db` | `sqlite:////abs/path.db` or `sqlite:///relative/path.db`. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins (read at startup). |
| `ENABLE_FAILURE_SIMULATION` | off | `true` enables the `X-Simulate-Failure` header. |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Build-time API URL for the frontend. |

## 3. Design decisions: key fallback, caching, SQLite session storage

**Key fallback.** `llm_gateway.generate` walks `[(key_1, ...), (key_2, ...), ...]` in order and calls `litellm.acompletion(..., num_retries=0)`. Retries are disabled on purpose: the gateway, not LiteLLM, decides when to move on, so `key_id` and `fallback_attempts` are exact.
- A key is skipped on `429` / quota errors, timeouts (408 and timeout exceptions), connection errors, any `5xx`, and `401`/`403`. A revoked or invalid key is the case a second key exists for.
- Other `4xx` errors, such as a malformed request, do not burn the next key. They return `502 {"detail": "provider rejected the request"}`.
- If every key fails the API returns `503 {"detail": "all providers unavailable"}`.
- Raw keys are only ever passed to LiteLLM. Responses carry `key_1`-style ids. Logs carry the id and the exception *class name*, never exception text, which providers sometimes echo with the credential in it.

**Caching.** `litellm.cache = Cache(type="local")` is configured at startup and `caching=True` is passed on live calls. The gateway also keeps a small in-process LRU keyed by `sha256(model + full ordered message array [system, ...history, new user message])`, checked before any provider call. Reasons:
- Hits are deterministic and never touch a provider or a key. A hit reports `LLM_CACHE`, `key_id: null`, `fallback_attempts: 0`, and the stored per-request token counts.
- Because the key is the whole message array, a different system prompt or different history is correctly a different entry.
- Requests carrying `X-Simulate-Failure` skip the cache entirely (read and write), so failover is exercised even for repeated prompts.

**SQLite session storage.** Stdlib `sqlite3` only, WAL mode, schema created at startup.
- `sessions` holds the per-session `system_prompt`, so `messages` stays a clean transcript (user/assistant only).
- Each turn is persisted in **one transaction** (`BEGIN IMMEDIATE` ... `COMMIT`): session upsert + user row + assistant row with JSON `metadata` and a UTC ISO-8601 `created_at`. A failed or invalid request writes nothing, and validation runs before the database is ever opened.
- History is read `ORDER BY id ASC`. A per-session `asyncio.Lock` serialises read-history, LLM call and persist, so two quick sends in one session cannot interleave turns.
- `input_tokens` / `output_tokens` are the provider's usage for that single request, never a running total.

**Validation.** `session_id` is trimmed to 1-128 characters. `message` is trimmed to 1-4000 characters. Both must be real JSON strings. Anything else, including malformed JSON, is a `422` with a compact `detail` list that does not echo the input back.

## 4. Verification: how to see failover, cache hits and per-message token usage

Set `ENABLE_FAILURE_SIMULATION=true` and use two keys. Everything below can be done from the UI. The header dropdown sets `X-Simulate-Failure` and **New Session** starts a fresh `session_id`.

**Health**

```bash
curl -s localhost:8000/health
# {"status":"ok","keys_configured":2,"cache_enabled":true}
```

**Multi-turn persistence and token usage**

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"demo","message":"What is the capital of France?"}'
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"demo","message":"And its population?"}'
curl -s localhost:8000/sessions/demo/messages
```

Each assistant message carries its own `metadata`, including that request's `input_tokens` / `output_tokens`. The second turn's input count is larger because the history is resent, but it is not cumulative. In the UI, each reply shows source, model, latency and tokens under the message.

**Cache hit**

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"another","message":"What is the capital of France?"}'
# resolution_source: LLM_CACHE, key_id: null, fallback_attempts: 0, same reply and token counts, lower latency_ms
```

**Failover**

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"f1","message":"hello"}'
# 200, key_id: "key_2", fallback_attempts: 1

curl -s -i localhost:8000/chat -H 'content-type: application/json' -H 'X-Simulate-Failure: *=429' \
  -d '{"session_id":"f2","message":"hello"}'
# 503 {"detail":"all providers unavailable"}
curl -s -i localhost:8000/sessions/f2/messages     # 404: the failed turn wrote nothing
```

Header format: comma-separated `<key_id>=<status>` pairs (`key_1=429, key_2=500`) or `*=<status>`, with statuses `429, 500, 502, 503`. It applies only to that request.

**Automated:** `make test` runs 41 tests, covering all six required behaviours plus cache-key isolation, bypass, key-leak and rollback checks.

## 5. Setup Used: your editor and AI tools

Built with Claude (Sonnet 5.5) through Claude Code, driven by a single written specification. Backend verification used `pytest` with a mocked `litellm.acompletion`. Edit this section to list your own editor and tooling if they differ.

## 6. Bonus features (if any): what you added beyond the requirements and why

- **Failure-simulation dropdown in the UI.** Failover and `503` handling can be demoed live without `curl`.
- **Visual source pills and failover badge.** Green `LLM_CACHE`, blue `LIVE_INFERENCE`, and an amber `key_2 · 1 failover` badge, so a reviewer can tell at a glance what happened.
- **Per-session lock.** Prevents interleaved turns from concurrent sends in one session.
- **Two-tier cache.** LiteLLM's cache plus a deterministic gateway LRU, so hits are verifiable without a live provider.
- **Hardened errors.** Compact `422` bodies that never echo input, `502` for non-retryable provider errors that do not burn the second key, and empty provider replies treated as a failover-worthy fault.
- **Non-root, healthchecked containers.** The backend runs as an unprivileged user with a stdlib-based healthcheck, so the frontend only starts once the API is ready.
- **Keys never leave the gateway.** A test asserts raw keys appear in no response body, header or log line, even when the provider's error text contains one.
#   l y f t r - l l m - c h a t b o t  
 #   l y f t r - l l m - c h a t b o t  
 