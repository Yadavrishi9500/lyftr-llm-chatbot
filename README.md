# Lyftr LLM Chatbot

A full-stack LLM chatbot built with **FastAPI + LiteLLM**, **React + Vite**, and **SQLite**.

The project focuses on two reliability features that are easy to verify:

- **Ordered API-key failover** — automatically falls back to the next configured key when a provider/key is unavailable, while accurately reporting which key served the response.
- **Deterministic response caching** — repeated requests can be served without contacting the LLM provider, reducing latency and API usage.

It also includes persistent multi-turn sessions, per-request token accounting, failure simulation, hardened validation, and Docker-based local deployment.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React, Vite |
| Backend | FastAPI, Python 3.11+ |
| LLM Gateway | LiteLLM |
| Database | SQLite |
| Reverse Proxy | nginx |
| Deployment | Docker Compose |
| Testing | pytest |

---

## 1. Local URLs

This repository is currently intended to run locally.

After running `make up`:

| Service | URL |
| --- | --- |
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Health Check | http://localhost:8000/health |
| Swagger / API Docs | http://localhost:8000/docs |

---

## 2. Getting Started

### Docker — Recommended

Create your environment file and add at least one valid LLM API key:

```bash
cp .env.example .env
```

Set your keys:

```env
LLM_API_KEYS=key-one,key-two
```

Then start the application:

```bash
make up
```

Open:

```text
http://localhost:3000
```

Useful commands:

```bash
make up       # Build and start the stack
make down     # Stop the stack
make test     # Run backend tests
```

Docker Compose waits for the backend health check before starting the frontend.

SQLite data is stored in the `sqlite_data` named volume mounted at `/data`, so chat history survives `make down` / `make up`.

---

### Run Without Docker

#### Backend

Requires **Python 3.11+**.

```bash
cd backend

python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt

export LLM_API_KEYS="key-one,key-two"
export DATABASE_URL="sqlite:///./app.db"
export ENABLE_FAILURE_SIMULATION=true

uvicorn app.main:app --port 8000
```

#### Frontend

Requires **Node.js 20+**.

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at:

```text
http://localhost:3000
```

and communicates with:

```text
http://localhost:8000
```

#### Tests

Tests do not require network access or real API keys because `litellm.acompletion` is mocked.

```bash
cd backend
pytest -v
```

---

## 3. Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_API_KEYS` | Required | Comma-separated API keys. Internally identified as `key_1`, `key_2`, etc. |
| `LLM_MODEL` | `gemini/gemini-2.0-flash` | Any supported LiteLLM model string. |
| `DATABASE_URL` | `sqlite:////data/app.db` | SQLite database URL. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins. |
| `ENABLE_FAILURE_SIMULATION` | `false` | Enables the `X-Simulate-Failure` test header. |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend API URL, evaluated at build time. |

---

# 4. Architecture & Design

## Ordered API-Key Failover

The gateway receives API keys in a defined order:

```text
key_1 → key_2 → ... → failure
```

`llm_gateway.generate()` tries each configured key sequentially.

LiteLLM is called with:

```python
num_retries=0
```

Retries are deliberately disabled at the LiteLLM layer. The application itself controls failover, which makes the following metadata deterministic:

```json
{
  "key_id": "key_2",
  "fallback_attempts": 1
}
```

### Errors That Trigger Failover

The next key is attempted for:

- `429` / quota or rate-limit errors
- `408` / timeout errors
- timeout exceptions
- connection errors
- `5xx` provider failures
- `401` / `403` authentication failures

Invalid or revoked credentials therefore automatically fall through to the next configured key.

### Non-Retryable Errors

Other `4xx` responses, such as malformed provider requests, do **not** consume another key.

They return:

```json
{
  "detail": "provider rejected the request"
}
```

with HTTP `502`.

If every configured key fails:

```json
{
  "detail": "all providers unavailable"
}
```

is returned with HTTP `503`.

### Key Safety

Raw API keys never leave the LLM gateway.

Only safe identifiers such as:

```text
key_1
key_2
```

appear in responses and logs.

Logs record the key ID and exception **class name**, but never provider exception text, since provider errors may contain credentials.

---

## Response Caching

The application uses two cache layers:

```text
Request
   │
   ▼
Gateway LRU Cache
   │ miss
   ▼
LiteLLM Cache
   │ miss
   ▼
Provider
```

LiteLLM caching is enabled at startup:

```python
litellm.cache = Cache(type="local")
```

and live calls use:

```python
caching=True
```

The gateway additionally maintains a small deterministic in-process LRU.

### Cache Key

Entries are keyed using:

```text
SHA-256(
    model
    +
    complete ordered message array
)
```

The message array includes:

```text
system prompt
+ previous conversation history
+ new user message
```

This prevents incorrect cache sharing between conversations with different prompts or histories.

### Cache Hits

A cache hit:

- Makes **no provider request**
- Uses **no API key**
- Reports `key_id: null`
- Reports `fallback_attempts: 0`
- Preserves the original request's token counts
- Returns significantly lower latency

Example metadata:

```json
{
  "resolution_source": "LLM_CACHE",
  "key_id": null,
  "fallback_attempts": 0
}
```

Requests using `X-Simulate-Failure` bypass both cache reads and writes so failover behaviour can always be tested reliably.

---

## SQLite Session Persistence

The application uses Python's standard `sqlite3` module with **WAL mode** enabled.

The schema is automatically created during startup.

### Sessions

`sessions` stores session-level data such as the `system_prompt`.

The `messages` table therefore remains a clean transcript containing only:

```text
user
assistant
```

messages.

### Atomic Turns

Each successful chat turn is stored in a single transaction:

```text
BEGIN IMMEDIATE

  session upsert
  + user message
  + assistant message
  + assistant metadata

COMMIT
```

If generation or validation fails, nothing from that turn is persisted.

Assistant metadata includes information such as:

```text
source
model
key_id
fallback_attempts
input_tokens
output_tokens
latency
created_at
```

Timestamps use UTC ISO-8601 format.

### Conversation Ordering

History is loaded using:

```sql
ORDER BY id ASC
```

A per-session `asyncio.Lock` serializes:

```text
read history
    ↓
LLM request
    ↓
persist turn
```

This prevents two simultaneous requests in the same session from interleaving conversation turns.

Token counts always represent the **individual provider request**, not a cumulative session total.

---

## Input Validation

The API validates input before opening the database.

### `session_id`

- Must be a JSON string
- Trimmed before validation
- Length: **1–128 characters**

### `message`

- Must be a JSON string
- Trimmed before validation
- Length: **1–4000 characters**

Malformed JSON, incorrect types, and invalid lengths return HTTP `422`.

Validation responses use a compact `detail` structure and never echo user input.

---

# 5. Verification

The important reliability behaviours can be tested directly through the API.

Enable failure simulation:

```env
ENABLE_FAILURE_SIMULATION=true
```

For failover testing, configure at least two keys.

The frontend also exposes failure simulation through a dropdown, so these behaviours can be demonstrated without `curl`.

---

## Health Check

```bash
curl -s localhost:8000/health
```

Example:

```json
{
  "status": "ok",
  "keys_configured": 2,
  "cache_enabled": true
}
```

---

## Multi-Turn Persistence

Send the first message:

```bash
curl -s localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"demo","message":"What is the capital of France?"}'
```

Continue the same conversation:

```bash
curl -s localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"demo","message":"And its population?"}'
```

Inspect the persisted transcript:

```bash
curl -s localhost:8000/sessions/demo/messages
```

Each assistant message contains metadata for its own request, including:

```text
input_tokens
output_tokens
```

The second request normally has a larger input-token count because previous conversation history is sent back to the model.

These counts are **per request**, not cumulative.

---

## Verify a Cache Hit

Make the same effective request from another session:

```bash
curl -s localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"another","message":"What is the capital of France?"}'
```

A successful cache hit reports:

```json
{
  "resolution_source": "LLM_CACHE",
  "key_id": null,
  "fallback_attempts": 0
}
```

The reply and token counts are preserved while latency is substantially lower.

---

## Verify Key Failover

Force `key_1` to behave as if it received a `429`:

```bash
curl -s localhost:8000/chat \
  -H 'content-type: application/json' \
  -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"f1","message":"hello"}'
```

Expected result:

```json
{
  "key_id": "key_2",
  "fallback_attempts": 1
}
```

with HTTP `200`.

---

## Verify Total Provider Failure

Force every key to fail:

```bash
curl -s -i localhost:8000/chat \
  -H 'content-type: application/json' \
  -H 'X-Simulate-Failure: *=429' \
  -d '{"session_id":"f2","message":"hello"}'
```

Expected:

```text
HTTP 503
```

```json
{
  "detail": "all providers unavailable"
}
```

Confirm that the failed turn was not persisted:

```bash
curl -s -i localhost:8000/sessions/f2/messages
```

Expected:

```text
HTTP 404
```

### Failure Simulation Format

The header accepts comma-separated key/status pairs:

```text
X-Simulate-Failure: key_1=429,key_2=500
```

or all keys:

```text
X-Simulate-Failure: *=503
```

Supported simulated statuses:

```text
429, 500, 502, 503
```

Simulation applies only to the current request.

---

# 6. Testing

Run the complete backend test suite with:

```bash
make test
```

The project includes **41 automated tests** covering the required behaviours and additional reliability cases, including:

- Ordered key failover
- Complete provider failure
- Cache hits
- Cache-key isolation
- Cache bypass during failure simulation
- Multi-turn persistence
- Per-message token accounting
- Input validation
- Transaction rollback
- API-key leak prevention

Provider calls are mocked, so the test suite does not require network access or real API credentials.

---

# 7. UI Features

The React frontend exposes the backend's reliability behaviour directly.

### Failure Simulation

A dropdown allows provider failures to be simulated without manually constructing HTTP requests.

### Response Metadata

Every assistant reply displays useful metadata such as:

```text
source • model • latency • token usage
```

Visual badges make the response path immediately visible:

```text
🟢 LLM_CACHE
🔵 LIVE_INFERENCE
🟠 key_2 · 1 failover
```

This makes caching and key failover easy to demonstrate during review.

---

# 8. Additional Engineering Features

Beyond the core chatbot requirements, the project includes:

**Failure simulation**  
Provider failures can be triggered from the UI or `X-Simulate-Failure` header, making failover reproducible without waiting for a real provider outage.

**Two-tier caching**  
LiteLLM's built-in cache is combined with a deterministic gateway LRU, allowing cache behaviour to be tested independently of a live provider.

**Per-session concurrency control**  
An `asyncio.Lock` prevents concurrent requests from corrupting conversational ordering.

**Atomic persistence**  
A chat turn is written only after successful generation. Failed requests leave no partial database state.

**Hardened error handling**  
Non-retryable provider errors return a controlled `502`, exhausted providers return `503`, and empty provider responses are treated as failover-worthy failures.

**Credential isolation**  
Raw API keys remain inside the gateway. Automated tests verify that credentials cannot appear in response bodies, headers, or logs.

**Healthchecked containers**  
The backend runs as an unprivileged container user and exposes a health check. The frontend starts only after the API becomes healthy.

---

# 9. Development Tools

The project was built with **Claude Code using Claude Sonnet 5.5**, based on a written project specification.

Backend behaviour was verified using **pytest**, with `litellm.acompletion` mocked to provide deterministic tests without consuming real API quota.

---

## Summary

Lyftr demonstrates a small but production-minded LLM application where reliability behaviour is explicit and testable:

```text
                 ┌───────────────┐
                 │ React + Vite  │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │    FastAPI    │
                 └───────┬───────┘
                         │
               ┌─────────┴─────────┐
               │                   │
               ▼                   ▼
        ┌─────────────┐      ┌───────────┐
        │ Cache Layer │      │  SQLite   │
        └──────┬──────┘      │ Sessions  │
               │ miss        └───────────┘
               ▼
        ┌─────────────┐
        │ LLM Gateway │
        └──────┬──────┘
               │
         key_1 │ → failure
               ▼
         key_2 │ → success
               ▼
        ┌─────────────┐
        │   LiteLLM   │
        │  Provider   │
        └─────────────┘
```

The result is a chatbot with **deterministic caching, observable failover, persistent conversations, accurate per-request accounting, and reproducible failure testing**.
