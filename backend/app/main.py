"""FastAPI application: /health, /chat, /sessions/{session_id}/messages."""

from __future__ import annotations

import asyncio
import logging
import weakref
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import database, llm_gateway
from .config import get_settings
from .schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MessagesResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("lyftr.api")

SIMULATE_HEADER = "x-simulate-failure"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    database.init_db()
    llm_gateway.configure_cache()
    logger.info("startup complete keys_configured=%d", len(get_settings().api_keys))
    yield


app = FastAPI(title="Lyftr LLM Chatbot", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().allowed_origins),
    allow_methods=["*"],
    allow_headers=["*"],  # includes X-Simulate-Failure
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with a compact, always-serialisable `detail`; never echoes the input."""
    detail: list[dict[str, Any]] = [
        {"loc": list(err.get("loc", ())), "msg": str(err.get("msg", "")), "type": str(err.get("type", ""))}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


# One lock per session id serialises read-history -> LLM -> persist for that
# session, so concurrent sends cannot interleave and corrupt turn order.
_session_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    if not await asyncio.to_thread(database.ping):
        raise HTTPException(status_code=503, detail="database unreachable")
    if len(settings.api_keys) < 1:
        raise HTTPException(status_code=503, detail="no API keys configured")
    return HealthResponse(
        status="ok",
        keys_configured=len(settings.api_keys),
        cache_enabled=llm_gateway.cache_enabled(),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    settings = get_settings()
    if not settings.api_keys:
        raise HTTPException(status_code=503, detail="all providers unavailable")

    simulate = None
    if settings.enable_failure_simulation:
        simulate = llm_gateway.parse_simulate_header(request.headers.get(SIMULATE_HEADER))

    session_id = body.session_id
    async with _lock_for(session_id):
        state = await asyncio.to_thread(database.load_session, session_id)

        # Request prompt overrides the stored one; blank means "no system prompt".
        effective_prompt = body.system_prompt if body.system_prompt is not None else state.system_prompt
        messages: list[dict[str, str]] = []
        if effective_prompt:
            messages.append({"role": "system", "content": effective_prompt})
        messages.extend(state.history)
        messages.append({"role": "user", "content": body.message})

        try:
            result = await llm_gateway.generate(
                model=settings.model,
                messages=messages,
                keys=settings.api_keys,
                simulate=simulate,
            )
        except llm_gateway.AllProvidersUnavailable:
            raise HTTPException(status_code=503, detail="all providers unavailable") from None
        except llm_gateway.ProviderRejected:
            raise HTTPException(status_code=502, detail="provider rejected the request") from None

        metadata = ChatMetadata(
            resolution_source=result.resolution_source,  # type: ignore[arg-type]
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )
        # Only a valid reply reaches this point; persist everything atomically.
        await asyncio.to_thread(
            database.persist_exchange,
            session_id,
            body.message,
            result.reply,
            metadata.model_dump(),
            body.system_prompt,
        )

    return ChatResponse(session_id=session_id, reply=result.reply, metadata=metadata)


@app.get("/sessions/{session_id}/messages", response_model=MessagesResponse)
async def get_session_messages(session_id: str) -> MessagesResponse:
    rows = await asyncio.to_thread(database.get_messages, session_id)
    if rows is None:
        raise HTTPException(status_code=404, detail="session not found")
    return MessagesResponse(session_id=session_id, messages=rows)  # type: ignore[arg-type]
