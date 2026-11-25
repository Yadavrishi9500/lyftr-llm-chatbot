"""LiteLLM gateway: explicit key failover + response caching.

Caching
    * `litellm.cache = Cache(type="local")` is configured (and `caching=True`
      is passed per call) so LiteLLM's own cache is active.
    * On top of that the gateway keeps a small in-process LRU keyed by
      `model + full ordered message array`. It is checked *before* any provider
      call, which makes cache hits deterministic and independent of which
      provider/key would have served the request.

Failover
    Keys are tried strictly in order with `num_retries=0`, so this module (not
    LiteLLM's router) decides when to move on and can report the exact
    `key_id` and `fallback_attempts`.

Secrets
    Raw keys are only handed to `litellm.acompletion`. Logs carry key ids and
    exception class names, never exception text (which can echo credentials).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import litellm
from litellm import Cache

from .config import ApiKey

logger = logging.getLogger("lyftr.gateway")

litellm.suppress_debug_info = True

REQUEST_TIMEOUT_SECONDS = 30
CACHE_MAX_ENTRIES = 1000
SIMULATABLE_STATUSES = frozenset({429, 500, 502, 503})

LIVE_INFERENCE = "LIVE_INFERENCE"
LLM_CACHE = "LLM_CACHE"


class AllProvidersUnavailable(Exception):
    """Every configured key failed with a failover-eligible error."""


class ProviderRejected(Exception):
    """Provider refused the request itself (e.g. 400); another key would not help."""


class EmptyProviderReply(Exception):
    """Provider answered 200 with no text; treated like a transient provider fault."""


class SimulatedProviderError(Exception):
    """Raised by the `X-Simulate-Failure` test hook in place of a provider call."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"simulated provider failure {status_code}")
        self.status_code = status_code


@dataclass(frozen=True)
class GatewayResult:
    reply: str
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


@dataclass(frozen=True)
class _CachedReply:
    reply: str
    input_tokens: int
    output_tokens: int


_reply_cache: "OrderedDict[str, _CachedReply]" = OrderedDict()


def configure_cache() -> None:
    """Enable LiteLLM's built-in in-memory cache."""
    litellm.cache = Cache(type="local")


def reset_cache() -> None:
    """Drop every cached reply (used by tests)."""
    _reply_cache.clear()
    configure_cache()


def cache_enabled() -> bool:
    return litellm.cache is not None


def cache_key(model: str, messages: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(
        {"model": model, "messages": [{"role": m["role"], "content": m["content"]} for m in messages]},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> _CachedReply | None:
    hit = _reply_cache.get(key)
    if hit is not None:
        _reply_cache.move_to_end(key)
    return hit


def _cache_put(key: str, value: _CachedReply) -> None:
    _reply_cache[key] = value
    _reply_cache.move_to_end(key)
    while len(_reply_cache) > CACHE_MAX_ENTRIES:
        _reply_cache.popitem(last=False)


def parse_simulate_header(value: str | None) -> dict[str, int] | None:
    """Parse `key_1=429, key_2=500` / `*=429` into {key_id_or_*: status}.

    Entries with an unknown shape or a status outside {429,500,502,503} are
    ignored. Returns None when the header is absent or yields no valid entry.
    """
    if not value:
        return None
    rules: dict[str, int] = {}
    for part in value.split(","):
        target, sep, status = part.strip().partition("=")
        target, status = target.strip(), status.strip()
        if not sep or not target or not status.isdigit():
            continue
        if int(status) in SIMULATABLE_STATUSES:
            rules[target] = int(status)
    return rules or None


_FAILOVER_CLASS_NAMES = frozenset(
    {
        "RateLimitError",
        "Timeout",
        "APITimeoutError",
        "APIConnectionError",
        "ServiceUnavailableError",
        "InternalServerError",
        "BadGatewayError",
    }
)


def is_failover_error(exc: BaseException) -> bool:
    """True for errors where trying the next key is the right move.

    Covers 429 / quota, timeouts (408), connection errors, any 5xx, and
    401/403 (a revoked or invalid key is exactly what a second key is for).
    """
    if isinstance(exc, (SimulatedProviderError, EmptyProviderReply)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (401, 403, 408, 429) or 500 <= status <= 599
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    if type(exc).__name__ in _FAILOVER_CLASS_NAMES:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in ("quota", "rate limit", "resource_exhausted"))


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_response(response: Any) -> tuple[str, int, int, bool]:
    choices = _field(response, "choices") or []
    content = None
    if choices:
        content = _field(_field(choices[0], "message"), "content")
    if not isinstance(content, str) or not content.strip():
        raise EmptyProviderReply()
    usage = _field(response, "usage")
    input_tokens = int(_field(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(_field(usage, "completion_tokens", 0) or 0)
    hidden = _field(response, "_hidden_params") or {}
    litellm_cache_hit = bool(hidden.get("cache_hit")) if isinstance(hidden, Mapping) else False
    return content, input_tokens, output_tokens, litellm_cache_hit


async def generate(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    keys: Sequence[ApiKey],
    simulate: Mapping[str, int] | None = None,
) -> GatewayResult:
    """Return a reply from cache or, failing that, from the first healthy key.

    `simulate` (already parsed, only supplied when the feature flag is on)
    forces failures for matching key ids and bypasses the cache entirely for
    this call so failover is exercised even for repeated prompts.
    """
    started = time.perf_counter()
    bypass_cache = simulate is not None
    ckey = cache_key(model, messages)

    def elapsed_ms() -> int:
        return max(0, int(round((time.perf_counter() - started) * 1000)))

    if not bypass_cache:
        cached = _cache_get(ckey)
        if cached is not None:
            return GatewayResult(
                reply=cached.reply,
                resolution_source=LLM_CACHE,
                model=model,
                latency_ms=elapsed_ms(),
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                key_id=None,
                fallback_attempts=0,
            )

    fallback_attempts = 0
    for key in keys:
        try:
            if simulate is not None:
                status = simulate.get(key.key_id, simulate.get("*"))
                if status is not None:
                    raise SimulatedProviderError(status)
            response = await litellm.acompletion(
                model=model,
                messages=list(messages),
                api_key=key.api_key,
                num_retries=0,
                timeout=REQUEST_TIMEOUT_SECONDS,
                caching=not bypass_cache,
            )
            reply, input_tokens, output_tokens, litellm_hit = _parse_response(response)
        except Exception as exc:  # noqa: BLE001 - classified below
            status = getattr(exc, "status_code", None)
            if not is_failover_error(exc):
                logger.error(
                    "provider rejected request key_id=%s error=%s status=%s",
                    key.key_id,
                    type(exc).__name__,
                    status,
                )
                raise ProviderRejected() from None
            fallback_attempts += 1
            logger.warning(
                "failover key_id=%s error=%s status=%s attempts=%d",
                key.key_id,
                type(exc).__name__,
                status,
                fallback_attempts,
            )
            continue

        if not bypass_cache:
            _cache_put(ckey, _CachedReply(reply, input_tokens, output_tokens))
        if litellm_hit:
            return GatewayResult(
                reply=reply,
                resolution_source=LLM_CACHE,
                model=model,
                latency_ms=elapsed_ms(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                key_id=None,
                fallback_attempts=0,
            )
        return GatewayResult(
            reply=reply,
            resolution_source=LIVE_INFERENCE,
            model=model,
            latency_ms=elapsed_ms(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            key_id=key.key_id,
            fallback_attempts=fallback_attempts,
        )

    raise AllProvidersUnavailable()
