"""Environment-driven configuration.

Settings are read from the environment on every call (no import-time snapshot)
so tests and operators can change them without re-importing modules.
Raw API keys live only inside `ApiKey.api_key` and are never repr'd or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "gemini/gemini-2.0-flash"
DEFAULT_DATABASE_URL = "sqlite:////data/app.db"
DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000"


@dataclass(frozen=True)
class ApiKey:
    """One provider credential with a deterministic public identifier."""

    key_id: str
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[ApiKey, ...]
    model: str
    database_url: str
    allowed_origins: tuple[str, ...]
    enable_failure_simulation: bool


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_api_keys(raw: str | None) -> tuple[ApiKey, ...]:
    """`a,b,c` -> key_1=a, key_2=b, key_3=c (blank entries are skipped)."""
    parts = _split_csv(raw or "")
    return tuple(ApiKey(key_id=f"key_{i}", api_key=p) for i, p in enumerate(parts, start=1))


def get_settings() -> Settings:
    return Settings(
        api_keys=load_api_keys(os.environ.get("LLM_API_KEYS")),
        model=os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL,
        database_url=os.environ.get("DATABASE_URL", "").strip() or DEFAULT_DATABASE_URL,
        allowed_origins=tuple(
            _split_csv(os.environ.get("ALLOWED_ORIGINS", "") or DEFAULT_ALLOWED_ORIGINS)
        ),
        enable_failure_simulation=os.environ.get("ENABLE_FAILURE_SIMULATION", "")
        .strip()
        .lower()
        == "true",
    )
