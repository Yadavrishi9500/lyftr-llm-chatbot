"""Pydantic v2 request/response models."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator, model_serializer

MAX_MESSAGE_CHARS = 4000
MAX_SESSION_ID_CHARS = 128
MAX_SYSTEM_PROMPT_CHARS = 8000


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: StrictStr
    message: StrictStr
    system_prompt: Optional[StrictStr] = None

    @field_validator("session_id")
    @classmethod
    def _session_id(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= MAX_SESSION_ID_CHARS:
            raise ValueError(f"session_id must be 1-{MAX_SESSION_ID_CHARS} characters")
        return v

    @field_validator("message")
    @classmethod
    def _message(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= MAX_MESSAGE_CHARS:
            raise ValueError(f"message must be 1-{MAX_MESSAGE_CHARS} characters after trimming")
        return v

    @field_validator("system_prompt")
    @classmethod
    def _system_prompt(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if len(v) > MAX_SYSTEM_PROMPT_CHARS:
            raise ValueError(f"system_prompt must be at most {MAX_SYSTEM_PROMPT_CHARS} characters")
        return v


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str] = None
    fallback_attempts: int = 0


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    metadata: Optional[ChatMetadata] = None

    @model_serializer(mode="wrap")
    def _drop_missing_metadata(self, handler: Any) -> dict[str, Any]:
        # `metadata` appears on assistant messages only; user messages omit the key
        # entirely (while key_id: null inside metadata is preserved).
        data = handler(self)
        if data.get("metadata") is None:
            data.pop("metadata", None)
        return data


class MessagesResponse(BaseModel):
    session_id: str
    messages: list[MessageOut]


class HealthResponse(BaseModel):
    status: str
    keys_configured: int
    cache_enabled: bool
