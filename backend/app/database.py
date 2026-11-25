"""SQLite persistence (stdlib `sqlite3` only).

Every helper opens its own short-lived connection, so helpers are safe to run
in worker threads via `asyncio.to_thread`. Writes for one chat turn happen in a
single explicit transaction: either session + user + assistant rows all land,
or nothing does.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    system_prompt TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, e.g. 2026-11-25T14:04:05.123Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_database_url(url: str) -> str:
    """Return the filesystem path for `sqlite:////abs/path.db` or `sqlite:///rel/path.db`."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise ValueError("DATABASE_URL must start with sqlite:///")
    path = url[len(prefix):]
    if not path:
        raise ValueError("DATABASE_URL has no file path")
    # sqlite:////abs/path.db -> "/abs/path.db"; sqlite:///rel/path.db -> "rel/path.db"
    return path


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = parse_database_url(get_settings().database_url)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)  # autocommit; we BEGIN explicitly
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=10000;")
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(SCHEMA)


def ping() -> bool:
    """True when SQLite is reachable (`SELECT 1`)."""
    try:
        with connection() as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1
    except (sqlite3.Error, OSError, ValueError):
        return False


@dataclass(frozen=True)
class SessionState:
    exists: bool
    system_prompt: str | None
    history: list[dict[str, str]]


def load_session(session_id: str) -> SessionState:
    """Stored system prompt + ordered user/assistant history (empty for unknown ids)."""
    with connection() as conn:
        row = conn.execute(
            "SELECT system_prompt FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return SessionState(False, None, [])
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? AND role IN ('user', 'assistant') ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return SessionState(
            True, row["system_prompt"], [{"role": r["role"], "content": r["content"]} for r in rows]
        )


def persist_exchange(
    session_id: str,
    user_content: str,
    assistant_content: str,
    assistant_metadata: dict[str, Any],
    new_system_prompt: str | None = None,
) -> None:
    """Atomically store session metadata + user message + assistant message.

    `new_system_prompt`: None leaves the stored prompt alone, "" clears it,
    any other string sets/overrides it.
    """
    now = utc_now_iso()
    with connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                conn.execute(
                    "INSERT INTO sessions (session_id, system_prompt, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, new_system_prompt or None, now, now),
                )
            elif new_system_prompt is not None:
                conn.execute(
                    "UPDATE sessions SET system_prompt = ?, updated_at = ? WHERE session_id = ?",
                    (new_system_prompt or None, now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id)
                )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, NULL, ?)",
                (session_id, user_content, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (session_id, assistant_content, json.dumps(assistant_metadata), now),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def get_messages(session_id: str) -> list[dict[str, Any]] | None:
    """Oldest-first user/assistant messages, or None when the session is unknown."""
    with connection() as conn:
        if (
            conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            is None
        ):
            return None
        rows = conn.execute(
            "SELECT role, content, metadata FROM messages "
            "WHERE session_id = ? AND role IN ('user', 'assistant') ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        item: dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if r["role"] == "assistant" and r["metadata"]:
            item["metadata"] = json.loads(r["metadata"])
        out.append(item)
    return out
