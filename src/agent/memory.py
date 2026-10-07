"""What the agent remembers between questions, kept in SQLite.

Two kinds of memory:
- a session keeps the questions and answers of one conversation, so a follow-up like "and
  what about renewal?" makes sense. The last few turns go back to the model.
- notes belong to a matter and outlast any one session. The model saves them with the
  save_note tool and sees them again in every later question about that matter.

Everything is scoped to one API key, so one user can never read another's memory. Only
redacted text is stored: callers pass the question as the agent saw it, and every string goes
through the redactor once more on the way in.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

from agent.guard import Redactor

DEFAULT_HISTORY_TURNS = 5
DEFAULT_HISTORY_CHARS = 4000
MAX_NOTES_SHOWN = 20
MAX_NOTES_STORED = 100  # per matter, the oldest are dropped past this

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, api_key_id TEXT NOT NULL, matter TEXT NOT NULL, created REAL NOT NULL,
    arm TEXT NOT NULL DEFAULT 'control');
CREATE TABLE IF NOT EXISTS feedback (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE, ts REAL NOT NULL, helpful INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts REAL NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, found INTEGER NOT NULL, contract TEXT);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, api_key_id TEXT NOT NULL, matter TEXT NOT NULL,
    ts REAL NOT NULL, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns(session_id, id);
CREATE INDEX IF NOT EXISTS notes_by_matter ON notes(api_key_id, matter, id);
"""


class Memory:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        if "arm" not in {row[1] for row in self._db.execute("PRAGMA table_info(sessions)")}:
            self._db.execute("ALTER TABLE sessions ADD COLUMN arm TEXT NOT NULL DEFAULT 'control'")  # from before A/B

    def _clean(self, text: str) -> str:
        # a new vault each time, so tokens are not shared between unrelated strings
        return Redactor()(text)

    # sessions

    def create_session(self, api_key_id: str, matter: str, arm: str = "control") -> str:
        session_id = "ses_" + uuid.uuid4().hex[:16]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO sessions (id, api_key_id, matter, created, arm) VALUES (?, ?, ?, ?, ?)",
                (session_id, api_key_id, matter, time.time(), arm),
            )
        return session_id

    def session_arm(self, session_id: str) -> str:
        """Which model arm a session was assigned to. A conversation stays on one arm."""
        with self._lock:
            row = self._db.execute("SELECT arm FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row[0] if row else "control"

    def add_feedback(self, session_id: str, api_key_id: str, helpful: bool) -> str | None:
        """Record a thumbs up or down for a session, replacing an earlier one. Returns the session's arm,
        or None if the session does not exist or belongs to another key."""
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT arm FROM sessions WHERE id = ? AND api_key_id = ?", (session_id, api_key_id)
            ).fetchone()
            if row is None:
                return None
            self._db.execute(
                "INSERT OR REPLACE INTO feedback (session_id, ts, helpful) VALUES (?, ?, ?)",
                (session_id, time.time(), int(helpful)),
            )
        return row[0]

    def feedback_by_arm(self) -> dict[str, dict[str, int]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.arm, f.helpful, COUNT(*) FROM feedback f JOIN sessions s ON s.id = f.session_id"
                " GROUP BY s.arm, f.helpful"
            ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for arm, helpful, count in rows:
            out.setdefault(arm, {"helpful": 0, "not_helpful": 0})["helpful" if helpful else "not_helpful"] = count
        return out

    def session_matter(self, session_id: str, api_key_id: str) -> str | None:
        """The matter of a session, or None if it does not exist or belongs to another key."""
        with self._lock:
            row = self._db.execute(
                "SELECT matter FROM sessions WHERE id = ? AND api_key_id = ?", (session_id, api_key_id)
            ).fetchone()
        return row[0] if row else None

    def add_turn(self, session_id: str, question: str, answer: str, found: bool, contract: str | None) -> None:
        question, answer = self._clean(question), self._clean(answer)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO turns (session_id, ts, question, answer, found, contract) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, time.time(), question, answer, int(found), contract),
            )

    def history(self, session_id: str, turns: int = DEFAULT_HISTORY_TURNS,
                max_chars: int = DEFAULT_HISTORY_CHARS) -> list[tuple[str, str]]:
        """The last few (question, answer) pairs, oldest first, cut from the oldest end to fit max_chars."""
        with self._lock:
            rows = self._db.execute(
                "SELECT question, answer FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, turns),
            ).fetchall()
        kept: list[tuple[str, str]] = []
        used = 0
        for question, answer in rows:  # newest first, so the newest turns survive the cut
            size = len(question) + len(answer)
            if kept and used + size > max_chars:
                break
            kept.append((question, answer))
            used += size
        return list(reversed(kept))

    def delete_session(self, session_id: str, api_key_id: str) -> bool:
        with self._lock, self._db:
            cursor = self._db.execute("DELETE FROM sessions WHERE id = ? AND api_key_id = ?", (session_id, api_key_id))
        return cursor.rowcount > 0

    # notes

    def add_note(self, api_key_id: str, matter: str, text: str) -> None:
        text = self._clean(text).strip()
        if not text:
            return
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO notes (api_key_id, matter, ts, text) VALUES (?, ?, ?, ?)",
                (api_key_id, matter, time.time(), text),
            )
            self._db.execute(
                "DELETE FROM notes WHERE api_key_id = ? AND matter = ? AND id NOT IN "
                "(SELECT id FROM notes WHERE api_key_id = ? AND matter = ? ORDER BY id DESC LIMIT ?)",
                (api_key_id, matter, api_key_id, matter, MAX_NOTES_STORED),
            )

    def notes(self, api_key_id: str, matter: str, limit: int = MAX_NOTES_SHOWN) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT text FROM notes WHERE api_key_id = ? AND matter = ? ORDER BY id DESC LIMIT ?",
                (api_key_id, matter, limit),
            ).fetchall()
        return [r[0] for r in reversed(rows)]

    def delete_all(self, api_key_id: str) -> None:
        """Remove everything stored for one API key."""
        with self._lock, self._db:
            self._db.execute("DELETE FROM sessions WHERE api_key_id = ?", (api_key_id,))
            self._db.execute("DELETE FROM notes WHERE api_key_id = ?", (api_key_id,))
