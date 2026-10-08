"""Idempotency keys for POST /v1/ask.

A client that times out and retries a question should not run it twice and pay twice. If the request
carries an `Idempotency-Key` header, the first answer is stored for a day and a retry with the same key
gets that answer back without touching the model, the rate limit or the budget.

Rules, the same ones payment APIs use:
- the same key with a different request body is a client bug and is refused
- the same key while the first request is still running is refused, not run a second time
- only a finished answer is stored, so a retry after a failure runs normally
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

TTL_SECONDS = 24 * 3600
MAX_KEY_CHARS = 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency (
    api_key_id TEXT NOT NULL, idem_key TEXT NOT NULL, fingerprint TEXT NOT NULL, ts REAL NOT NULL,
    status INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY (api_key_id, idem_key));
"""


@dataclass(frozen=True)
class Begin:
    # new: run the request. replay: send the stored answer. mismatch: same key, different request.
    # in_flight: the first request with this key has not finished.
    kind: str
    status: int = 0
    body: dict | None = None


def fingerprint(**fields) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


class IdempotencyStore:
    def __init__(self, path: str | Path = ":memory:", clock=time.time) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._clock = clock
        self._running: set[tuple[str, str]] = set()

    def begin(self, api_key_id: str, idem_key: str, request_fingerprint: str) -> Begin:
        token = (api_key_id, idem_key)
        with self._lock, self._db:
            self._db.execute("DELETE FROM idempotency WHERE ts < ?", (self._clock() - TTL_SECONDS,))
            row = self._db.execute(
                "SELECT fingerprint, status, body FROM idempotency WHERE api_key_id = ? AND idem_key = ?", token
            ).fetchone()
            if row is not None:
                if row[0] != request_fingerprint:
                    return Begin("mismatch")
                return Begin("replay", row[1], json.loads(row[2]))
            if token in self._running:
                return Begin("in_flight")
            self._running.add(token)
        return Begin("new")

    def finish(self, api_key_id: str, idem_key: str, request_fingerprint: str, status: int, body: dict) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO idempotency (api_key_id, idem_key, fingerprint, ts, status, body)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (api_key_id, idem_key, request_fingerprint, self._clock(), status, json.dumps(body)),
            )
            self._running.discard((api_key_id, idem_key))

    def abandon(self, api_key_id: str, idem_key: str) -> None:
        """The request did not produce a storable answer. A retry with the same key may run again."""
        with self._lock:
            self._running.discard((api_key_id, idem_key))

    def delete_for_key(self, api_key_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM idempotency WHERE api_key_id = ?", (api_key_id,))
