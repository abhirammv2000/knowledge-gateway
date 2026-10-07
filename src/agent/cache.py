"""A cache of past answers, matched by what the question means and not by its exact words.

"What notice is needed to terminate?" and "How much notice do I have to give to end it?"
should get the same stored answer, which skips the model call and costs nothing.

A hit needs all of these:
- the same contract
- the same model setup (a different model or prompt version gives different answers)
- an embedding similarity at or above the threshold
- an entry younger than the time to live

It can still be wrong, and the cache_eval script measures how. Questions that differ by one
word that flips the meaning ("allow" and "prohibit") are close in embedding space, so the
threshold has to be high, and some pairs will still be mixed up. Nothing is cached for a
question that came with earlier turns, because its answer depends on them.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, contract TEXT NOT NULL, model_key TEXT NOT NULL,
    question TEXT NOT NULL, vec BLOB NOT NULL, answer TEXT NOT NULL, created REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS answers_by_scope ON answers(contract, model_key);
"""


@dataclass(frozen=True)
class CacheHit:
    answer: dict[str, Any]
    similarity: float
    matched_question: str


class SemanticCache:
    def __init__(
        self,
        embed: Callable[[str], np.ndarray],
        path: str | Path = ":memory:",
        threshold: float = 0.92,
        ttl_seconds: float = 7 * 86400,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._embed = embed
        self.threshold = threshold
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(_SCHEMA)

    def _vector(self, text: str) -> np.ndarray:
        vec = np.asarray(self._embed(text), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def get(self, question: str, contract: str, model_key: str) -> CacheHit | None:
        vec = self._vector(question)
        oldest_allowed = self._clock() - self._ttl
        with self._lock:
            rows = self._db.execute(
                "SELECT id, question, vec, answer FROM answers WHERE contract = ? AND model_key = ? AND created >= ?",
                (contract, model_key, oldest_allowed),
            ).fetchall()
        best: tuple[float, tuple] | None = None
        for row in rows:
            similarity = float(np.dot(vec, np.frombuffer(row[2], dtype=np.float32)))
            if best is None or similarity > best[0]:
                best = (similarity, row)
        if best is None or best[0] < self.threshold:
            return None
        with self._lock, self._db:
            self._db.execute("UPDATE answers SET hits = hits + 1 WHERE id = ?", (best[1][0],))
        return CacheHit(json.loads(best[1][3]), round(best[0], 4), best[1][1])

    def put(self, question: str, contract: str, model_key: str, answer: dict[str, Any]) -> None:
        vec = self._vector(question)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO answers (contract, model_key, question, vec, answer, created) VALUES (?,?,?,?,?,?)",
                (contract, model_key, question, vec.tobytes(), json.dumps(answer), self._clock()),
            )
            self._db.execute(
                "DELETE FROM answers WHERE id NOT IN (SELECT id FROM answers ORDER BY id DESC LIMIT ?)", (self._max,)
            )

    def stats(self) -> dict[str, int]:
        with self._lock:
            entries, hits = self._db.execute("SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM answers").fetchone()
        return {"entries": entries, "hits": hits}
