"""API keys, who spent what, and the limits that stop a key from spending too much.

Keys are random and shown once. Only a SHA-256 hash is stored, so a copy of the database does
not give anyone a working key. Three limits apply to every request:
- a per-minute request limit for the key
- a daily dollar budget for the key
- a daily dollar budget for the whole service, which is what protects a public demo from a
  bill nobody planned for
Spend is checked before a request starts and recorded after it ends, so a few requests that
start together can pass the budget by the cost of those requests. The limits are a backstop,
not an exact meter.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

KEY_PREFIX = "kg_"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE, created REAL NOT NULL,
    daily_budget_usd REAL NOT NULL, rpm INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 1,
    is_admin INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT, api_key_id TEXT NOT NULL, ts REAL NOT NULL, status TEXT NOT NULL,
    model TEXT, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0, seconds REAL NOT NULL DEFAULT 0, cached INTEGER NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0, fallback INTEGER NOT NULL DEFAULT 0, arm TEXT NOT NULL DEFAULT 'control', found INTEGER);
CREATE INDEX IF NOT EXISTS usage_by_key_time ON usage(api_key_id, ts);
"""


@dataclass(frozen=True)
class ApiKey:
    id: str
    name: str
    daily_budget_usd: float
    rpm: int
    is_admin: bool


@dataclass(frozen=True)
class Spend:
    spent_usd: float
    budget_usd: float
    requests: int


class Denied(Exception):
    """A request the limits do not allow. `status` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _start_of_utc_day(now: float) -> float:
    return now - (now % 86400)


class Accounts:
    def __init__(self, path: str | Path = ":memory:", global_daily_budget_usd: float = 0.0, clock=time.time) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        if "arm" not in {row[1] for row in self._db.execute("PRAGMA table_info(usage)")}:
            self._db.execute("ALTER TABLE usage ADD COLUMN arm TEXT NOT NULL DEFAULT 'control'")  # a database from before A/B
        if "found" not in {row[1] for row in self._db.execute("PRAGMA table_info(usage)")}:
            self._db.execute("ALTER TABLE usage ADD COLUMN found INTEGER")  # a database from before drift checks
        self._global_budget = global_daily_budget_usd
        self._clock = clock
        self._recent: dict[str, deque[float]] = defaultdict(deque)

    def create_key(self, name: str, daily_budget_usd: float = 1.0, rpm: int = 10, is_admin: bool = False) -> tuple[str, ApiKey]:
        """Make a key. The plain text is returned once and is not stored."""
        plain = KEY_PREFIX + secrets.token_urlsafe(24)
        key_id = "key_" + uuid.uuid4().hex[:10]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO api_keys (id, name, key_hash, created, daily_budget_usd, rpm, is_admin) VALUES (?,?,?,?,?,?,?)",
                (key_id, name, _hash(plain), self._clock(), daily_budget_usd, rpm, int(is_admin)),
            )
        return plain, ApiKey(key_id, name, daily_budget_usd, rpm, is_admin)

    def authenticate(self, plain: str | None) -> ApiKey | None:
        if not plain or not plain.startswith(KEY_PREFIX):
            return None
        wanted = _hash(plain)
        with self._lock:
            row = self._db.execute(
                "SELECT id, name, key_hash, daily_budget_usd, rpm, is_admin FROM api_keys WHERE key_hash = ? AND active = 1",
                (wanted,),
            ).fetchone()
        if row is None or not hmac.compare_digest(row[2], wanted):
            return None
        return ApiKey(row[0], row[1], row[3], row[4], bool(row[5]))

    def revoke(self, key_id: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("UPDATE api_keys SET active = 0 WHERE id = ?", (key_id,)).rowcount > 0

    # limits

    def spent_today(self, api_key_id: str | None = None) -> float:
        since = _start_of_utc_day(self._clock())
        query, args = "SELECT COALESCE(SUM(cost_usd), 0) FROM usage WHERE ts >= ?", [since]
        if api_key_id:
            query += " AND api_key_id = ?"
            args.append(api_key_id)
        with self._lock:
            return float(self._db.execute(query, args).fetchone()[0])

    def check(self, key: ApiKey) -> None:
        """Raise Denied if this key may not start a request now."""
        now = self._clock()
        with self._lock:
            window = self._recent[key.id]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= key.rpm:
                retry = int(60 - (now - window[0])) + 1
                raise Denied(429, f"rate limit of {key.rpm} requests a minute reached", retry)
        if self._global_budget and self.spent_today() >= self._global_budget:
            raise Denied(503, "the service has reached its spending limit for today")
        if self.spent_today(key.id) >= key.daily_budget_usd:
            raise Denied(402, "this key has reached its daily budget")
        # only a request that is allowed counts against the rate limit
        with self._lock:
            self._recent[key.id].append(now)

    def record(self, key: ApiKey, status: str, model: str | None, input_tokens: int, output_tokens: int,
               cost_usd: float, seconds: float, cached: bool, tool_calls: int, fallback: bool,
               arm: str = "control", found: bool | None = None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO usage (api_key_id, ts, status, model, input_tokens, output_tokens, cost_usd, seconds,"
                " cached, tool_calls, fallback, arm, found) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key.id, self._clock(), status, model, input_tokens, output_tokens, cost_usd, seconds,
                 int(cached), tool_calls, int(fallback), arm, None if found is None else int(found)),
            )

    def rows_between(self, start: float, end: float) -> list[dict]:
        """Requests that reached a model between two times, for drift checks."""
        with self._lock:
            rows = self._db.execute(
                "SELECT status, cost_usd, seconds, tool_calls, fallback, found FROM usage"
                " WHERE cached = 0 AND status != 'summarized' AND ts >= ? AND ts < ?", (start, end)
            ).fetchall()
        return [{"status": r[0], "cost_usd": r[1], "seconds": r[2], "tool_calls": r[3], "fallback": r[4],
                 "found": None if r[5] is None else bool(r[5])} for r in rows]

    def by_arm(self, since: float = 0.0) -> dict[str, dict]:
        """What each arm of the A/B split did: counts, outcomes, cost and latency. Cache hits are left out,
        because they never reached a model."""
        with self._lock:
            rows = self._db.execute(
                "SELECT arm, status, cost_usd, seconds, fallback FROM usage WHERE cached = 0 AND status != 'summarized' AND ts >= ?", (since,)
            ).fetchall()
        arms: dict[str, list[tuple]] = defaultdict(list)
        for row in rows:
            arms[row[0]].append(row[1:])
        report = {}
        for arm, items in arms.items():
            n = len(items)
            seconds = sorted(s for _, _, s, _ in items)
            report[arm] = {
                "requests": n,
                "answered": sum(st == "answered" for st, *_ in items) / n,
                "failed": sum(st in ("error", "timeout") for st, *_ in items) / n,
                "cost_per_request_usd": round(sum(c for _, c, _, _ in items) / n, 6),
                "median_seconds": seconds[n // 2],
                "fallback_rate": sum(bool(f) for *_, f in items) / n,
            }
        return report

    def spend(self, key: ApiKey) -> Spend:
        since = _start_of_utc_day(self._clock())
        with self._lock:
            count = self._db.execute(
                "SELECT COUNT(*) FROM usage WHERE api_key_id = ? AND status != 'summarized' AND ts >= ?", (key.id, since)
            ).fetchone()[0]
        return Spend(round(self.spent_today(key.id), 6), key.daily_budget_usd, count)
