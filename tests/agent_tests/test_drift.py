"""Drift checks over the usage ledger. Pure statistics, no model."""
import io
import sqlite3

import pytest

from agent import admin
from agent.accounts import Accounts
from agent.drift import MIN_REQUESTS, compare


def row(status="answered", cost=0.01, seconds=4.0, tools=1, fallback=False, found=True):
    return {"status": status, "cost_usd": cost, "seconds": seconds, "tool_calls": tools, "fallback": fallback,
            "found": found}


def flagged(findings):
    return {f.metric for f in findings if f.flagged}


def mixed(n, found_share, **kw):
    """n answered requests where the given share found something."""
    hits = round(n * found_share)
    return [row(found=i < hits, **kw) for i in range(n)]


def test_identical_windows_flag_nothing():
    window = mixed(100, 0.5)

    assert flagged(compare(window, list(window))) == set()


def test_a_drop_in_how_often_it_finds_something_is_flagged():
    findings = compare(mixed(100, 0.5), mixed(100, 0.2))

    assert "found something" in flagged(findings)


def test_a_small_wobble_is_not_flagged():
    assert flagged(compare(mixed(100, 0.50), mixed(100, 0.45))) == set()


def test_more_provider_failures_are_flagged():
    recent = [row("error" if i < 20 else "answered") for i in range(100)]

    assert "provider failures" in flagged(compare([row() for _ in range(100)], recent))


def test_a_slower_agent_is_flagged_by_the_ratio_of_medians():
    findings = compare([row(seconds=4.0) for _ in range(50)], [row(seconds=9.0) for _ in range(50)])

    assert flagged(findings) == {"median seconds"}


def test_a_cheaper_agent_is_flagged_too():
    findings = compare([row(cost=0.012) for _ in range(50)], [row(cost=0.002) for _ in range(50)])

    assert "median cost_usd" in flagged(findings)


def test_too_few_requests_never_flag_anything():
    few = MIN_REQUESTS - 1

    assert flagged(compare(mixed(few, 0.9), mixed(few, 0.1))) == set()
    assert flagged(compare(mixed(100, 0.9), mixed(few, 0.1))) == set()


def test_empty_windows_give_no_findings():
    assert compare([], []) == []


def test_failures_are_not_counted_as_not_found():
    # a provider outage must not look like the agent finding less
    baseline = mixed(100, 0.5)
    recent = mixed(60, 0.5) + [row("error", found=None) for _ in range(40)]

    assert "found something" not in flagged(compare(baseline, recent))


# the ledger and the command

def test_the_ledger_returns_requests_in_a_time_range_and_skips_cache_hits():
    now = [1000.0]
    accounts = Accounts(clock=lambda: now[0])
    _, key = accounts.create_key("k")
    accounts.record(key, "answered", "m", 1, 1, 0.01, 2.0, False, 1, False, "control", found=True)
    now[0] = 2000.0
    accounts.record(key, "answered", "m", 1, 1, 0.01, 2.0, False, 1, False, "control", found=False)
    accounts.record(key, "cached", None, 0, 0, 0.0, 0.0, True, 0, False, "control")

    early = accounts.rows_between(0, 1500)
    late = accounts.rows_between(1500, 3000)

    assert [r["found"] for r in early] == [True] and [r["found"] for r in late] == [False]


def test_an_older_usage_table_without_found_is_upgraded(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT, api_key_id TEXT NOT NULL, ts REAL NOT NULL,"
                     " status TEXT NOT NULL, model TEXT, input_tokens INTEGER NOT NULL DEFAULT 0,"
                     " output_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,"
                     " seconds REAL NOT NULL DEFAULT 0, cached INTEGER NOT NULL DEFAULT 0,"
                     " tool_calls INTEGER NOT NULL DEFAULT 0, fallback INTEGER NOT NULL DEFAULT 0,"
                     " arm TEXT NOT NULL DEFAULT 'control');"
                     " INSERT INTO usage (api_key_id, ts, status) VALUES ('k', 1.0, 'answered');")
    db.commit()
    db.close()

    assert Accounts(path).rows_between(0, 10)[0]["found"] is None


def test_the_drift_command_prints_changes(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    accounts = Accounts(tmp_path / "accounts.db")
    _, key = accounts.create_key("k")
    day = 86400
    now = time.time()
    for i in range(60):
        for offset, hit in ((now - 10 * day, i % 2 == 0), (now - 2 * day, i % 6 == 0)):
            accounts._clock = lambda t=offset: t
            accounts.record(key, "answered", "m", 1, 1, 0.01, 2.0, False, 1, False, "control", found=hit)
    out = io.StringIO()

    assert admin.main(["drift-report", "--days", "7"], out) == 0

    text = out.getvalue()
    assert "60 requests" in text and "CHANGED found something" in text


def test_the_drift_command_says_when_there_is_no_history(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    out = io.StringIO()

    admin.main(["drift-report"], out)

    assert "not enough history" in out.getvalue()
