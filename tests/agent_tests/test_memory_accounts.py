"""Memory and accounts, against in-memory SQLite."""
import sqlite3

import pytest

from agent.accounts import Accounts, Denied
from agent.memory import Memory


@pytest.fixture
def memory():
    return Memory()


# memory

def test_a_session_remembers_its_turns_oldest_first(memory):
    session = memory.create_session("key_a", "matter-1")
    memory.add_turn(session, "What is the term?", "Two years.", True, "Acme")
    memory.add_turn(session, "And renewal?", "Automatic.", True, "Acme")

    assert memory.history(session) == [("What is the term?", "Two years."), ("And renewal?", "Automatic.")]


def test_history_keeps_only_the_most_recent_turns(memory):
    session = memory.create_session("key_a", "m")
    for i in range(8):
        memory.add_turn(session, f"q{i}", f"a{i}", True, None)

    assert [q for q, _ in memory.history(session, turns=3)] == ["q5", "q6", "q7"]


def test_history_drops_the_oldest_turns_first_when_it_is_too_long(memory):
    session = memory.create_session("key_a", "m")
    for i in range(4):
        memory.add_turn(session, f"question {i}", "x" * 100, True, None)

    kept = memory.history(session, turns=10, max_chars=250)

    assert [q for q, _ in kept] == ["question 2", "question 3"]


def test_the_newest_turn_is_kept_even_if_it_alone_is_over_the_limit(memory):
    session = memory.create_session("key_a", "m")
    memory.add_turn(session, "q", "y" * 5000, True, None)

    assert len(memory.history(session, max_chars=100)) == 1


def test_a_session_belongs_to_one_key(memory):
    session = memory.create_session("key_a", "matter-1")

    assert memory.session_matter(session, "key_a") == "matter-1"
    assert memory.session_matter(session, "key_b") is None
    assert memory.delete_session(session, "key_b") is False
    assert memory.session_matter(session, "key_a") == "matter-1"


def test_personal_data_is_removed_before_it_is_stored(memory):
    session = memory.create_session("key_a", "m")
    memory.add_turn(session, "Email maria.garcia@lawfirm.com about it", "Contact john.smith@acme.com", True, None)

    question, answer = memory.history(session)[0]

    assert "maria.garcia@lawfirm.com" not in question and "john.smith@acme.com" not in answer
    raw = memory._db.execute("SELECT question, answer FROM turns").fetchone()
    assert "lawfirm.com" not in raw[0] and "acme.com" not in raw[1]


def test_notes_outlive_a_session_and_are_per_matter(memory):
    memory.add_note("key_a", "matter-1", "The client cares about termination rights.")
    memory.add_note("key_a", "matter-2", "Different matter.")

    assert memory.notes("key_a", "matter-1") == ["The client cares about termination rights."]
    assert memory.notes("key_b", "matter-1") == []


def test_notes_are_returned_oldest_first_and_capped(memory):
    for i in range(30):
        memory.add_note("key_a", "m", f"note {i}")

    shown = memory.notes("key_a", "m", limit=5)

    assert shown == [f"note {i}" for i in range(25, 30)]


def test_old_notes_are_dropped_past_the_storage_limit(memory):
    for i in range(110):
        memory.add_note("key_a", "m", f"note {i}")

    count = memory._db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    assert count == 100
    assert memory.notes("key_a", "m", limit=1) == ["note 109"]


def test_a_note_with_personal_data_is_stored_redacted(memory):
    memory.add_note("key_a", "m", "Client contact is bob.jones@client.com")

    assert "bob.jones@client.com" not in memory.notes("key_a", "m")[0]


def test_deleting_a_session_removes_its_turns(memory):
    session = memory.create_session("key_a", "m")
    memory.add_turn(session, "q", "a", True, None)

    assert memory.delete_session(session, "key_a") is True

    assert memory._db.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0


def test_delete_all_removes_everything_for_one_key_only(memory):
    mine, theirs = memory.create_session("key_a", "m"), memory.create_session("key_b", "m")
    memory.add_turn(mine, "q", "a", True, None)
    memory.add_turn(theirs, "q", "a", True, None)
    memory.add_note("key_a", "m", "mine")
    memory.add_note("key_b", "m", "theirs")

    memory.delete_all("key_a")

    assert memory.session_matter(mine, "key_a") is None and memory.notes("key_a", "m") == []
    assert memory.history(theirs) and memory.notes("key_b", "m") == ["theirs"]


# accounts

class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def accounts(clock):
    return Accounts(clock=clock)


def test_a_key_works_and_the_plain_text_is_never_stored(accounts):
    plain, key = accounts.create_key("demo")

    assert plain.startswith("kg_") and len(plain) > 30
    assert accounts.authenticate(plain) == key
    dump = "\n".join(str(row) for row in accounts._db.execute("SELECT * FROM api_keys"))
    assert plain not in dump and plain[3:] not in dump


@pytest.mark.parametrize("bad", [None, "", "kg_wrong", "nope", "Bearer x", "kg_"])
def test_wrong_keys_are_refused(accounts, bad):
    accounts.create_key("demo")
    assert accounts.authenticate(bad) is None


def test_a_revoked_key_stops_working(accounts):
    plain, key = accounts.create_key("demo")
    accounts.revoke(key.id)
    assert accounts.authenticate(plain) is None


def test_the_rate_limit_allows_rpm_requests_a_minute(accounts, clock):
    _, key = accounts.create_key("demo", rpm=3)
    for _ in range(3):
        accounts.check(key)

    with pytest.raises(Denied) as denied:
        accounts.check(key)
    assert denied.value.status == 429 and denied.value.retry_after > 0

    clock.now += 61
    accounts.check(key)


def test_the_rate_limit_is_per_key(accounts):
    _, a = accounts.create_key("a", rpm=1)
    _, b = accounts.create_key("b", rpm=1)
    accounts.check(a)
    accounts.check(b)


def test_a_key_stops_at_its_daily_budget_and_gets_a_fresh_one_the_next_day(accounts, clock):
    _, key = accounts.create_key("demo", daily_budget_usd=0.10, rpm=100)
    accounts.check(key)
    accounts.record(key, "answered", "m", 100, 10, 0.06, 1.0, False, 1, False)
    accounts.check(key)
    accounts.record(key, "answered", "m", 100, 10, 0.06, 1.0, False, 1, False)

    with pytest.raises(Denied) as denied:
        accounts.check(key)
    assert denied.value.status == 402

    clock.now += 86400
    accounts.check(key)


def test_one_keys_spending_does_not_count_against_another(accounts):
    _, a = accounts.create_key("a", daily_budget_usd=0.05, rpm=100)
    _, b = accounts.create_key("b", daily_budget_usd=0.05, rpm=100)
    accounts.record(a, "answered", "m", 1, 1, 0.5, 1.0, False, 1, False)

    accounts.check(b)


def test_the_global_budget_stops_every_key(clock):
    accounts = Accounts(global_daily_budget_usd=0.10, clock=clock)
    _, a = accounts.create_key("a", daily_budget_usd=100, rpm=100)
    _, b = accounts.create_key("b", daily_budget_usd=100, rpm=100)
    accounts.record(a, "answered", "m", 1, 1, 0.11, 1.0, False, 1, False)

    with pytest.raises(Denied) as denied:
        accounts.check(b)
    assert denied.value.status == 503


def test_a_denied_request_does_not_use_up_the_rate_limit(accounts):
    _, key = accounts.create_key("demo", daily_budget_usd=0.01, rpm=2)
    accounts.record(key, "answered", "m", 1, 1, 0.5, 1.0, False, 1, False)

    for _ in range(5):
        with pytest.raises(Denied) as denied:
            accounts.check(key)
        assert denied.value.status == 402


def test_spend_adds_up_todays_requests(accounts):
    _, key = accounts.create_key("demo", daily_budget_usd=2.0)
    accounts.record(key, "answered", "m", 1, 1, 0.25, 1.0, False, 1, False)
    accounts.record(key, "answered", "m", 1, 1, 0.50, 1.0, True, 0, False)

    spend = accounts.spend(key)

    assert (spend.spent_usd, spend.budget_usd, spend.requests) == (0.75, 2.0, 2)


def test_the_database_can_be_reopened_from_a_file(tmp_path, clock):
    path = tmp_path / "a.db"
    plain, _ = Accounts(path, clock=clock).create_key("demo")

    assert Accounts(path, clock=clock).authenticate(plain) is not None
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM api_keys").fetchone()[0] == 1
