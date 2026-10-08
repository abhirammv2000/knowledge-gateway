"""The HTTP API, over FastAPI's test client with the scripted model."""
import io

import numpy as np
import pytest
from conftest import FakeEmbedder, FakeReranker, FakeStore
from fakes import FakeLLM, call, reply, submit
from fastapi.testclient import TestClient

from agent import admin
from agent.accounts import Accounts
from agent.api import create_app
from agent.cache import SemanticCache
from agent.config import Settings
from agent.memory import Memory
from agent.service import AgentService
from gateway.service import GatewayService

CONTRACT = "Acme Services Agreement"
TEXT = "1. Termination\n\nEither party may terminate this agreement on ninety days written notice."
GOOD = {"source_id": "S1", "quote": "terminate this agreement on ninety days written notice"}


def good_run():
    return [reply(call("search_contract", {"title": CONTRACT, "query": "termination"}, "c1")),
            reply(submit(answer="Ninety days.", citations=[GOOD]))]


def make_client(*script, rpm=100, budget=5.0):
    gateway = GatewayService(FakeStore({CONTRACT: TEXT}), FakeEmbedder(), FakeReranker())
    accounts = Accounts()
    user, user_key = accounts.create_key("user", daily_budget_usd=budget, rpm=rpm)
    admin_plain, _ = accounts.create_key("ops", is_admin=True, rpm=100)
    other, _ = accounts.create_key("other", daily_budget_usd=5, rpm=100)
    cache = SemanticCache(lambda t: np.ones(4), threshold=0.99)
    service = AgentService(Settings(), FakeLLM(*script), gateway, Memory(), accounts, cache)
    client = TestClient(create_app(service), raise_server_exceptions=False)
    return client, {"user": user, "admin": admin_plain, "other": other}, service


def auth(key):
    return {"Authorization": f"Bearer {key}"}


def ask(client, key, **body):
    body.setdefault("question", "How can this be terminated?")
    body.setdefault("contract", CONTRACT)
    return client.post("/v1/ask", json=body, headers=auth(key))


# public routes

def test_health_and_the_demo_page_need_no_key():
    client, _, _ = make_client()

    assert client.get("/healthz").json() == {"ok": True}
    page = client.get("/")
    assert page.status_code == 200 and "Contract review agent" in page.text
    assert "kg_" not in page.text.replace("kg_...", "")  # no key is built into the page


def test_every_response_carries_a_request_id_and_nosniff():
    client, _, _ = make_client()
    response = client.get("/healthz")

    assert len(response.headers["x-request-id"]) == 12
    assert response.headers["x-content-type-options"] == "nosniff"


# auth

@pytest.mark.parametrize("header", [None, "", "Bearer", "Bearer kg_nope", "Basic abc", "kg_nope"])
def test_asking_without_a_valid_key_is_a_401(header):
    client, _, service = make_client()
    headers = {"Authorization": header} if header is not None else {}

    response = client.post("/v1/ask", json={"question": "x"}, headers=headers)

    assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"
    assert service.llm.calls == []


def test_the_other_routes_need_a_key_too():
    client, _, _ = make_client()

    for method, path in [("get", "/v1/contracts"), ("get", "/v1/usage"), ("delete", "/v1/data"),
                         ("delete", "/v1/sessions/x"), ("get", "/metrics")]:
        assert getattr(client, method)(path).status_code == 401, path


# asking

def test_a_good_question_returns_a_checked_answer():
    client, keys, _ = make_client(*good_run())

    response = ask(client, keys["user"])

    body = response.json()
    assert response.status_code == 200
    assert body["found"] and body["verified"] and body["stop_reason"] == "answered"
    assert body["citations"][0]["quote"] == GOOD["quote"]
    assert body["session_id"].startswith("ses_") and body["usage"]["tool_calls"] == 1


def test_an_unknown_contract_is_a_404_with_a_message():
    client, keys, service = make_client()

    response = ask(client, keys["user"], contract="No Such Contract")

    assert response.status_code == 404 and "exact title" in response.json()["error"]
    assert service.llm.calls == []


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "x" * 5000}, {"question": 5}])
def test_a_malformed_request_is_a_422(body):
    client, keys, _ = make_client()

    assert client.post("/v1/ask", json=body, headers=auth(keys["user"])).status_code == 422


def test_the_rate_limit_answers_429_with_retry_after():
    client, keys, _ = make_client(*good_run(), rpm=1)
    assert ask(client, keys["user"]).status_code == 200

    response = ask(client, keys["user"])

    assert response.status_code == 429 and int(response.headers["retry-after"]) > 0


def test_a_spent_budget_answers_402():
    client, keys, service = make_client(*good_run(), budget=0.0001)
    service.accounts.record(service.accounts.authenticate(keys["user"]), "answered", "m", 1, 1, 1.0, 1.0, False, 0, False)

    assert ask(client, keys["user"]).status_code == 402


def test_a_crash_inside_returns_a_plain_500_without_details():
    client, keys, _ = make_client()

    class Boom:
        async def ask(self, *a, **k):
            raise RuntimeError("password=hunter2 in /srv/app/secret.py")

    client.app.state.service.ask = Boom().ask
    response = ask(client, keys["user"])

    assert response.status_code == 500 and response.json() == {"error": "internal error"}


# the rest of the API

def test_contracts_can_be_searched():
    client, keys, _ = make_client()

    body = client.get("/v1/contracts?contains=acme", headers=auth(keys["user"])).json()

    assert body["titles"] == [CONTRACT]


def test_usage_shows_todays_spend():
    client, keys, _ = make_client(*good_run())
    ask(client, keys["user"])

    usage = client.get("/v1/usage", headers=auth(keys["user"])).json()

    assert usage["requests_today"] == 1 and usage["spent_today_usd"] == pytest.approx(0.002)
    assert usage["key"] == "user" and usage["daily_budget_usd"] == 5.0


def test_metrics_need_an_admin_key_and_show_the_agent_counters():
    client, keys, _ = make_client(*good_run())
    ask(client, keys["user"])

    assert client.get("/metrics", headers=auth(keys["user"])).status_code == 403
    text = client.get("/metrics", headers=auth(keys["admin"])).text
    assert "agent_requests_total" in text and "agent_cost_usd_total" in text


def test_a_session_can_be_deleted_by_its_owner_only():
    client, keys, _ = make_client(*good_run())
    session = ask(client, keys["user"]).json()["session_id"]

    assert client.delete(f"/v1/sessions/{session}", headers=auth(keys["other"])).status_code == 404
    assert client.delete(f"/v1/sessions/{session}", headers=auth(keys["user"])).status_code == 200
    assert client.delete(f"/v1/sessions/{session}", headers=auth(keys["user"])).status_code == 404


def test_someone_elses_session_cannot_be_used_to_ask():
    client, keys, _ = make_client(*good_run())
    session = ask(client, keys["user"]).json()["session_id"]

    response = ask(client, keys["other"], session_id=session)

    assert response.status_code == 404


def test_delete_data_wipes_the_keys_memory():
    client, keys, service = make_client(*good_run())
    session = ask(client, keys["user"]).json()["session_id"]

    assert client.delete("/v1/data", headers=auth(keys["user"])).status_code == 200
    assert service.memory.session_matter(session, service.accounts.authenticate(keys["user"]).id) is None


# the admin command

def test_the_admin_command_creates_lists_and_revokes_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    out = io.StringIO()

    assert admin.main(["create-key", "--name", "demo", "--budget", "2", "--rpm", "5"], out) == 0
    text = out.getvalue()
    plain = next(line.split()[1] for line in text.splitlines() if line.startswith("key:"))
    key_id = next(line.split()[1] for line in text.splitlines() if line.startswith("id:"))
    assert plain.startswith("kg_") and Accounts(tmp_path / "accounts.db").authenticate(plain)

    listing = io.StringIO()
    admin.main(["list"], listing)
    assert "demo" in listing.getvalue() and "plain" not in listing.getvalue() and plain not in listing.getvalue()

    assert admin.main(["revoke", key_id], io.StringIO()) == 0
    assert Accounts(tmp_path / "accounts.db").authenticate(plain) is None
    assert admin.main(["revoke", "key_missing"], io.StringIO()) == 1


def test_a_model_outage_is_a_503_not_a_normal_answer():
    client, keys, _ = make_client(RuntimeError("every provider is down"))

    response = ask(client, keys["user"])

    assert response.status_code == 503 and response.headers["retry-after"] == "10"
    body = response.json()
    assert body["stop_reason"] == "error" and "not available" in body["error"]
    assert "every provider" not in response.text and "found" not in body


def test_a_refusal_to_answer_is_still_a_200():
    bad = lambda i: reply(submit(answer="x", citations=[{"source_id": "S1", "quote": "not in the text"}], call_id=f"s{i}"))
    client, keys, _ = make_client(reply(call("search_contract", {"title": CONTRACT, "query": "x"}, "c1")), bad(1), bad(2), bad(3))

    response = ask(client, keys["user"])

    assert response.status_code == 200 and response.json()["stop_reason"] == "refused_unverified"


def test_feedback_is_recorded_for_the_owner_only():
    client, keys, service = make_client(*good_run())
    session = ask(client, keys["user"]).json()["session_id"]

    assert client.post("/v1/feedback", json={"session_id": session, "helpful": True},
                       headers=auth(keys["other"])).status_code == 404
    ok = client.post("/v1/feedback", json={"session_id": session, "helpful": False}, headers=auth(keys["user"]))

    assert ok.status_code == 200 and service.memory.feedback_by_arm() == {"control": {"helpful": 0, "not_helpful": 1}}


def test_feedback_needs_a_key_and_a_boolean():
    client, keys, _ = make_client()

    assert client.post("/v1/feedback", json={"session_id": "x", "helpful": True}).status_code == 401
    assert client.post("/v1/feedback", json={"session_id": "x"}, headers=auth(keys["user"])).status_code == 422


def test_the_ab_report_command_prints_each_arm(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    accounts = Accounts(tmp_path / "accounts.db")
    _, key = accounts.create_key("k")
    accounts.record(key, "answered", "m", 1, 1, 0.02, 3.0, False, 1, False, "challenger")
    memory = Memory(tmp_path / "memory.db")
    memory.add_feedback(memory.create_session(key.id, "general", "challenger"), key.id, True)
    out = io.StringIO()

    assert admin.main(["ab-report"], out) == 0

    line = out.getvalue()
    assert "challenger" in line and "1 requests" in line and "thumbs up 1 down 0" in line


def test_notes_can_be_listed_approved_and_deleted_by_their_owner():
    client, keys, service = make_client()
    mine = service.memory.add_note(service.accounts.authenticate(keys["user"]).id, CONTRACT, "a finding", approved=False)

    listed = client.get("/v1/notes", headers=auth(keys["user"])).json()["notes"]
    assert listed == [{"id": mine, "matter": CONTRACT, "text": "a finding", "approved": False}]
    assert client.get("/v1/notes", headers=auth(keys["other"])).json() == {"notes": []}

    assert client.post(f"/v1/notes/{mine}/approve", headers=auth(keys["other"])).status_code == 404
    assert client.post(f"/v1/notes/{mine}/approve", headers=auth(keys["user"])).status_code == 200
    assert client.get("/v1/notes", headers=auth(keys["user"])).json()["notes"][0]["approved"] is True
    assert client.delete(f"/v1/notes/{mine}", headers=auth(keys["other"])).status_code == 404
    assert client.delete(f"/v1/notes/{mine}", headers=auth(keys["user"])).status_code == 200
    assert client.delete(f"/v1/notes/{mine}", headers=auth(keys["user"])).status_code == 404


def test_the_notes_routes_need_a_key():
    client, _, _ = make_client()

    assert client.get("/v1/notes").status_code == 401
    assert client.post("/v1/notes/1/approve").status_code == 401
    assert client.delete("/v1/notes/1").status_code == 401


def test_an_answer_says_how_many_notes_the_model_proposed():
    note = call("save_note", {"text": "worth remembering"}, "n1")
    client, keys, _ = make_client(reply(note), reply(submit(found=False, answer="Noted.")))

    assert ask(client, keys["user"]).json()["notes_proposed"] == 1


# idempotency keys

def keyed(client, key, idem, **body):
    body.setdefault("question", "How can this be terminated?")
    body.setdefault("contract", CONTRACT)
    return client.post("/v1/ask", json=body, headers={**auth(key), "Idempotency-Key": idem})


def test_a_retry_with_the_same_idempotency_key_gets_the_first_answer_without_a_second_run():
    client, keys, service = make_client(*good_run())

    first = keyed(client, keys["user"], "order-1")
    second = keyed(client, keys["user"], "order-1")

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json() and second.headers["idempotent-replay"] == "true"
    assert "idempotent-replay" not in first.headers
    assert len(service.llm.calls) == 2  # one run of two model steps, not two runs
    assert service.accounts.spend(service.accounts.authenticate(keys["user"])).requests == 1


def test_a_replay_does_not_use_up_the_rate_limit():
    client, keys, _ = make_client(*good_run(), rpm=1)

    assert keyed(client, keys["user"], "k1").status_code == 200
    assert keyed(client, keys["user"], "k1").status_code == 200
    assert keyed(client, keys["user"], "k2").status_code == 429


def test_the_same_key_with_a_different_request_is_refused():
    client, keys, service = make_client(*good_run())
    keyed(client, keys["user"], "k1")

    response = keyed(client, keys["user"], "k1", question="Something else entirely?")

    assert response.status_code == 422 and "different request" in response.json()["detail"]
    assert len(service.llm.calls) == 2


def test_the_same_key_while_the_first_request_is_running_is_a_409():
    client, keys, service = make_client(*good_run())
    key_id = service.accounts.authenticate(keys["user"]).id
    from agent.idempotency import fingerprint
    service.idempotency.begin(key_id, "k1", fingerprint(question="How can this be terminated?", contract=CONTRACT, session_id=None))

    assert keyed(client, keys["user"], "k1").status_code == 409
    assert service.llm.calls == []


def test_a_failed_request_is_not_stored_so_the_retry_runs():
    client, keys, service = make_client(RuntimeError("provider down"), *good_run())

    first = keyed(client, keys["user"], "k1")
    second = keyed(client, keys["user"], "k1")

    assert first.status_code == 503
    assert second.status_code == 200 and "idempotent-replay" not in second.headers


def test_two_users_can_use_the_same_idempotency_key():
    client, keys, service = make_client(*good_run(), *good_run())

    assert keyed(client, keys["user"], "same").status_code == 200
    other = keyed(client, keys["other"], "same")

    assert other.status_code == 200 and "idempotent-replay" not in other.headers


@pytest.mark.parametrize("value", ["", "x" * 65])
def test_a_bad_idempotency_key_is_a_422(value):
    client, keys, service = make_client()

    assert keyed(client, keys["user"], value).status_code == 422
    assert service.llm.calls == []


def test_deleting_my_data_also_forgets_stored_answers():
    client, keys, service = make_client(*good_run(), *good_run())
    keyed(client, keys["user"], "k1")

    client.delete("/v1/data", headers=auth(keys["user"]))
    again = keyed(client, keys["user"], "k1")

    assert "idempotent-replay" not in again.headers


def test_stored_answers_expire_after_a_day():
    from agent.idempotency import IdempotencyStore, TTL_SECONDS
    now = [1000.0]
    store = IdempotencyStore(clock=lambda: now[0])
    store.begin("k", "i", "fp")
    store.finish("k", "i", "fp", 200, {"a": 1})
    assert store.begin("k", "i", "fp").kind == "replay"

    now[0] += TTL_SECONDS + 1

    assert store.begin("k", "i", "fp").kind == "new"
