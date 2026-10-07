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
