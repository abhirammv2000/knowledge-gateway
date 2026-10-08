"""POST /v1/ask/stream: progress as server-sent events, then the checked answer. Over FastAPI's test client."""
import json

import pytest
from fakes import call, reply, submit
from test_api import CONTRACT, GOOD, auth, good_run, make_client


def stream(client, key, **body):
    body.setdefault("question", "How can this be terminated?")
    body.setdefault("contract", CONTRACT)
    return client.post("/v1/ask/stream", json=body, headers=auth(key))


def events(response):
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), block
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def test_a_good_question_streams_progress_then_the_checked_answer():
    client, keys, _ = make_client(*good_run())

    response = stream(client, keys["user"])
    got = events(response)

    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert [name for name, _ in got] == ["started", "step", "tool", "step", "verifying", "answer"]
    assert got[0][1]["session_id"].startswith("ses_")
    assert got[2][1] == {"type": "tool", "name": "search_contract", "ok": True}
    final = got[-1][1]
    assert final["found"] and final["verified"] and final["citations"][0]["quote"] == GOOD["quote"]
    assert final["session_id"] == got[0][1]["session_id"]


def test_progress_events_carry_only_names_and_counts_never_text():
    client, keys, _ = make_client(*good_run())

    got = events(stream(client, keys["user"]))

    for name, data in got[:-1]:
        assert set(data) <= {"type", "iteration", "name", "ok", "session_id"}
        assert "ninety days" not in json.dumps(data) and "terminated" not in json.dumps(data)


def test_the_streamed_answer_is_the_same_body_the_plain_endpoint_returns():
    client, keys, _ = make_client(*good_run(), *good_run())
    plain = client.post("/v1/ask", json={"question": "How can this be terminated?", "contract": CONTRACT},
                        headers=auth(keys["user"])).json()

    final = events(stream(client, keys["user"]))[-1][1]

    for field in ("found", "answer", "verified", "confidence", "stop_reason", "citations"):
        assert final[field] == plain[field]


def test_an_answer_whose_citation_fails_is_not_streamed_as_an_answer():
    bad = lambda i: reply(submit(answer="x", citations=[{"source_id": "S1", "quote": "not in the text"}], call_id=f"s{i}"))
    client, keys, _ = make_client(reply(call("search_contract", {"title": CONTRACT, "query": "x"}, "c1")), bad(1), bad(2), bad(3))

    got = events(stream(client, keys["user"]))

    final = got[-1][1]
    assert final["stop_reason"] == "refused_unverified" and final["found"] is False and "not in the text" not in json.dumps(got)




def test_a_missing_key_is_a_401_not_a_stream():
    client, _, service = make_client()

    response = client.post("/v1/ask/stream", json={"question": "x"})

    assert response.status_code == 401 and service.llm.calls == []


def test_an_unknown_contract_is_a_404_not_a_stream():
    client, keys, service = make_client()

    response = stream(client, keys["user"], contract="No Such Contract")

    assert response.status_code == 404 and "exact title" in response.json()["error"] and service.llm.calls == []


def test_the_rate_limit_and_the_budget_keep_their_statuses():
    client, keys, service = make_client(*good_run(), rpm=1)
    assert stream(client, keys["user"]).status_code == 200

    limited = stream(client, keys["user"])
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) > 0

    client2, keys2, service2 = make_client(*good_run(), budget=0.0001)
    service2.accounts.record(service2.accounts.authenticate(keys2["user"]), "answered", "m", 1, 1, 1.0, 1.0, False, 0, False)
    assert stream(client2, keys2["user"]).status_code == 402


def test_a_malformed_request_is_a_422():
    client, keys, _ = make_client()

    assert client.post("/v1/ask/stream", json={"question": ""}, headers=auth(keys["user"])).status_code == 422


def test_a_model_outage_arrives_as_a_final_answer_event_marked_as_an_error():
    client, keys, _ = make_client(RuntimeError("every provider is down"))

    got = events(stream(client, keys["user"]))

    assert got[-1][0] == "answer" and got[-1][1]["error"] is True and got[-1][1]["stop_reason"] == "error"
    assert "every provider" not in json.dumps(got)


def test_a_cached_answer_streams_started_then_the_answer_with_no_steps():
    client, keys, _ = make_client(*good_run())
    client.post("/v1/ask", json={"question": "How can this be terminated?", "contract": CONTRACT}, headers=auth(keys["user"]))
    # the test cache has a threshold of 0.99 and a constant vector, so the same question hits it

    got = events(stream(client, keys["user"]))

    assert [name for name, _ in got] == ["started", "answer"] and got[-1][1]["usage"]["cached"] is True


def test_a_streamed_question_is_recorded_once_and_remembered():
    client, keys, service = make_client(*good_run())

    final = events(stream(client, keys["user"]))[-1][1]

    assert service.accounts.spend(service.accounts.authenticate(keys["user"])).requests == 1
    assert service.memory.history(final["session_id"])[0][1] == "Ninety days."
