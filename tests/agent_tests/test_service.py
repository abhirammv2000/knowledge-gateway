"""AgentService: limits, memory, cache and bookkeeping around the loop. No API is called."""
import numpy as np
import pytest
from conftest import FakeEmbedder, FakeReranker, FakeStore
from fakes import FakeLLM, call, reply, submit

from agent.accounts import Accounts, Denied
from agent.cache import SemanticCache
from agent.config import Settings
from agent.memory import Memory
from agent.service import AgentService, BadRequest
from gateway.service import GatewayService

CONTRACT = "Acme Services Agreement"
TEXT = "1. Termination\n\nEither party may terminate this agreement on ninety days written notice."
GOOD = {"source_id": "S1", "quote": "terminate this agreement on ninety days written notice"}


def words_vector(text):
    """A bag-of-words vector, so questions that share words are close."""
    vec = np.zeros(64)
    for word in text.lower().split():
        vec[hash(word) % 64] += 1
    return vec


def search():
    return call("search_contract", {"title": CONTRACT, "query": "termination"}, "c1")


def good_run():
    return [reply(search()), reply(submit(answer="Ninety days.", citations=[GOOD]))]


@pytest.fixture
def parts():
    gateway = GatewayService(FakeStore({CONTRACT: TEXT, "Other": "1. Rent\n\nDue monthly."}), FakeEmbedder(), FakeReranker())
    accounts = Accounts()
    plain, key = accounts.create_key("test", daily_budget_usd=5.0, rpm=100)
    return gateway, accounts, key


def make(parts, *script, cache=True, **kw):
    gateway, accounts, key = parts
    llm = FakeLLM(*script)
    service = AgentService(Settings(), llm, gateway, Memory(), accounts,
                           SemanticCache(words_vector, threshold=0.8) if cache else None, **kw)
    return service, llm, key


async def test_a_question_is_answered_recorded_and_remembered(parts):
    service, llm, key = make(parts, *good_run())

    answer = await service.ask(key, "How can this be terminated?", CONTRACT)

    assert answer.found and answer.verified and answer.stop_reason == "answered"
    assert answer.citations[0]["contract"] == CONTRACT and "ninety days" in answer.citations[0]["passage"]
    assert answer.usage["cached"] is False and answer.usage["tool_calls"] == 1
    assert service.accounts.spend(key).requests == 1
    assert service.memory.history(answer.session_id)[0][1] == "Ninety days."


async def test_an_empty_or_huge_question_is_rejected_before_any_cost(parts):
    service, llm, key = make(parts)

    with pytest.raises(BadRequest):
        await service.ask(key, "   ")
    with pytest.raises(BadRequest):
        await service.ask(key, "x" * 5000)
    assert llm.calls == [] and service.accounts.spend(key).requests == 0


async def test_an_unknown_contract_is_a_404(parts):
    service, _, key = make(parts)

    with pytest.raises(BadRequest) as error:
        await service.ask(key, "hello", "No Such Contract")

    assert error.value.status == 404


async def test_a_key_over_its_budget_is_stopped_before_the_model_is_called(parts):
    gateway, accounts, _ = parts
    _, poor = accounts.create_key("poor", daily_budget_usd=0.01, rpm=100)
    accounts.record(poor, "answered", "m", 1, 1, 0.5, 1.0, False, 0, False)
    service, llm, _ = make(parts)

    with pytest.raises(Denied) as denied:
        await service.ask(poor, "question", CONTRACT)

    assert denied.value.status == 402 and llm.calls == []


async def test_a_follow_up_in_the_same_session_sees_the_earlier_turn(parts):
    service, llm, key = make(parts, *good_run(), reply(submit(found=False, answer="Not stated.")))
    first = await service.ask(key, "How can this be terminated?", CONTRACT)

    await service.ask(key, "And what about renewal?", CONTRACT, session_id=first.session_id)

    follow_up = llm.calls[-1]["messages"]
    assert follow_up[1] == {"role": "user", "content": "How can this be terminated?"}
    assert follow_up[2] == {"role": "assistant", "content": "Ninety days."}


async def test_someone_elses_session_is_a_404(parts):
    gateway, accounts, key = parts
    _, other = accounts.create_key("other", daily_budget_usd=5, rpm=100)
    service, _, _ = make(parts, *good_run())
    first = await service.ask(key, "How can this be terminated?", CONTRACT)

    with pytest.raises(BadRequest) as error:
        await service.ask(other, "What did they ask?", CONTRACT, session_id=first.session_id)

    assert error.value.status == 404


async def test_a_note_the_model_saves_is_shown_in_a_later_session(parts):
    note = call("save_note", {"text": "The user cares about termination."}, "n1")
    service, llm, key = make(parts, reply(note), reply(submit(found=False, answer="Noted.")),
                             reply(submit(found=False, answer="x")))
    await service.ask(key, "Remember I care about termination.", CONTRACT)

    await service.ask(key, "Anything else?", CONTRACT)  # a new session, same matter

    assert "The user cares about termination." in llm.calls[-1]["messages"][0]["content"]


# cache

async def test_the_same_question_again_is_a_free_cache_hit(parts):
    service, llm, key = make(parts, *good_run())
    first = await service.ask(key, "How can this be terminated?", CONTRACT)

    second = await service.ask(key, "How can this be terminated?", CONTRACT)

    assert second.usage["cached"] is True and second.usage["cost_usd"] == 0.0
    assert second.answer == first.answer and second.citations == first.citations
    assert len(llm.calls) == 2  # only the first question reached the model
    assert second.session_id != first.session_id


async def test_a_reworded_question_can_hit_the_cache(parts):
    service, llm, key = make(parts, *good_run())
    await service.ask(key, "how can this agreement be terminated", CONTRACT)

    second = await service.ask(key, "how can this agreement be terminated early", CONTRACT)

    assert second.usage["cached"] is True and second.usage["cache_similarity"] >= 0.8


async def test_a_different_question_misses_the_cache(parts):
    service, llm, key = make(parts, *good_run(), reply(submit(found=False, answer="No fees section.")))
    await service.ask(key, "how can this agreement be terminated", CONTRACT)

    second = await service.ask(key, "what are the payment fees due", CONTRACT)

    assert second.usage["cached"] is False


async def test_a_cache_hit_is_not_shared_across_contracts(parts):
    service, llm, key = make(parts, *good_run(), reply(submit(found=False, answer="x")))
    await service.ask(key, "how can this agreement be terminated", CONTRACT)

    other = await service.ask(key, "how can this agreement be terminated", "Other")

    assert other.usage["cached"] is False


async def test_no_cache_without_a_contract_or_in_a_follow_up(parts):
    service, llm, key = make(parts, *[reply(submit(found=False, answer=x)) for x in "abcd"])
    await service.ask(key, "what is this", None)
    second = await service.ask(key, "what is this", None)
    first = await service.ask(key, "how long is the term", CONTRACT)

    again = await service.ask(key, "how long is the term", CONTRACT, session_id=first.session_id)

    assert not second.usage["cached"]  # no contract, so nothing is cached
    assert service.cache.stats()["entries"] == 1  # only the first CONTRACT question was stored
    assert not again.usage["cached"]  # it has earlier turns, so its answer depends on them
    assert len(llm.calls) == 4


async def test_a_failed_or_refused_run_is_never_cached(parts):
    bad = lambda i: reply(submit(answer="Thirty days.", citations=[{"source_id": "S1", "quote": "thirty days"}],
                                 call_id=f"s{i}"))
    service, llm, key = make(parts, reply(search()), bad(1), bad(2), bad(3))

    answer = await service.ask(key, "how can this agreement be terminated", CONTRACT)

    assert answer.stop_reason == "refused_unverified"
    assert service.cache.stats()["entries"] == 0


async def test_a_broken_cache_does_not_break_answering(parts, monkeypatch):
    service, llm, key = make(parts, *good_run())
    monkeypatch.setattr(service.cache, "get", lambda *a: (_ for _ in ()).throw(RuntimeError("db locked")))
    monkeypatch.setattr(service.cache, "put", lambda *a: (_ for _ in ()).throw(RuntimeError("db locked")))

    answer = await service.ask(key, "how can this agreement be terminated", CONTRACT)

    assert answer.found


async def test_the_cache_stores_the_redacted_question_only(parts):
    service, llm, key = make(parts, *good_run())

    await service.ask(key, "Our contact jane.roe@example.com asks how this can be terminated", CONTRACT)

    stored = service.cache._db.execute("SELECT question FROM answers").fetchone()[0]
    assert "jane.roe@example.com" not in stored
    assert "jane.roe@example.com" not in service.memory._db.execute("SELECT question FROM turns").fetchone()[0]


async def test_a_model_outage_is_reported_and_costs_nothing(parts):
    service, llm, key = make(parts, RuntimeError("all providers down"))

    answer = await service.ask(key, "how can this agreement be terminated", CONTRACT)

    assert answer.stop_reason == "error" and answer.usage["cost_usd"] == 0
    assert "all providers down" not in answer.answer
