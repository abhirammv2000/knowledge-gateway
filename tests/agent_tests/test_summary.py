"""A long conversation: the last turns word for word, older ones folded into a running summary. A scripted model."""
import pytest
from conftest import FakeEmbedder, FakeReranker, FakeStore
from fakes import FakeLLM, reply, submit

from agent.accounts import Accounts
from agent.config import Settings
from agent.llm import RouterLLM
from agent.loop import SUMMARY_PROMPT
from agent.memory import Memory
from agent.service import AgentService
from gateway.service import GatewayService

CONTRACT = "Acme Services Agreement"


def answer(n):
    return reply(submit(found=False, answer=f"Answer {n}.", call_id=f"s{n}"))


def make(script, **settings):
    gateway = GatewayService(FakeStore({CONTRACT: "1. Rent\n\nRent is due monthly."}), FakeEmbedder(), FakeReranker())
    accounts = Accounts()
    _, key = accounts.create_key("k", daily_budget_usd=5.0, rpm=1000)
    llm = FakeLLM(*script)
    service = AgentService(Settings(**settings), llm, gateway, Memory(), accounts, None)
    return service, llm, key


async def converse(service, key, turns):
    session = None
    for n in range(1, turns + 1):
        result = await service.ask(key, f"Question {n}?", CONTRACT, session_id=session)
        session = result.session_id
    return session


def summary_calls(llm):
    return [c for c in llm.calls if c["messages"][0]["content"] == SUMMARY_PROMPT]


async def test_a_short_conversation_never_makes_a_summary():
    service, llm, key = make([answer(n) for n in range(1, 8)])

    session = await converse(service, key, 7)

    assert summary_calls(llm) == [] and service.memory.summary(session) == ""


async def test_once_three_turns_fall_out_of_the_window_they_are_folded_into_a_summary():
    # turns 1 to 8: the window holds 4 to 8, so turns 1 to 3 are old enough to summarise after turn 8
    script = [answer(n) for n in range(1, 9)]
    script.insert(8, reply(text="The user asked about rent. Nothing found."))
    service, llm, key = make(script)

    session = await converse(service, key, 8)

    assert len(summary_calls(llm)) == 1
    assert service.memory.summary(session) == "The user asked about rent. Nothing found."
    folded = summary_calls(llm)[0]["messages"][1]["content"]
    assert "Question 1?" in folded and "Question 3?" in folded and "Question 4?" not in folded


async def test_the_next_question_sees_the_summary_then_the_last_turns():
    script = [answer(n) for n in range(1, 9)]
    script.insert(8, reply(text="Earlier: rent was discussed."))
    script.append(answer(9))
    service, llm, key = make(script)
    session = await converse(service, key, 8)

    await service.ask(key, "Question 9?", CONTRACT, session_id=session)

    messages = llm.calls[-1]["messages"]
    texts = [m["content"] for m in messages if isinstance(m["content"], str)]
    assert any("summarised" in t and "Earlier: rent was discussed." in t for t in texts)
    assert "Question 4?" in texts and "Question 8?" in texts and "Question 3?" not in texts


async def test_the_summary_is_updated_with_the_previous_one_not_started_again():
    # a summary call follows the answer to turn 8, and another follows the answer to turn 11
    script = [answer(n) for n in range(1, 9)] + [reply(text="Summary one.")]
    script += [answer(n) for n in range(9, 12)] + [reply(text="Summary two.")]
    service, llm, key = make(script)

    session = await converse(service, key, 11)

    second = summary_calls(llm)[1]["messages"][1]["content"]
    assert "Summary so far:\nSummary one." in second and "Question 4?" in second and "Question 6?" in second
    assert service.memory.summary(session) == "Summary two."


async def test_a_failed_summary_leaves_the_conversation_working_and_tries_again_next_turn():
    script = [answer(n) for n in range(1, 9)]
    script.insert(8, RuntimeError("provider down"))
    script += [answer(9), reply(text="Now it works.")]
    service, llm, key = make(script)
    session = await converse(service, key, 8)
    assert service.memory.summary(session) == ""

    await service.ask(key, "Question 9?", CONTRACT, session_id=session)

    assert service.memory.summary(session) == "Now it works."


async def test_an_empty_summary_is_not_stored():
    script = [answer(n) for n in range(1, 9)]
    script.insert(8, reply(text="   "))
    service, llm, key = make(script)

    session = await converse(service, key, 8)

    assert service.memory.summary(session) == ""


async def test_a_batch_of_zero_turns_the_summary_off():
    service, llm, key = make([answer(n) for n in range(1, 11)], summarize_batch=0)

    session = await converse(service, key, 10)

    assert summary_calls(llm) == [] and service.memory.summary(session) == ""


async def test_the_summary_call_is_billed_but_not_counted_as_a_question():
    script = [answer(n) for n in range(1, 9)]
    script.insert(8, reply(text="A summary.", input_tokens=700, output_tokens=60, cost=0.004))
    service, llm, key = make(script)

    await converse(service, key, 8)

    assert service.accounts.spend(key).requests == 8
    assert service.accounts.spent_today(key.id) == pytest.approx(8 * 0.001 + 0.004)
    assert "summarized" not in service.accounts.by_arm()  # arms hold request outcomes only
    assert service.accounts.by_arm()["control"]["requests"] == 8


async def test_personal_data_in_a_summary_is_removed_before_it_is_stored():
    script = [answer(n) for n in range(1, 9)]
    script.insert(8, reply(text="The client is reachable at jane.roe@example.com."))
    service, llm, key = make(script)

    session = await converse(service, key, 8)

    assert "jane.roe@example.com" not in service.memory.summary(session)


async def test_the_summary_is_capped():
    script = [answer(n) for n in range(1, 9)]
    script.insert(8, reply(text="word " * 1000))
    service, llm, key = make(script, summary_max_chars=200)

    session = await converse(service, key, 8)

    assert len(service.memory.summary(session)) <= 200


async def test_a_summary_belongs_to_its_session_and_goes_with_it():
    script = [answer(n) for n in range(1, 9)]
    script.insert(8, reply(text="Private summary."))
    service, llm, key = make(script)
    session = await converse(service, key, 8)

    service.memory.delete_session(session, key.id)

    assert service.memory.summary(session) == ""


def test_overflow_lists_only_old_turns_the_summary_does_not_cover_yet():
    memory = Memory()
    session = memory.create_session("k", "m")
    for n in range(1, 10):
        memory.add_turn(session, f"q{n}", f"a{n}", True, None)

    first = memory.overflow(session)
    memory.set_summary(session, "s", first[1][0])  # covers the first two

    assert [q for _, q, _ in first] == ["q1", "q2", "q3", "q4"]
    assert [q for _, q, _ in memory.overflow(session)] == ["q3", "q4"]


def test_an_older_database_without_the_summary_columns_is_upgraded(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE sessions (id TEXT PRIMARY KEY, api_key_id TEXT NOT NULL, matter TEXT NOT NULL,"
                     " created REAL NOT NULL, arm TEXT NOT NULL DEFAULT 'control');"
                     " INSERT INTO sessions (id, api_key_id, matter, created) VALUES ('ses_old', 'k', 'm', 1.0);")
    db.commit()
    db.close()

    memory = Memory(path)

    assert memory.summary("ses_old") == ""


async def test_a_call_with_no_tools_sends_no_tools_list():
    from agent.llm import build_router

    class Capture:
        kwargs = None

        async def acompletion(self, **kwargs):
            Capture.kwargs = kwargs
            raise RuntimeError("stop")

    llm = RouterLLM(Settings(), router=build_router(Settings(), models=["openai/gpt-4o"], with_fallbacks=False))
    llm.router = Capture()
    with pytest.raises(RuntimeError):
        await llm.complete([{"role": "user", "content": "x"}], [])

    assert "tools" not in Capture.kwargs
