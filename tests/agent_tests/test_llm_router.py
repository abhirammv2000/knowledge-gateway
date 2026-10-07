"""RouterLLM against LiteLLM's mock responses, so no provider is called."""
import pytest
from litellm import Router

from agent.config import Settings
from agent.llm import RouterLLM, _retry_policy
from agent.tools import tool_specs


def router(first, second=None):
    models = [{"model_name": "m0", "litellm_params": first}]
    if second:
        models.append({"model_name": "m1", "litellm_params": second})
    return Router(model_list=models, fallbacks=[{"m0": ["m1"]}] if second else [], num_retries=0,
                  retry_policy=_retry_policy())


async def complete(llm):
    return await llm.complete([{"role": "user", "content": "hi"}], tool_specs())


async def test_a_reply_from_the_first_model_is_not_a_fallback():
    llm = RouterLLM(Settings(), router=router({"model": "openai/gpt-4o", "mock_response": "hello"},
                                              {"model": "openai/gpt-4o-mini", "mock_response": "backup"}))

    reply = await complete(llm)

    assert reply.text == "hello" and reply.fallback_used is False


# LiteLLM's mock can raise only some error types. A bad key and a timeout are covered with real calls in
# eval/failure_eval.py.
@pytest.mark.parametrize("error", ["litellm.RateLimitError", "litellm.InternalServerError"])
async def test_a_failing_first_model_falls_back_and_is_reported(error):
    llm = RouterLLM(Settings(), router=router({"model": "openai/gpt-4o", "mock_response": error},
                                              {"model": "openai/gpt-4o-mini", "mock_response": "backup"}))

    reply = await complete(llm)

    assert reply.text == "backup" and reply.fallback_used is True


async def test_a_fallback_inside_one_model_family_is_still_noticed():
    # gpt-4o-mini starts with gpt-4o, which fooled a check that compared model names
    llm = RouterLLM(Settings(), router=router({"model": "openai/gpt-4o", "mock_response": "litellm.RateLimitError"},
                                              {"model": "openai/gpt-4o-mini", "mock_response": "backup"}))

    reply = await complete(llm)

    assert reply.fallback_used is True


async def test_with_nothing_to_fall_back_to_the_error_reaches_the_caller():
    llm = RouterLLM(Settings(), router=router({"model": "openai/gpt-4o", "mock_response": "litellm.RateLimitError"}))

    with pytest.raises(Exception):
        await complete(llm)


async def test_usage_and_cost_come_back_in_one_reply():
    llm = RouterLLM(Settings(), router=router({"model": "openai/gpt-4o", "mock_response": "hello"}))

    reply = await complete(llm)

    assert reply.input_tokens >= 0 and reply.output_tokens >= 0 and reply.cost_usd >= 0
    assert reply.message["role"] == "assistant" and reply.message["content"] == "hello"
    assert reply.model
