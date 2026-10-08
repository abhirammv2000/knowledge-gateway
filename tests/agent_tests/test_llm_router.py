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


def test_temperature_is_sent_to_every_model_except_anthropic():
    from agent.llm import deployment_params

    settings = Settings(temperature=0.0)

    assert deployment_params(settings, "openai/gpt-4o") == {"model": "openai/gpt-4o", "temperature": 0.0}
    assert deployment_params(settings, "gemini/gemini-3.6-flash")["temperature"] == 0.0
    assert "temperature" not in deployment_params(settings, "anthropic/claude-sonnet-5-5")


async def test_the_built_router_carries_the_temperature():
    from agent.llm import build_router

    router = build_router(Settings(temperature=0.0), ["openai/gpt-4o", "anthropic/claude-sonnet-5-5"])
    params = {d["model_name"]: d["litellm_params"] for d in router.get_model_list()}

    assert params["m0"]["temperature"] == 0.0 and "temperature" not in params["m1"]


def test_a_local_ollama_model_gets_the_server_address_and_a_bigger_context():
    from agent.llm import deployment_params

    settings = Settings(temperature=0.0, ollama_num_ctx=8192, ollama_api_base="http://localhost:11434")

    assert deployment_params(settings, "ollama_chat/qwen2.5-coder:7b") == {
        "model": "ollama_chat/qwen2.5-coder:7b", "temperature": 0.0,
        "api_base": "http://localhost:11434", "num_ctx": 8192}
    assert deployment_params(settings, "ollama/llama3.2:3b")["num_ctx"] == 8192


def test_hosted_models_get_no_ollama_settings():
    from agent.llm import deployment_params

    for model in ("openai/gpt-4o", "gemini/gemini-3.6-flash", "anthropic/claude-sonnet-5-5"):
        params = deployment_params(Settings(), model)
        assert "api_base" not in params and "num_ctx" not in params


# provider prompt caching

def test_only_the_anthropic_deployment_asks_for_a_cache_marker_on_the_system_prompt():
    from agent.llm import deployment_params

    settings = Settings()

    assert deployment_params(settings, "anthropic/claude-sonnet-5-5")["cache_control_injection_points"] == [
        {"location": "message", "role": "system"}]
    for model in ("openai/gpt-4o", "gemini/gemini-3.6-flash", "ollama_chat/qwen2.5-coder:7b"):
        assert "cache_control_injection_points" not in deployment_params(settings, model)


def test_prompt_caching_can_be_switched_off():
    from agent.llm import deployment_params

    assert "cache_control_injection_points" not in deployment_params(Settings(prompt_caching=False), "anthropic/claude-sonnet-5-5")


async def test_the_request_to_anthropic_carries_the_marker_on_the_system_block(monkeypatch):
    """Intercepts the HTTP call, so nothing is sent. Checks what LiteLLM would send to Anthropic."""
    from litellm.llms.custom_httpx import http_handler

    sent = {}

    async def capture(self, *args, **kwargs):
        sent["body"] = kwargs.get("json") or kwargs.get("data")
        raise RuntimeError("stop before the network")

    monkeypatch.setattr(http_handler.AsyncHTTPHandler, "post", capture)
    from agent.llm import build_router

    router = build_router(Settings(primary_model="anthropic/claude-sonnet-5-5"), models=["anthropic/claude-sonnet-5-5"],
                          with_fallbacks=False)
    with pytest.raises(Exception):
        await router.acompletion(model="m0", api_key="sk-test", max_tokens=5, num_retries=0,
                                 messages=[{"role": "system", "content": "rules " * 40}, {"role": "user", "content": "hi"}])

    import json
    body = json.loads(sent["body"]) if isinstance(sent["body"], (str, bytes)) else sent["body"]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in json.dumps(body["messages"])


def test_cached_input_tokens_are_read_from_the_usage_block():
    from types import SimpleNamespace

    from agent.llm import _cached_tokens

    assert _cached_tokens(SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=900))) == 900
    assert _cached_tokens(SimpleNamespace(prompt_tokens_details=None)) == 0
    assert _cached_tokens(SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=None))) == 0
    assert _cached_tokens(SimpleNamespace()) == 0
    assert _cached_tokens(None) == 0
