"""Calls the language model through a LiteLLM router.

The router retries a failed call, skips a model that keeps failing for a short cooldown, and
falls back to the next provider. Everything the rest of the agent needs from a reply
(text, tool calls, tokens, cost, which model answered) comes back in one LLMReply, so the loop
never touches a provider response directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import litellm
from litellm import Router
from litellm.router import RetryPolicy

from agent.config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # the JSON text exactly as the model wrote it


@dataclass
class LLMReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    fallback_used: bool = False
    # the assistant message to add to the conversation, in the OpenAI format LiteLLM uses for every provider
    message: dict[str, Any] = field(default_factory=dict)


class LLM(Protocol):
    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Any = None
    ) -> LLMReply: ...


def _retry_policy() -> RetryPolicy:
    # Retry what can pass on a second try. A bad request or a bad key fails the same way again.
    # One retry each: measured on real failures, every extra retry adds about 2 seconds of backoff before
    # the fallback model is tried. With 2 to 3 retries the failover took 5 to 10 seconds, with one about 3.
    return RetryPolicy(
        TimeoutErrorRetries=1,
        RateLimitErrorRetries=1,
        InternalServerErrorRetries=1,
        BadRequestErrorRetries=0,
        AuthenticationErrorRetries=0,
        ContentPolicyViolationErrorRetries=0,
    )


def deployment_params(settings: Settings, model: str) -> dict[str, Any]:
    """The LiteLLM settings for one model. Anthropic's current models refuse a temperature, so they get none."""
    params: dict[str, Any] = {"model": model}
    if not model.startswith("anthropic/"):
        params["temperature"] = settings.temperature
    if model.startswith(("ollama/", "ollama_chat/")):
        # ollama_chat/ is the endpoint that supports tool calls. A local model costs nothing per token.
        params["api_base"] = settings.ollama_api_base
        params["num_ctx"] = settings.ollama_num_ctx
    return params


def build_router(settings: Settings, models: list[str] | None = None, with_fallbacks: bool = True) -> Router:
    """A router for the primary model and its fallbacks, or for an explicit list of models.

    With `models` given and `with_fallbacks` off the router has one model and no safety net,
    which is what the evals use to measure one model by itself.
    """
    chain = models if models is not None else [settings.primary_model, *settings.fallback_models]
    names = [f"m{i}" for i in range(len(chain))]
    kwargs: dict[str, Any] = {}
    if with_fallbacks and len(chain) > 1:
        kwargs["fallbacks"] = [{names[0]: names[1:]}]
    return Router(
        model_list=[{"model_name": n, "litellm_params": deployment_params(settings, m)} for n, m in zip(names, chain)],
        timeout=settings.request_timeout_seconds,
        num_retries=settings.num_retries,
        retry_policy=_retry_policy(),
        allowed_fails=settings.allowed_fails,
        cooldown_time=settings.cooldown_seconds,
        **kwargs,
    )


class RouterLLM:
    def __init__(self, settings: Settings, router: Router | None = None, models: list[str] | None = None,
                 with_fallbacks: bool = True) -> None:
        self.settings = settings
        self.router = router or build_router(settings, models, with_fallbacks)
        # The first model's deployment ids. A reply from any other deployment came from a fallback.
        # Comparing model names would not work: "gpt-4o-mini" starts with "gpt-4o".
        self._primary_ids = {
            d.get("model_info", {}).get("id") for d in self.router.get_model_list(model_name="m0") or []
        }

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Any = None
    ) -> LLMReply:
        started = time.monotonic()
        kwargs: dict[str, Any] = {"model": "m0", "messages": messages, "tools": tools, "max_tokens": 2000}
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        response = await self.router.acompletion(**kwargs)
        return self._reply(response, time.monotonic() - started)

    def _reply(self, response: Any, seconds: float) -> LLMReply:
        message = response.choices[0].message
        calls = [
            ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in (message.tool_calls or [])
        ]
        text = message.content if isinstance(message.content, str) else ""
        assistant: dict[str, Any] = {"role": "assistant", "content": text}
        if message.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"},
                }
                for c in message.tool_calls
            ]
        usage = getattr(response, "usage", None)
        served = getattr(response, "model", "") or ""
        return LLMReply(
            text=text,
            tool_calls=calls,
            model=served,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=_cost_of(response),
            seconds=round(seconds, 3),
            fallback_used=self._served_by_fallback(response),
            message=assistant,
        )


    def _served_by_fallback(self, response: Any) -> bool:
        served_id = (getattr(response, "_hidden_params", None) or {}).get("model_id")
        return bool(served_id) and bool(self._primary_ids) and served_id not in self._primary_ids


def _cost_of(response: Any) -> float:
    hidden = getattr(response, "_hidden_params", None) or {}
    cost = hidden.get("response_cost")
    if cost is None:
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
    return float(cost or 0.0)
