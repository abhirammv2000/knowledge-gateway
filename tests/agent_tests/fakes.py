"""A scripted stand-in for the language model, and helpers to build the replies it gives."""
from __future__ import annotations

import copy
import json
from typing import Any, Callable

from agent.llm import LLMReply, ToolCall


def call(name: str, arguments: dict[str, Any] | str, call_id: str = "call_1") -> ToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(call_id, name, raw)


def reply(*calls: ToolCall, text: str = "", model: str = "fake-model", input_tokens: int = 100,
          output_tokens: int = 20, cost: float = 0.001, fallback: bool = False) -> LLMReply:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls
        ]
    return LLMReply(text=text, tool_calls=list(calls), model=model, input_tokens=input_tokens,
                    output_tokens=output_tokens, cost_usd=cost, seconds=0.01, fallback_used=fallback, message=message)


def submit(found=True, answer="An answer.", citations=None, call_id="call_s", **extra) -> ToolCall:
    return call("submit_answer", {"found": found, "answer": answer, "citations": citations or [], **extra}, call_id)


class FakeLLM:
    """Gives the scripted replies in order. A reply can also be a function of the messages so far, or an exception."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages, tools, tool_choice=None):
        # keep a copy, because the loop keeps adding to the same list
        self.calls.append({"messages": copy.deepcopy(messages), "tools": [t["function"]["name"] for t in tools]})
        if not self.script:
            raise AssertionError("the model was called more times than the test scripted")
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            step = step(messages)
        return step

    @property
    def all_text(self) -> str:
        """Everything that was ever sent to the model, as one string."""
        return json.dumps([c["messages"] for c in self.calls])


def last_tool_message(llm_call: dict[str, Any]) -> str:
    return next(m["content"] for m in reversed(llm_call["messages"]) if m["role"] == "tool")


__all__ = ["FakeLLM", "call", "reply", "submit", "last_tool_message", "Callable"]
