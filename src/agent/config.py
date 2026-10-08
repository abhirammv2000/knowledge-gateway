"""Settings for the contract review agent, all from environment variables.

Model names use LiteLLM's provider/model form. API keys are read by LiteLLM from the
usual variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _list(name: str, default: str) -> list[str]:
    return [m.strip() for m in os.environ.get(name, default).split(",") if m.strip()]


@dataclass(frozen=True)
class Settings:
    # 0 makes answers repeatable. Newer Claude models reject the parameter, so it is only sent to the others.
    temperature: float = field(default_factory=lambda: _float("AGENT_TEMPERATURE", 0.0))

    # the model that answers, then the ones tried in order if it fails
    primary_model: str = field(default_factory=lambda: os.environ.get("AGENT_PRIMARY_MODEL", "anthropic/claude-sonnet-5-5"))
    fallback_models: list[str] = field(
        default_factory=lambda: _list("AGENT_FALLBACK_MODELS", "openai/gpt-4o,gemini/gemini-3.6-flash")
    )

    # A/B test: this share of new conversations (0 to 100) is answered by the challenger model instead of
    # the primary. A conversation stays on the model it started with.
    ab_model: str = field(default_factory=lambda: os.environ.get("AGENT_AB_MODEL", ""))
    ab_percent: int = field(default_factory=lambda: min(100, max(0, _int("AGENT_AB_PERCENT", 0))))

    # Local open-weight models through Ollama (model names like ollama_chat/qwen2.5-coder:7b). Ollama's own default
    # context is 2048 tokens, which cuts off a contract passage and the tool results, so ask for more.
    ollama_num_ctx: int = field(default_factory=lambda: _int("AGENT_OLLAMA_NUM_CTX", 8192))
    ollama_api_base: str = field(default_factory=lambda: os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"))

    # one model call may take this long, and is retried this many times before falling back
    request_timeout_seconds: float = field(default_factory=lambda: _float("AGENT_REQUEST_TIMEOUT", 60.0))
    num_retries: int = field(default_factory=lambda: _int("AGENT_NUM_RETRIES", 1))
    # a model that fails this many times in a row is skipped for the cooldown
    allowed_fails: int = field(default_factory=lambda: _int("AGENT_ALLOWED_FAILS", 3))
    cooldown_seconds: float = field(default_factory=lambda: _float("AGENT_COOLDOWN_SECONDS", 30.0))

    # limits on one question, so a confused model cannot loop or spend without end
    max_iterations: int = field(default_factory=lambda: _int("AGENT_MAX_ITERATIONS", 8))
    max_tool_calls: int = field(default_factory=lambda: _int("AGENT_MAX_TOOL_CALLS", 12))
    max_run_tokens: int = field(default_factory=lambda: _int("AGENT_MAX_RUN_TOKENS", 80_000))
    max_run_seconds: float = field(default_factory=lambda: _float("AGENT_MAX_RUN_SECONDS", 120.0))
    max_citation_repairs: int = field(default_factory=lambda: _int("AGENT_MAX_CITATION_REPAIRS", 2))
    # identical tool calls tolerated in one run before the model may only submit an answer
    max_repeated_calls: int = field(default_factory=lambda: _int("AGENT_MAX_REPEATED_CALLS", 2))
    max_question_chars: int = field(default_factory=lambda: _int("AGENT_MAX_QUESTION_CHARS", 2000))

    # Turn the defences against hostile contract text off. Only the prompt-injection eval does this,
    # to measure what the defences are worth. It must stay on in anything real.
    harden: bool = field(default_factory=lambda: os.environ.get("AGENT_HARDEN", "1") != "0")

    # a secret-looking string placed in the system prompt. If it ever shows up in an
    # answer, something got the model to leak its instructions.
    canary: str = field(default_factory=lambda: os.environ.get("AGENT_CANARY", "KG-CANARY-7f3a91c2"))


def get_settings() -> Settings:
    return Settings()
