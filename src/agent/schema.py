"""The types the agent passes around: tool arguments, the final answer, and the result of a run."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ListContractsArgs(BaseModel):
    contains: str = Field(default="", max_length=200)
    limit: int = Field(default=25, ge=1, le=50)


class SearchContractArgs(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class SaveNoteArgs(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class Citation(BaseModel):
    source_id: str = Field(description="The id of a passage you were shown, like S1.")
    quote: str = Field(min_length=1, description="Words copied exactly from that passage.")


class SubmitAnswerArgs(BaseModel):
    """What the model sends to finish. The server checks every citation before it is trusted."""

    found: bool = Field(description="True only if the contract contains what the question asks about.")
    answer: str = Field(min_length=1, max_length=4000)
    citations: list[Citation] = Field(default_factory=list, max_length=8)
    confidence: Literal["high", "medium", "low"] = "medium"
    contract: str | None = Field(default=None, description="Exact title of the contract the answer is about.")

    @field_validator("answer")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer is blank")
        return value


@dataclass
class Passage:
    """A piece of contract text the model was shown. The text is the redacted version."""

    source_id: str
    contract: str
    chunk_index: int
    text: str


@dataclass
class VerifiedCitation:
    source_id: str
    contract: str
    chunk_index: int
    quote: str
    passage: str


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    ok: bool
    error: str = ""
    seconds: float = 0.0


@dataclass
class AgentResult:
    found: bool
    answer: str
    citations: list[VerifiedCitation] = field(default_factory=list)
    # True when every citation was checked against the passages the model saw
    verified: bool = False
    confidence: str = "low"
    contract: str | None = None
    # answered, refused_unverified, max_iterations, budget, timeout, no_answer, leak_blocked, error
    stop_reason: str = "answered"
    iterations: int = 0
    tool_events: list[ToolEvent] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    llm_seconds: float = 0.0  # time spent waiting for the model, as the router measured it
    models_used: list[str] = field(default_factory=list)
    fallback_used: bool = False
    repeated_calls: int = 0
    cached: bool = False
    # the question after redaction. This is the only form of it that may be stored.
    question_redacted: str = ""
    # passages the model was shown, by id. Not part of the API response, used by the evals.
    passages: dict[str, Passage] = field(default_factory=dict)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_events)
