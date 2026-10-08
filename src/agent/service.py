"""One question, start to finish: limits, memory, cache, the agent run, and the bookkeeping after.

The web layer calls ask() and nothing else, so everything that matters can be tested without HTTP.
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from agent import metrics
from agent.accounts import Accounts, ApiKey
from agent.cache import SemanticCache
from agent.config import Settings
from agent.idempotency import IdempotencyStore
from agent.llm import LLM
from agent.loop import run_agent
from agent.memory import Memory
from agent.schema import AgentResult
from gateway.service import GatewayService
from gateway.tracing import get_tracer

log = logging.getLogger("agent.service")

# Change this when the system prompt or the tools change, so cached answers from the old
# behaviour are not served by the new one.
PROMPT_VERSION = "1"
GENERAL_MATTER = "general"


class BadRequest(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass
class Answer:
    session_id: str
    found: bool
    answer: str
    verified: bool
    confidence: str
    contract: str | None
    citations: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, Any] = field(default_factory=dict)
    # notes the model proposed during this question, waiting for the user to approve them
    notes_proposed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentService:
    def __init__(self, settings: Settings, llm: LLM, gateway: GatewayService, memory: Memory,
                 accounts: Accounts, cache: SemanticCache | None = None, max_concurrent: int = 4,
                 challenger: LLM | None = None, rng: random.Random | None = None,
                 idempotency: IdempotencyStore | None = None) -> None:
        self.settings = settings
        self.idempotency = idempotency or IdempotencyStore()
        self.llm = llm
        self.challenger = challenger
        self._rng = rng or random.Random()
        self.gateway = gateway
        self.memory = memory
        self.accounts = accounts
        self.cache = cache
        self._slots = asyncio.Semaphore(max_concurrent)
        self.model_key = f"{settings.primary_model}|{PROMPT_VERSION}"
        # an answer cached from one model is not served as another model's answer, or the test would mix
        self._challenger_key = f"{settings.ab_model}|{PROMPT_VERSION}"

    async def ask(self, key: ApiKey, question: str, contract: str | None = None,
                  session_id: str | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise BadRequest("question is empty")
        if len(question) > self.settings.max_question_chars:
            raise BadRequest(f"question is longer than {self.settings.max_question_chars} characters")
        if contract is not None and contract not in await asyncio.to_thread(self._titles):
            raise BadRequest("no contract has that exact title. Use /v1/contracts to find it.", 404)

        self.accounts.check(key)  # raises Denied

        if session_id:
            matter = self.memory.session_matter(session_id, key.id)
            if matter is None:
                raise BadRequest("no such session", 404)
        else:
            matter = contract or GENERAL_MATTER
            session_id = self.memory.create_session(key.id, matter, self._new_arm())
        history = self.memory.history(session_id)

        async with self._slots:
            metrics.ACTIVE.inc()
            try:
                with get_tracer().start_as_current_span("agent.request"):
                    answer = await self._answer(key, question, contract, session_id, matter, history)
            finally:
                metrics.ACTIVE.dec()
        return answer

    def _new_arm(self) -> str:
        if self.challenger is not None and self._rng.random() * 100 < self.settings.ab_percent:
            return "challenger"
        return "control"

    def _titles(self) -> list[str]:
        return self.gateway.store.titles()

    async def _answer(self, key, question, contract, session_id, matter, history) -> Answer:
        arm = self.memory.session_arm(session_id)
        on_challenger = arm == "challenger" and self.challenger is not None
        llm = self.challenger if on_challenger else self.llm
        model_key = self._challenger_key if on_challenger else self.model_key
        cacheable = self.cache is not None and contract is not None and not history
        if cacheable:
            hit = await self._cache_get(question, contract, model_key)
            metrics.CACHE.labels("hit" if hit else "miss").inc()
            if hit:
                return self._from_cache(key, question, contract, session_id, hit.answer, hit.similarity, arm)
        else:
            metrics.CACHE.labels("skipped").inc()

        result = await run_agent(
            question, llm=llm, service=self.gateway, settings=self.settings, contract=contract, history=history,
            notes=self.memory.notes(key.id, matter) or None,
            save_note=lambda text: self.memory.add_note(key.id, matter, text, approved=False),
        )
        self._record(key, result, arm)
        self.memory.add_turn(session_id, result.question_redacted or question, result.answer, result.found,
                             result.contract or contract)

        answer = self._to_answer(session_id, result, arm)
        if cacheable and result.stop_reason == "answered" and result.verified:
            await self._cache_put(result.question_redacted or question, contract, answer, model_key)
        return answer

    # cache

    async def _cache_get(self, question: str, contract: str, model_key: str):
        try:
            return await asyncio.to_thread(self.cache.get, question, contract, model_key)
        except Exception:
            log.exception("cache lookup failed")
            return None

    async def _cache_put(self, question: str, contract: str, answer: Answer, model_key: str) -> None:
        stored = answer.to_dict()
        stored.pop("session_id", None)
        stored.pop("usage", None)
        try:
            await asyncio.to_thread(self.cache.put, question, contract, model_key, stored)
        except Exception:
            log.exception("cache store failed")

    def _from_cache(self, key, question, contract, session_id, stored, similarity, arm) -> Answer:
        metrics.REQUESTS.labels("cached").inc()
        self.accounts.record(key, "cached", None, 0, 0, 0.0, 0.0, True, 0, False, arm)
        self.memory.add_turn(session_id, question, stored["answer"], stored["found"], stored.get("contract") or contract)
        return Answer(
            session_id=session_id, found=stored["found"], answer=stored["answer"], verified=stored["verified"],
            confidence=stored["confidence"], contract=stored.get("contract"), citations=stored["citations"],
            stop_reason="answered",
            usage={"cached": True, "arm": arm, "cache_similarity": similarity, "cost_usd": 0.0, "tokens": 0, "seconds": 0.0,
                   "model": None, "tool_calls": 0, "fallback_used": False},
        )

    # bookkeeping

    def _to_answer(self, session_id: str, result: AgentResult, arm: str) -> Answer:
        return Answer(
            session_id=session_id, found=result.found, answer=result.answer, verified=result.verified,
            confidence=result.confidence, contract=result.contract, stop_reason=result.stop_reason,
            citations=[{"source_id": c.source_id, "contract": c.contract, "quote": c.quote, "passage": c.passage}
                       for c in result.citations],
            usage={"cached": False, "arm": arm, "cost_usd": result.cost_usd, "tokens": result.input_tokens + result.output_tokens,
                   "seconds": result.seconds, "model": result.models_used[-1] if result.models_used else None,
                   "tool_calls": result.tool_call_count, "fallback_used": result.fallback_used,
                   "iterations": result.iterations},
            notes_proposed=sum(1 for e in result.tool_events if e.name == "save_note" and e.ok),
        )

    def _record(self, key: ApiKey, result: AgentResult, arm: str) -> None:
        model = result.models_used[-1] if result.models_used else "none"
        self.accounts.record(key, result.stop_reason, model, result.input_tokens, result.output_tokens,
                             result.cost_usd, result.seconds, False, result.tool_call_count, result.fallback_used, arm,
                             found=result.found if result.stop_reason == "answered" else None)
        metrics.REQUESTS.labels(result.stop_reason).inc()
        metrics.COST.labels(model).inc(result.cost_usd)
        metrics.TOKENS.labels("input").inc(result.input_tokens)
        metrics.TOKENS.labels("output").inc(result.output_tokens)
        metrics.LATENCY.observe(result.seconds)
        if result.fallback_used:
            metrics.FALLBACKS.inc()
        for event in result.tool_events:
            metrics.TOOL_CALLS.labels(event.name, str(event.ok).lower()).inc()
        # lengths and counts only, never the question or the answer
        log.info("ask key=%s arm=%s stop=%s iters=%d tools=%d tokens=%d cost=%.4f seconds=%.1f model=%s fallback=%s",
                 key.id, arm, result.stop_reason, result.iterations, result.tool_call_count,
                 result.input_tokens + result.output_tokens, result.cost_usd, result.seconds, model,
                 result.fallback_used)
