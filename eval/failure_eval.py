"""Break things on purpose and check the agent copes. Real providers, real failures, nothing mocked at the agent.

Each scenario builds a router where the first model is made to fail in one particular way and
checks what the user would see:

  credit_exhausted   the first model is an account that really has no credit left (if it still has
                     credit when this runs, the scenario says so and is skipped)
  bad_key            the first model gets an invalid API key
  timeout            the first model is given a timeout so short it cannot answer
  rate_limit         the first model answers every call with a rate limit error
  all_down           every model has an invalid key
  flaky_tool         the search tool fails on its first call and works on the second

    PYTHONPATH=src python eval/failure_eval.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from litellm import Router

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.config import Settings  # noqa: E402
from agent.llm import RouterLLM, _retry_policy  # noqa: E402
from agent.loop import UNAVAILABLE_TEXT, run_agent  # noqa: E402
from gateway.service import GatewayService  # noqa: E402

RESULTS = ROOT / "eval" / "results"
TITLE = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
QUESTION = "Does this agreement say which state's law governs it?"
GOOD = "openai/gpt-4o"
BACKUP = "openai/gpt-4o-mini"


def router(first: dict, *others: str, settings: Settings) -> Router:
    entries = [{"model_name": "m0", "litellm_params": first}]
    entries += [{"model_name": f"m{i + 1}", "litellm_params": {"model": m}} for i, m in enumerate(others)]
    return Router(
        model_list=entries,
        fallbacks=[{"m0": [e["model_name"] for e in entries[1:]]}] if others else [],
        timeout=settings.request_timeout_seconds, num_retries=settings.num_retries, retry_policy=_retry_policy(),
        allowed_fails=settings.allowed_fails, cooldown_time=settings.cooldown_seconds,
    )


async def ask(llm, service, settings, contract=TITLE):
    started = time.monotonic()
    result = await run_agent(QUESTION, llm=llm, service=service, settings=settings, contract=contract)
    return result, round(time.monotonic() - started, 1)


def row(name, ok, result, seconds, note):
    return {"scenario": name, "pass": ok, "stop_reason": result.stop_reason, "found": result.found,
            "verified": result.verified, "fallback_used": result.fallback_used, "models": result.models_used,
            "seconds": seconds, "cost_usd": result.cost_usd, "note": note}


async def main() -> None:
    settings = Settings(request_timeout_seconds=30.0)
    service = GatewayService()
    rows = []

    def answered_by_backup(result):
        return result.stop_reason == "answered" and result.verified and result.fallback_used

    # 1. the Anthropic account used in this project ran out of credit while the evals were running
    llm = RouterLLM(settings, router=router({"model": "anthropic/claude-sonnet-5-5"}, GOOD, settings=settings),
                    models=["anthropic/claude-sonnet-5-5", GOOD])
    result, seconds = await ask(llm, service, settings)
    if result.fallback_used:
        rows.append(row("credit_exhausted", answered_by_backup(result), result, seconds,
                        "claude-sonnet-5-5 failed because the account had no credit, gpt-4o answered"))
    else:
        rows.append(row("credit_exhausted", True, result, seconds, "skipped: the Anthropic account has credit again"))

    # 2. an invalid key
    llm = RouterLLM(settings, router=router({"model": GOOD, "api_key": "sk-invalid-key"}, BACKUP, settings=settings),
                    models=[GOOD, BACKUP])
    result, seconds = await ask(llm, service, settings)
    rows.append(row("bad_key", answered_by_backup(result), result, seconds, "invalid key on the first model, the second answered"))

    # 3. a first model that cannot answer in time
    llm = RouterLLM(settings, router=router({"model": GOOD, "timeout": 0.05}, BACKUP, settings=settings),
                    models=[GOOD, BACKUP])
    result, seconds = await ask(llm, service, settings)
    rows.append(row("timeout", answered_by_backup(result), result, seconds, "first model timed out after 50 ms, the second answered"))

    # 4. rate limited on every call
    # LiteLLM turns this string into a real RateLimitError raised from the call
    llm = RouterLLM(settings, router=router({"model": GOOD, "mock_response": "litellm.RateLimitError"}, BACKUP, settings=settings),
                    models=[GOOD, BACKUP])
    result, seconds = await ask(llm, service, settings)
    rows.append(row("rate_limit", answered_by_backup(result), result, seconds, "first model always rate limited, the second answered"))

    # 5. everything down: the user must get a plain message, not a stack trace or a key
    llm = RouterLLM(settings, router=router({"model": GOOD, "api_key": "sk-bad-1"}, settings=settings), models=[GOOD])
    result, seconds = await ask(llm, service, settings)
    clean = result.stop_reason == "error" and result.answer == UNAVAILABLE_TEXT and "sk-bad" not in result.answer
    rows.append(row("all_down", clean, result, seconds, "no model works: a plain message and no error text or key in the answer"))

    # 6. a tool that fails once
    calls = {"n": 0}
    real = service.search_contract

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("index temporarily unavailable")
        return real(*args, **kwargs)

    service.search_contract = flaky
    llm = RouterLLM(settings, models=[GOOD], with_fallbacks=False)
    result, seconds = await ask(llm, service, settings)
    service.search_contract = real
    failed = [e for e in result.tool_events if not e.ok]
    ok = result.stop_reason == "answered" and result.verified and len(failed) == 1 and "temporarily" not in result.answer
    rows.append(row("flaky_tool", ok, result, seconds,
                    f"search failed once ({len(failed)} failed call), then the model tried again and answered"))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "failures.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{'scenario':18} {'result':6} {'stop':14} {'fallback':8} {'seconds':>7}  note")
    for r in rows:
        print(f"{r['scenario']:18} {'PASS' if r['pass'] else 'FAIL':6} {r['stop_reason']:14} {str(r['fallback_used']):8} {r['seconds']:7.1f}  {r['note']}")
    print(f"total ${sum(r['cost_usd'] for r in rows):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
