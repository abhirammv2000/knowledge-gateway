"""Where does the time go in one question? Runs questions one at a time and splits the wait into
model calls, tool calls (search and listing) and everything else (redaction, bookkeeping).

    PYTHONPATH=src python eval/latency_eval.py --model openai/gpt-4o --n 16
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from agent.config import Settings  # noqa: E402
from agent.llm import RouterLLM  # noqa: E402
from agent.loop import run_agent  # noqa: E402
from agent_eval import build_questions  # noqa: E402
from gateway.service import GatewayService  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=16)
    args = parser.parse_args()

    settings = Settings()
    service = GatewayService()
    llm = RouterLLM(settings, models=[args.model], with_fallbacks=False)
    questions = build_questions(100, 7)[: args.n]

    # warm the models so the first question does not pay for loading them
    service.list_contracts()
    service.search_contract(questions[0]["contract"], "term", 1)

    rows = []
    for q in questions:
        index_cold = q["contract"] not in service._indexes
        started = time.monotonic()
        result = await run_agent(q["question"], llm=llm, service=service, settings=settings, contract=q["contract"])
        total = time.monotonic() - started
        if result.stop_reason in ("error", "timeout"):
            sys.exit("a provider failure, so the timings would mean nothing. Check the key and credit.")
        tools = sum(e.seconds for e in result.tool_events)
        rows.append({"total": total, "llm": result.llm_seconds, "tools": tools,
                     "other": max(0.0, total - result.llm_seconds - tools), "cold": index_cold,
                     "iterations": result.iterations, "tokens": result.input_tokens + result.output_tokens})

    def line(label, group):
        if not group:
            return
        m = lambda k: statistics.mean(r[k] for r in group)
        print(f"{label:22} n={len(group):3d}  total {m('total'):5.1f}s = model {m('llm'):5.1f}s + tools {m('tools'):5.1f}s "
              f"+ other {m('other'):4.1f}s   iterations {m('iterations'):.1f}   tokens {m('tokens'):.0f}")

    print(args.model)
    line("all questions", rows)
    line("index already built", [r for r in rows if not r["cold"]])
    line("index built on demand", [r for r in rows if r["cold"]])
    totals = sorted(r["total"] for r in rows)
    print(f"p50 {statistics.median(totals):.1f}s   p95 {totals[int(0.95 * (len(totals) - 1))]:.1f}s   max {totals[-1]:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
