"""Does the agent call the right tools with the right arguments? Graded from the tool trace, not the final text.

Three kinds of question, each with a rule that can be checked without reading the answer:

  exact    the exact contract title is given. The right move is search_contract on that title, with no
           list_contracts and no failed calls.
  partial  only the company name is in the question. The right move is list_contracts with a text that
           is part of the real title, then search_contract on the exact title it returned.
  missing  the contract does not exist. The right move is to look it up and then say it was not found.
           Searching a title that was never listed (a made-up title) is a failure, and so is claiming
           to have found something.

    PYTHONPATH=src python eval/tool_eval.py --model anthropic/claude-sonnet-5-5 --per-kind 12
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.config import Settings  # noqa: E402
from agent.llm import RouterLLM  # noqa: E402
from agent.loop import run_agent  # noqa: E402
from gateway.service import GatewayService  # noqa: E402

RESULTS = ROOT / "eval" / "results"
MISSING_NAMES = ["Zephyr Quantum Holdings", "Blue Harbor Maritime Trust", "Northwind Orchard Cooperative",
                 "Silverpine Robotics", "Halcyon Textile Partners", "Marrow Creek Mining", "Oakridge Dental Group",
                 "Tidewater Analytics", "Ember Valley Brewing", "Pinnacle Glass Works", "Juniper Satellite Services",
                 "Granite Harbor Foods"]
TOPICS = ["the governing law", "how it can be terminated", "the payment terms", "the term of the agreement"]


def company(title: str) -> str:
    return title.split("_")[0]


def build_cases(per_kind: int, seed: int, titles: list[str]) -> list[dict]:
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for t in titles:
        counts[company(t)] = counts.get(company(t), 0) + 1
    unique_company = [t for t in titles if counts[company(t)] == 1 and len(company(t)) >= 6]
    rng.shuffle(unique_company)
    exact = rng.sample(titles, per_kind)
    cases = []
    for i, title in enumerate(exact):
        cases.append({"kind": "exact", "title": title, "contract": title,
                      "question": f"What is {TOPICS[i % len(TOPICS)]}?"})
    for i, title in enumerate(unique_company[:per_kind]):
        cases.append({"kind": "partial", "title": title, "contract": None,
                      "question": f"In the {company(title)} agreement, what is {TOPICS[i % len(TOPICS)]}?"})
    for i in range(per_kind):
        name = MISSING_NAMES[i % len(MISSING_NAMES)]
        cases.append({"kind": "missing", "title": None, "contract": None,
                      "question": f"In the {name} agreement, what is {TOPICS[i % len(TOPICS)]}?"})
    return cases


def grade(case: dict, result, titles: set[str]) -> dict:
    events = result.tool_events
    searches = [e for e in events if e.name == "search_contract"]
    lists = [e for e in events if e.name == "list_contracts"]
    failed = [e for e in events if not e.ok]
    made_up = [e for e in searches if e.arguments.get("title") not in titles]
    repeated = len(events) - len({(e.name, json.dumps(e.arguments, sort_keys=True)) for e in events})
    problems: list[str] = []

    if case["kind"] == "exact":
        if not any(e.arguments.get("title") == case["title"] for e in searches):
            problems.append("never searched the given contract")
        if made_up:
            problems.append("searched a title that does not exist")
        if lists:
            problems.append("listed contracts although the exact title was given")
    elif case["kind"] == "partial":
        first = events[0] if events else None
        if first is None or first.name != "list_contracts":
            problems.append("did not start by listing contracts")
        elif case["title"].lower().find(str(first.arguments.get("contains", "")).lower()) < 0 or not first.arguments.get("contains"):
            problems.append("list_contracts was not given text from the real title")
        if not any(e.arguments.get("title") == case["title"] for e in searches):
            problems.append("never searched the right contract")
        if made_up:
            problems.append("searched a title that does not exist")
    else:  # missing
        if not lists:
            problems.append("did not look the contract up")
        if made_up:
            problems.append("searched a made-up title")
        if result.found:
            problems.append("claimed to find an answer in a contract that does not exist")

    if failed:
        problems.append(f"{len(failed)} tool call(s) failed")
    if repeated:
        problems.append(f"{repeated} repeated identical call(s)")
    return {"kind": case["kind"], "pass": not problems, "problems": problems, "calls": [e.name for e in events],
            "tool_calls": len(events), "failed": len(failed), "made_up": len(made_up), "cost_usd": result.cost_usd,
            "seconds": result.seconds, "found": result.found, "stop_reason": result.stop_reason,
            "question": case["question"]}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--per-kind", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    settings = Settings()
    service = GatewayService()
    titles = service.store.titles()
    llm = RouterLLM(settings, models=[args.model], with_fallbacks=False)
    cases = build_cases(args.per_kind, args.seed, titles)
    gate = asyncio.Semaphore(args.concurrency)

    async def one(case):
        async with gate:
            result = await run_agent(case["question"], llm=llm, service=service, settings=settings,
                                     contract=case["contract"])
        return grade(case, result, set(titles))

    rows = await asyncio.gather(*(one(c) for c in cases))
    broken = [r for r in rows if r["stop_reason"] in ("error", "timeout")]
    if broken:
        sys.exit(f"{len(broken)} of {len(rows)} questions hit a provider failure, so nothing was saved. Check the key and credit.")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"tools_{args.model.replace('/', '_')}.json"
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"{args.model}")
    for kind in ("exact", "partial", "missing"):
        group = [r for r in rows if r["kind"] == kind]
        passed = sum(r["pass"] for r in group)
        calls = sum(r["tool_calls"] for r in group) / len(group)
        print(f"  {kind:8} {passed}/{len(group)} pass   {calls:.1f} tool calls per question   "
              f"{sum(r['failed'] for r in group)} failed calls, {sum(r['made_up'] for r in group)} made-up titles")
    problems: dict[str, int] = {}
    for r in rows:
        for p in r["problems"]:
            key = p.split("(")[0].strip()
            problems[key] = problems.get(key, 0) + 1
    for p, n in sorted(problems.items(), key=lambda kv: -kv[1]):
        print(f"    {n:3}x {p}")
    print(f"  total ${sum(r['cost_usd'] for r in rows):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
