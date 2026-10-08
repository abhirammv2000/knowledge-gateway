"""Does the agent find the right clause, and does it say so when there isn't one?

Questions come from CUAD, where lawyers marked the clauses that match each of 41 categories in 510
contracts. For each sampled (contract, category) pair the agent gets CUAD's own question. It is
graded against the lawyers' marks:

- answerable: right if the agent says it found something, its citations all verified, and a cited
  passage covers at least half of a marked span (the same rule chunking_eval.py used)
- not answerable: right if the agent says nothing was found

Results are appended to a jsonl file one question at a time, so a run that stops can be resumed
without paying for the questions it already finished.

    PYTHONPATH=src python eval/agent_eval.py run --model anthropic/claude-sonnet-5-5 --n 100
    PYTHONPATH=src python eval/agent_eval.py report eval/results/agent_*.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.config import Settings  # noqa: E402
from agent.llm import RouterLLM  # noqa: E402
from agent.loop import run_agent  # noqa: E402
from gateway.chunking import structure_aware_chunks  # noqa: E402
from gateway.service import GatewayService  # noqa: E402

CUAD = ROOT / "data" / "raw" / "CUAD_v1.json"
RESULTS = ROOT / "eval" / "results"
COVER = 0.5  # a cited passage must cover this much of a marked span


def load_cuad() -> dict[str, dict]:
    docs = json.loads(CUAD.read_text(encoding="utf-8"))["data"]
    return {d["title"]: d["paragraphs"][0] for d in docs}


def build_questions(n: int, seed: int, answerable_share: float = 0.5) -> list[dict]:
    """A fixed, reproducible sample: about half answerable, spread over a few contracts."""
    cuad = load_cuad()
    rng = random.Random(seed)
    titles = sorted(cuad)
    rng.shuffle(titles)
    want_answerable = round(n * answerable_share)
    per_contract = 5
    questions: list[dict] = []
    answerable = impossible = 0
    for title in titles:
        qas = cuad[title]["qas"]
        yes = [q for q in qas if not q["is_impossible"]]
        no = [q for q in qas if q["is_impossible"]]
        rng.shuffle(yes)
        rng.shuffle(no)
        take_yes = min(3, len(yes), want_answerable - answerable)
        take_no = min(per_contract - take_yes, len(no), (n - want_answerable) - impossible)
        for q in yes[:take_yes] + no[:take_no]:
            questions.append({
                "id": q["id"], "contract": title, "category": q["id"].split("__")[-1], "question": q["question"],
                "answerable": not q["is_impossible"],
                "gold": [[a["answer_start"], a["answer_start"] + len(a["text"])] for a in q["answers"]],
            })
        answerable += take_yes
        impossible += take_no
        if len(questions) >= n:
            break
    return questions[:n]


def covered(span: list[int], chunk_start: int, chunk_end: int) -> bool:
    overlap = min(span[1], chunk_end) - max(span[0], chunk_start)
    return span[1] > span[0] and overlap / (span[1] - span[0]) >= COVER


def grade(question: dict, result, chunks_by_contract: dict) -> dict:
    """Score one result against the lawyers' marks."""
    cited_chunks = sorted({c.chunk_index for c in result.citations if c.contract == question["contract"]})
    hit = False
    if question["answerable"] and result.found and result.verified:
        chunks = chunks_by_contract[question["contract"]]
        hit = any(covered(span, chunks[i].start, chunks[i].end) for i in cited_chunks for span in question["gold"])
    correct = hit if question["answerable"] else (not result.found)
    return {
        "id": question["id"], "category": question["category"], "answerable": question["answerable"],
        "correct": correct, "found": result.found, "verified": result.verified, "hit": hit,
        "stop_reason": result.stop_reason, "confidence": result.confidence, "citations": len(result.citations),
        "iterations": result.iterations, "tool_calls": result.tool_call_count,
        "tools": [e.name for e in result.tool_events], "tool_errors": sum(1 for e in result.tool_events if not e.ok),
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "cost_usd": result.cost_usd,
        "seconds": result.seconds, "models": result.models_used, "fallback": result.fallback_used,
        "answer": result.answer[:600],
    }


async def run(args: argparse.Namespace) -> None:
    RESULTS.mkdir(exist_ok=True)
    tag = args.model.replace("/", "_")
    out_path = RESULTS / f"agent_{tag}_n{args.n}_s{args.seed}.jsonl"
    done = {json.loads(line)["id"] for line in out_path.read_text().splitlines()} if out_path.exists() else set()
    questions = [q for q in build_questions(args.n, args.seed) if q["id"] not in done]
    print(f"{args.model}: {len(done)} done, {len(questions)} to run -> {out_path.name}")

    settings = Settings()
    service = GatewayService()
    llm = RouterLLM(settings, models=[args.model], with_fallbacks=False)
    cuad = load_cuad()
    chunks_by_contract = {t: structure_aware_chunks(p["context"]) for t, p in cuad.items()}
    gate = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    failures = {"in_a_row": 0, "total": 0}

    async def one(q: dict) -> None:
        if failures["in_a_row"] >= 5:
            return  # the provider is down or out of credit, so stop spending and stop writing
        async with gate:
            result = await run_agent(q["question"], llm=llm, service=service, settings=settings, contract=q["contract"])
        if result.stop_reason in ("error", "timeout"):
            # a provider failure says nothing about the model, so it is never saved as a result
            failures["in_a_row"] += 1
            failures["total"] += 1
            print(f"  provider failure ({result.stop_reason}), not saved: {q['category']}")
            return
        failures["in_a_row"] = 0
        row = grade(q, result, chunks_by_contract)
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {'ok ' if row['correct'] else 'MISS'} {q['category'][:34]:34} found={row['found']!s:5} "
                  f"${row['cost_usd']:.3f} {row['seconds']:5.1f}s {row['stop_reason']}")

    started = time.time()
    await asyncio.gather(*(one(q) for q in questions))
    print(f"finished in {time.time() - started:.0f}s, {failures['total']} provider failures not saved")
    if failures["in_a_row"] >= 5:
        sys.exit("stopped early: 5 provider failures in a row. Check the API key, credit and model name, then run again to resume.")


# report

def bootstrap_ci(values: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = rng.choice(arr, size=(reps, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    imp = [r for r in rows if not r["answerable"]]
    said_found = [r for r in rows if r["found"] and r["verified"]]
    summary = {
        "n": len(rows), "answerable": len(ans), "impossible": len(imp),
        "accuracy": float(np.mean([r["correct"] for r in rows])),
        "recall": float(np.mean([r["hit"] for r in ans])) if ans else float("nan"),
        "refusal": float(np.mean([not r["found"] for r in imp])) if imp else float("nan"),
        "precision": float(np.mean([r["hit"] for r in said_found])) if said_found else float("nan"),
        "verified": float(np.mean([r["verified"] for r in rows])),
        "cost_per_q": float(np.mean([r["cost_usd"] for r in rows])),
        "seconds_p50": float(np.percentile([r["seconds"] for r in rows], 50)),
        "seconds_p95": float(np.percentile([r["seconds"] for r in rows], 95)),
        "tool_calls": float(np.mean([r["tool_calls"] for r in rows])),
        "tool_error_rate": float(sum(r["tool_errors"] for r in rows) / max(1, sum(r["tool_calls"] for r in rows))),
        "fallback": float(np.mean([r["fallback"] for r in rows])),
        "stops": {s: sum(1 for r in rows if r["stop_reason"] == s) for s in sorted({r["stop_reason"] for r in rows})},
    }
    summary["accuracy_ci"] = bootstrap_ci([float(r["correct"]) for r in rows])
    return summary


def report(args: argparse.Namespace) -> None:
    import glob

    paths = [p for pattern in args.files for p in glob.glob(pattern)]
    table: dict[str, dict] = {}
    by_id: dict[str, dict[str, bool]] = {}
    for path in sorted(paths):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not rows:
            continue
        name = Path(path).stem.removeprefix("agent_")
        table[name] = summarize(rows)
        by_id[name] = {r["id"]: r["correct"] for r in rows}

    print(f"{'run':52} {'n':>4} {'acc':>6} {'95% CI':>13} {'recall':>7} {'refuse':>7} {'prec':>6} {'$/q':>7} {'p50':>6} {'p95':>6} {'tools':>5}")
    for name, s in table.items():
        lo, hi = s["accuracy_ci"]
        print(f"{name:52} {s['n']:4d} {s['accuracy']:6.1%} [{lo:5.1%},{hi:5.1%}] {s['recall']:7.1%} {s['refusal']:7.1%} "
              f"{s['precision']:6.1%} {s['cost_per_q']:7.4f} {s['seconds_p50']:6.1f} {s['seconds_p95']:6.1f} {s['tool_calls']:5.1f}")
        print(f"{'':52} stops={s['stops']} tool_error_rate={s['tool_error_rate']:.1%} fallback={s['fallback']:.1%}")

    names = list(by_id)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(set(by_id[a]) & set(by_id[b]))
            if len(shared) < 10:
                continue
            diffs = [float(by_id[a][q]) - float(by_id[b][q]) for q in shared]
            lo, hi = bootstrap_ci(diffs)
            print(f"\npaired difference on {len(shared)} shared questions, {a} minus {b}: "
                  f"{np.mean(diffs):+.1%} [{lo:+.1%}, {hi:+.1%}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--model", required=True)
    run_p.add_argument("--n", type=int, default=100)
    run_p.add_argument("--seed", type=int, default=7)
    run_p.add_argument("--concurrency", type=int, default=3)
    rep = sub.add_parser("report")
    rep.add_argument("files", nargs="+")
    args = parser.parse_args()
    if args.command == "run":
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
            if not os.environ.get(key):
                print(f"note: {key} is not set", file=sys.stderr)
        asyncio.run(run(args))
    else:
        report(args)


if __name__ == "__main__":
    main()
