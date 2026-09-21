"""Compares fixed-window and structure-aware chunking on CUAD contracts.

Run from the repo root (needs data/raw/CUAD_v1.json, see README):
    PYTHONPATH=src .venv/Scripts/python eval/chunking_eval.py

Definitions, fixed before running:
- Each question is answered within its own contract, so retrieval ranks only
  that contract's chunks. (CUAD's clause questions are per-contract.)
- Retrieval is BM25 over lowercase word tokens. It is deliberately simple and
  free: it isolates the effect of chunking from any embedding model.
- Query = the clause category plus CUAD's own one-line description of it. The
  rest of the CUAD question text is identical boilerplate and is dropped.
- Only questions with at least one expert-marked answer span are scored.
- Hit@k: the union of the top-k chunks covers at least 50% of the characters of
  at least one gold answer span.
- Span integrity: fraction of gold spans that lie entirely inside a single
  chunk. Independent of retrieval; measures how often a chunker cuts a clause.
- Uncertainty: 95% bootstrap interval over CONTRACTS (not questions) for the
  paired differences, since questions within a contract are correlated.
- Equal-context comparison: structure-aware chunks are smaller on average, so
  equal top-k gives the fixed-window system more text. Hit at a word budget
  (500, 1000, 1250) retrieves ranked chunks until the budget is used. This was
  added AFTER seeing the equal-top-k result, to address that confound; both
  are reported. The primary budget (1250 words = top-5 of fixed windows) was
  declared before the budgeted run.

Both chunkers cap chunks at about 250 words.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from gateway.chunking import fixed_window_chunks, structure_aware_chunks

DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "CUAD_v1.json"
KS = (1, 3, 5)
# Equal-context comparison. 1250 words is what the top-5 of fixed windows returns
# (5 x 250), and is the primary budget, declared before the budgeted run.
BUDGETS = (500, 1000, 1250)
PRIMARY_BUDGET = 1250
QUESTION = re.compile(r'related to "(.*?)" that should be reviewed by a lawyer\. Details: (.*)$', re.S)
TOKEN = re.compile(r"\w+")
BOOTSTRAP_SAMPLES = 2000
SEED = 20260921


def tokenize(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def covered_fraction(chunks, top_idx, span) -> float:
    s, e = span
    pieces = sorted((max(s, chunks[i].start), min(e, chunks[i].end)) for i in top_idx if chunks[i].start < e and chunks[i].end > s)
    covered, cur_s, cur_e = 0, None, None
    for ps, pe in pieces:
        if cur_e is None or ps > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = ps, pe
        else:
            cur_e = max(cur_e, pe)
    if cur_e is not None:
        covered += cur_e - cur_s
    return covered / max(e - s, 1)


def inside_one_chunk(chunks, span) -> bool:
    return any(c.start <= span[0] and c.end >= span[1] for c in chunks)


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))["data"]
    chunkers = {"fixed_window": fixed_window_chunks, "structure_aware": structure_aware_chunks}

    per_contract = {name: [] for name in chunkers}  # list of (n_questions, hits_by_k dict)
    per_category = {name: defaultdict(lambda: defaultdict(list)) for name in chunkers}
    integrity = {name: [0, 0] for name in chunkers}
    chunk_stats = {name: {"chunks": [], "words": []} for name in chunkers}
    unparsed = 0

    for doc in data:
        para = doc["paragraphs"][0]
        text = para["context"]
        questions = []
        for qa in para["qas"]:
            if not qa["answers"]:
                continue
            m = QUESTION.search(qa["question"])
            if m:
                category, query = m.group(1), f"{m.group(1)} {m.group(2)}"
            else:
                unparsed += 1
                category, query = "unparsed", qa["question"]
            spans = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]]
            questions.append((category, tokenize(query), spans))
        if not questions:
            continue

        for name, chunk_fn in chunkers.items():
            chunks = chunk_fn(text)
            chunk_stats[name]["chunks"].append(len(chunks))
            chunk_stats[name]["words"].extend(len(c.text.split()) for c in chunks)
            bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])

            words = [len(c.text.split()) for c in chunks]
            hits = {k: 0 for k in KS}
            hits.update({("w", b): 0 for b in BUDGETS})
            for category, q_tokens, spans in questions:
                scores = bm25.get_scores(q_tokens)
                order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
                for k in KS:
                    top = order[:k]
                    hit = any(covered_fraction(chunks, top, sp) >= 0.5 for sp in spans)
                    hits[k] += hit
                    per_category[name][category][k].append(hit)
                for b in BUDGETS:
                    top, used = [], 0
                    for i in order:
                        if used + words[i] > b:
                            break
                        top.append(i)
                        used += words[i]
                    hit = any(covered_fraction(chunks, top, sp) >= 0.5 for sp in spans)
                    hits[("w", b)] += hit
                for sp in spans:
                    integrity[name][1] += 1
                    integrity[name][0] += inside_one_chunk(chunks, sp)
            per_contract[name].append((len(questions), hits))

    def overall(name: str, k) -> float:
        n = sum(q for q, _ in per_contract[name])
        return sum(h[k] for _, h in per_contract[name]) / n

    # paired bootstrap over contracts on Hit@5 difference
    rng = np.random.default_rng(SEED)
    nq = np.array([q for q, _ in per_contract["fixed_window"]])

    def paired_ci(key):
        hf = np.array([h[key] for _, h in per_contract["fixed_window"]])
        hs = np.array([h[key] for _, h in per_contract["structure_aware"]])
        diffs = []
        for _ in range(BOOTSTRAP_SAMPLES):
            idx = rng.integers(0, len(nq), len(nq))
            diffs.append(hs[idx].sum() / nq[idx].sum() - hf[idx].sum() / nq[idx].sum())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    results = {
        "n_contracts": len(per_contract["fixed_window"]),
        "n_scored_questions": int(nq.sum()),
        "unparsed_questions": unparsed,
        "hit_at_k": {name: {f"@{k}": round(overall(name, k), 4) for k in KS} for name in chunkers},
        "hit_at_5_difference_structure_minus_fixed": {
            "point": round(overall("structure_aware", 5) - overall("fixed_window", 5), 4),
            "ci95": paired_ci(5),
        },
        "hit_at_word_budget": {name: {f"{b}w": round(overall(name, ("w", b)), 4) for b in BUDGETS} for name in chunkers},
        "primary_budget_difference_structure_minus_fixed": {
            "budget_words": PRIMARY_BUDGET,
            "point": round(overall("structure_aware", ("w", PRIMARY_BUDGET)) - overall("fixed_window", ("w", PRIMARY_BUDGET)), 4),
            "ci95": paired_ci(("w", PRIMARY_BUDGET)),
        },
        "span_integrity": {n: round(a / b, 4) for n, (a, b) in integrity.items()},
        "chunk_stats": {
            n: {
                "median_chunks_per_contract": statistics.median(v["chunks"]),
                "median_words_per_chunk": statistics.median(v["words"]),
                "mean_words_per_chunk": round(statistics.mean(v["words"]), 1),
            }
            for n, v in chunk_stats.items()
        },
        "per_category_hit_at_5": {
            cat: {
                name: round(sum(per_category[name][cat][5]) / len(per_category[name][cat][5]), 3)
                for name in chunkers
            }
            for cat in sorted(per_category["fixed_window"])
        },
    }
    out = Path(__file__).parent / "results" / "chunking_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {k: v for k, v in results.items() if k != "per_category_hit_at_5"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
