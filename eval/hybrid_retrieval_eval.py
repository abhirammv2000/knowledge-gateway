"""BM25 only, dense only, hybrid (RRF) and hybrid plus rerank on CUAD, on structure-aware chunks.

Run from the repo root (needs data/raw/CUAD_v1.json, see README):
    PYTHONPATH=src .venv/Scripts/python eval/hybrid_retrieval_eval.py

The question parsing, span scoring and bootstrap come from chunking_eval.py, so this only adds the retrieval
methods. Structure-aware chunking is the base because it tied fixed windows on retrieval at equal context while
cutting fewer clauses. Models are all-MiniLM-L6-v2 and cross-encoder/ms-marco-MiniLM-L-6-v2, both open and run
on cpu.

Only the top RERANK_POOL hybrid results are reranked and the rest keep their order. The pool is 10 because the
cross-encoder took about 16 to 18 ms per (query, chunk) pair on this machine whether batched or not, so a pool of 15
over all 6,702 questions would take about 2.2 hours.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.chunking_eval import QUESTION, covered_fraction
from gateway.chunking import structure_aware_chunks
from gateway.retrieval import build_index, dense_search, get_embedder, get_reranker, rerank, reciprocal_rank_fusion, sparse_search

DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "CUAD_v1.json"
KS = (1, 3, 5)
RERANK_POOL = 10
BOOTSTRAP_SAMPLES = 2000
SEED = 20260921
METHODS = ("bm25_only", "dense_only", "hybrid", "hybrid_rerank")


def score_question(chunks, bm25_order, dense_order, hybrid_order, reranked_order, spans) -> dict:
    orders = {"bm25_only": bm25_order, "dense_only": dense_order, "hybrid": hybrid_order, "hybrid_rerank": reranked_order}
    return {
        method: {k: any(covered_fraction(chunks, order[:k], sp) >= 0.5 for sp in spans) for k in KS}
        for method, order in orders.items()
    }


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))["data"]
    embedder = get_embedder()
    reranker = get_reranker()

    per_contract = {m: [] for m in METHODS}  # list of (n_questions, {k: hit_count})
    unparsed = 0
    n_contracts_used = 0

    for doc_i, doc in enumerate(data):
        para = doc["paragraphs"][0]
        text = para["context"]
        questions = []
        for qa in para["qas"]:
            if not qa["answers"]:
                continue
            m = QUESTION.search(qa["question"])
            query = f"{m.group(1)} {m.group(2)}" if m else qa["question"]
            if not m:
                unparsed += 1
            spans = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]]
            questions.append((query, spans))
        if not questions:
            continue
        n_contracts_used += 1

        chunks = structure_aware_chunks(text)
        chunk_texts = [c.text for c in chunks]
        index = build_index(chunk_texts, embedder=embedder)

        hits = {m: {k: 0 for k in KS} for m in METHODS}
        for query, spans in questions:
            bm25_order = sparse_search(index, query)
            dense_order = dense_search(index, query, embedder=embedder)
            hybrid_order = reciprocal_rank_fusion([bm25_order, dense_order])
            reranked_pool = rerank(index, query, hybrid_order[:RERANK_POOL], reranker=reranker)
            reranked_order = reranked_pool + hybrid_order[RERANK_POOL:]

            per_q = score_question(chunks, bm25_order, dense_order, hybrid_order, reranked_order, spans)
            for method in METHODS:
                for k in KS:
                    hits[method][k] += per_q[method][k]

        for method in METHODS:
            per_contract[method].append((len(questions), hits[method]))

        if (doc_i + 1) % 50 == 0:
            print(f"...{doc_i + 1}/{len(data)} contracts", flush=True)

    def overall(method: str, k: int) -> float:
        n = sum(q for q, _ in per_contract[method])
        return sum(h[k] for _, h in per_contract[method]) / n

    rng = np.random.default_rng(SEED)
    nq = np.array([q for q, _ in per_contract["bm25_only"]])

    def paired_ci(method_a: str, method_b: str, k: int) -> list[float]:
        ha = np.array([h[k] for _, h in per_contract[method_a]])
        hb = np.array([h[k] for _, h in per_contract[method_b]])
        diffs = []
        for _ in range(BOOTSTRAP_SAMPLES):
            idx = rng.integers(0, len(nq), len(nq))
            diffs.append(ha[idx].sum() / nq[idx].sum() - hb[idx].sum() / nq[idx].sum())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    results = {
        "n_contracts": n_contracts_used,
        "n_scored_questions": int(nq.sum()),
        "unparsed_questions": unparsed,
        "rerank_pool": RERANK_POOL,
        "hit_at_k": {m: {f"@{k}": round(overall(m, k), 4) for k in KS} for m in METHODS},
        "hit_at_5_differences": {
            "dense_minus_bm25": {"point": round(overall("dense_only", 5) - overall("bm25_only", 5), 4), "ci95": paired_ci("dense_only", "bm25_only", 5)},
            "hybrid_minus_bm25": {"point": round(overall("hybrid", 5) - overall("bm25_only", 5), 4), "ci95": paired_ci("hybrid", "bm25_only", 5)},
            "hybrid_minus_dense": {"point": round(overall("hybrid", 5) - overall("dense_only", 5), 4), "ci95": paired_ci("hybrid", "dense_only", 5)},
            "rerank_minus_hybrid": {"point": round(overall("hybrid_rerank", 5) - overall("hybrid", 5), 4), "ci95": paired_ci("hybrid_rerank", "hybrid", 5)},
        },
        "hit_at_1_differences": {
            "rerank_minus_hybrid": {"point": round(overall("hybrid_rerank", 1) - overall("hybrid", 1), 4), "ci95": paired_ci("hybrid_rerank", "hybrid", 1)},
        },
    }
    out = Path(__file__).parent / "results" / "hybrid_retrieval_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
