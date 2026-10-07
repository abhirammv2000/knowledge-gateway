"""When does a semantic cache give the wrong answer? Measured with the real embedding model, no LLM calls.

Three groups of question pairs:
  paraphrase   the same question in different words. These should hit.
  different    questions about different clauses (every cross-topic pair). These must not hit.
  flipped      questions that read almost the same but ask something else: a negation, a different
               number, the other party. These must not hit, and they are the hard case.

For each threshold the script prints how many of each group would be served the stored answer, with
and without the guard that refuses a match when the numbers, negations or parties differ.

    PYTHONPATH=src python eval/cache_eval.py                       # the local MiniLM model
    PYTHONPATH=src python eval/cache_eval.py --embedder openai     # text-embedding-3-small
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.cache import meaning_markers  # noqa: E402
from gateway.retrieval import get_embedder  # noqa: E402

TOPICS = {
    "termination notice": [
        "How much notice is required to terminate this agreement?",
        "How many days of notice does either party have to give to end the contract?",
        "What is the notice period for terminating the agreement?",
    ],
    "governing law": [
        "Which state's law governs this agreement?",
        "What law applies to this contract?",
        "Under what jurisdiction's law is this agreement interpreted?",
    ],
    "assignment": [
        "Can either party assign this agreement to someone else?",
        "Is assignment of the contract to a third party allowed?",
        "Can the agreement be transferred to another company?",
    ],
    "exclusivity": [
        "Does the agreement give anyone exclusive rights?",
        "Is the distributor the exclusive seller in the territory?",
        "Are the rights granted on an exclusive basis?",
    ],
    "liability cap": [
        "Is there a cap on liability?",
        "What is the maximum amount either party can be held liable for?",
        "Does the contract limit damages to a certain amount?",
    ],
    "non-compete": [
        "Is either party restricted from competing with the other?",
        "Does the agreement contain a non-compete clause?",
        "Are there any limits on competing businesses?",
    ],
    "renewal": [
        "Does the contract renew automatically?",
        "What happens when the initial term ends, does it renew?",
        "Is there an automatic renewal of the agreement?",
    ],
    "insurance": [
        "What insurance must each party carry?",
        "Does the agreement require the parties to maintain insurance?",
        "Are there insurance requirements?",
    ],
    "ip ownership": [
        "Who owns the intellectual property created under this agreement?",
        "Who has ownership of IP developed during the contract?",
        "To whom do the IP rights belong?",
    ],
    "payment": [
        "What are the payment terms?",
        "When are invoices due and how are fees paid?",
        "How much must be paid and by when?",
    ],
}

FLIPPED = [
    ("Can either party assign this agreement without consent?", "Is assignment of this agreement prohibited without consent?"),
    ("Does the agreement allow termination for convenience?", "Does the agreement prohibit termination for convenience?"),
    ("Is the license exclusive?", "Is the license non-exclusive?"),
    ("Does the contract renew automatically?", "Does the contract expire without renewal?"),
    ("Is there a cap on liability?", "Is liability unlimited?"),
    ("Can the supplier terminate for breach?", "Can the customer terminate for breach?"),
    ("Is a 30 day notice required to terminate?", "Is a 90 day notice required to terminate?"),
    ("Is the non-compete limited to one year?", "Is the non-compete limited to five years?"),
    ("Does the licensee have to pay the fees?", "Does the licensor have to pay the fees?"),
    ("Does the buyer have to pay within 30 days?", "Does the seller have to pay within 30 days?"),
    ("Is the confidentiality duty mutual?", "Is the confidentiality duty not mutual?"),
    ("Are there any penalties for late payment?", "Are there no penalties for late payment?"),
    ("Can the tenant sublease the premises?", "Can the landlord sublease the premises?"),
    ("Does the warranty last two years?", "Does the warranty last ten years?"),
    ("Is arbitration mandatory?", "Is arbitration prohibited?"),
]

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96]


def pairs() -> dict[str, list[tuple[str, str]]]:
    paraphrase = [p for qs in TOPICS.values() for p in itertools.combinations(qs, 2)]
    different = [(a, b) for (t1, q1), (t2, q2) in itertools.combinations(TOPICS.items(), 2)
                 for a in q1 for b in q2]
    return {"paraphrase": paraphrase, "different": different, "flipped": FLIPPED}


def embed_all(texts: list[str], which: str) -> dict[str, np.ndarray]:
    if which == "openai":
        import os

        import litellm

        data = litellm.embedding(model="text-embedding-3-small", input=texts).data
        vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return dict(zip(texts, vecs))
    return dict(zip(texts, get_embedder().encode(texts, normalize_embeddings=True)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["minilm", "openai"], default="minilm")
    args = parser.parse_args()
    groups = pairs()
    texts = sorted({t for ps in groups.values() for p in ps for t in p})
    vectors = embed_all(texts, args.embedder)

    sims = {g: np.array([float(np.dot(vectors[a], vectors[b])) for a, b in ps]) for g, ps in groups.items()}
    same_markers = {g: np.array([meaning_markers(a) == meaning_markers(b) for a, b in ps]) for g, ps in groups.items()}

    print("similarity of each group (cosine):")
    for g, s in sims.items():
        print(f"  {g:11} n={len(s):4d}  min {s.min():.3f}  median {np.median(s):.3f}  max {s.max():.3f}")
    print(f"\nthe guard lets {same_markers['paraphrase'].mean():.0%} of paraphrases through "
          f"and {same_markers['flipped'].mean():.0%} of flipped pairs, {same_markers['different'].mean():.0%} of different-clause pairs\n")

    print(f"{'threshold':>9} | {'paraphrase hit':>14} {'different hit':>14} {'flipped hit':>12} | with the guard: "
          f"{'paraphrase':>10} {'different':>10} {'flipped':>8}")
    table = []
    for t in THRESHOLDS:
        row = {"threshold": t}
        for guard in (False, True):
            for g in groups:
                hit = sims[g] >= t
                if guard:
                    hit = hit & same_markers[g]
                row[f"{g}{'_guard' if guard else ''}"] = float(hit.mean())
        table.append(row)
        print(f"{t:9.2f} | {row['paraphrase']:14.1%} {row['different']:14.1%} {row['flipped']:12.1%} | "
              f"{'':15}{row['paraphrase_guard']:10.1%} {row['different_guard']:10.1%} {row['flipped_guard']:8.1%}")

    unresolved = [(a, b, round(float(sims['flipped'][i]), 3)) for i, (a, b) in enumerate(FLIPPED) if same_markers['flipped'][i]]
    print(f"\nflipped pairs the guard cannot tell apart ({len(unresolved)}):")
    for a, b, s in unresolved:
        print(f"  {s:.3f}  {a}  |  {b}")

    out = ROOT / "eval" / "results" / f"cache_eval_{args.embedder}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": "all-MiniLM-L6-v2" if args.embedder == "minilm" else "text-embedding-3-small", "table": table,
        "similarity": {g: {"min": float(s.min()), "median": float(np.median(s)), "max": float(s.max()), "n": len(s)}
                       for g, s in sims.items()},
        "guard_passes": {g: float(m.mean()) for g, m in same_markers.items()},
        "unresolved_flips": [{"a": a, "b": b, "similarity": s} for a, b, s in unresolved],
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
