"""Would a cheap-first cascade beat using the strong model alone? Worked out from two saved runs.

Run the cheap model first. If its answer looks unsure, run the strong model too and use that answer.
Both models were run on the same 100 questions at temperature 0, so each question's result with each
model is already known. This replays the cascade from those rows, so it assumes a rerun would give the
same answers. It costs nothing and calls no API.

    python eval/routing_sim.py
"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
STRONG = RESULTS / "agent_openai_gpt-4o_n100_s7.jsonl"
CHEAP = RESULTS / "agent_openai_gpt-4o-mini_n100_s7.jsonl"


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def replay(strong: dict, cheap: dict, escalate) -> dict:
    ids = list(strong)
    right = cost = went = 0
    seconds = []
    for i in ids:
        s, c = strong[i], cheap[i]
        go = escalate(c)
        final = s if go else c
        right += final["correct"]
        cost += c["cost_usd"] + (s["cost_usd"] if go else 0)
        seconds.append(c["seconds"] + (s["seconds"] if go else 0))
        went += go
    seconds.sort()
    n = len(ids)
    return {"accuracy": right / n, "cost_per_question": cost / n, "escalated": went / n, "median_seconds": seconds[n // 2]}


def main() -> None:
    strong, cheap = load(STRONG), load(CHEAP)
    table = {
        "cheap model only": replay(strong, cheap, lambda c: False),
        "strong model only": {
            "accuracy": sum(r["correct"] for r in strong.values()) / len(strong),
            "cost_per_question": sum(r["cost_usd"] for r in strong.values()) / len(strong),
            "escalated": 1.0,
            "median_seconds": sorted(r["seconds"] for r in strong.values())[len(strong) // 2],
        },
        "cascade: escalate unless found and verified": replay(strong, cheap, lambda c: not (c["found"] and c["verified"])),
        "cascade: escalate only on errors or unverified": replay(
            strong, cheap, lambda c: c["stop_reason"] != "answered" or (c["found"] and not c["verified"])),
    }
    for name, row in table.items():
        print(f"{name:48} accuracy {row['accuracy']:.0%}  ${row['cost_per_question']:.4f}/q  "
              f"escalated {row['escalated']:.0%}  median {row['median_seconds']:.1f}s")
    (RESULTS / "routing_cascade_sim.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
