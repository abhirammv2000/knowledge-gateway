"""Where does the agent do worse? Accuracy by slice, from a saved run. No model is called.

One overall accuracy hides who is being served badly. This splits a run by the things that could matter and says
which slices are worse than the rest with more than noise:

  clause group   header facts (name, parties, dates) against clause terms (liability, assignment, termination)
  clause present whether the question has a marked clause at all
  contract length  short, medium and long contracts (thirds by word count)
  contract type    the kind of agreement named at the end of the CUAD title, where three or more questions share it

A slice is flagged only when its Wilson interval lies entirely below the interval of everything else, widened the more
slices a dimension has, so a slice of five questions is not flagged for being unlucky. CUAD contracts carry no information about the people involved,
so this is performance slicing, not a fairness audit across demographic groups. That would need data this set lacks.

    python eval/slice_report.py eval/results/agent_openai_gpt-4o_n100_s7.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
CUAD = ROOT / "data" / "raw" / "CUAD_v1.json"
HEADER_FACTS = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term"}
MIN_SLICE = 3


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def contract_of(row: dict) -> str:
    return row["id"].rsplit("__", 1)[0]


def contract_type(title: str) -> str:
    """The kind of agreement: the last underscore-separated part of a CUAD title, without the year and exhibit number
    in front of it (2012-EX-10.6-TRANSPORTATION CONTRACT becomes transportation contract) or a trailing digit."""
    last = title.rsplit("_", 1)[-1]
    last = re.sub(r"^.*?\bex-?[\w.()]+[-_ ]", "", last, flags=re.I)
    return re.sub(r"\d+$", "", last).strip().lower() or "unknown"


def length_bands(words_by_contract: dict[str, int]) -> dict[str, str]:
    """short, medium or long for each contract, by thirds of the contracts in the run."""
    ordered = sorted(words_by_contract, key=lambda t: words_by_contract[t])
    third = max(1, math.ceil(len(ordered) / 3))
    return {t: ("short", "medium", "long")[min(2, i // third)] for i, t in enumerate(ordered)}


def slices(rows: list[dict], words_by_contract: dict[str, int] | None = None) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {"clause group": {}, "clause present": {}, "contract length": {}, "contract type": {}}
    bands = length_bands(words_by_contract) if words_by_contract else {}
    for row in rows:
        title = contract_of(row)
        out["clause group"].setdefault("header facts" if row["category"] in HEADER_FACTS else "clause terms", []).append(row)
        out["clause present"].setdefault("clause marked" if row["answerable"] else "no clause marked", []).append(row)
        if title in bands:
            out["contract length"].setdefault(bands[title], []).append(row)
        out["contract type"].setdefault(contract_type(title), []).append(row)
    out["contract type"] = {k: v for k, v in out["contract type"].items() if len(v) >= MIN_SLICE}
    return out


def report(rows: list[dict], words_by_contract: dict[str, int] | None = None) -> list[dict]:
    """One entry per slice: its size, accuracy, interval, and whether it is clearly worse than the rest.

    Comparing many slices makes some look bad by luck, so the flag uses a stricter interval the more slices a
    dimension has (a Bonferroni correction: the 5% is shared out among them). The interval printed is the usual 95%."""
    entries = []
    for dimension, groups in slices(rows, words_by_contract).items():
        z = NormalDist().inv_cdf(1 - 0.025 / max(1, len(groups)))
        for name, group in sorted(groups.items()):
            wins = sum(r["correct"] for r in group)
            rest = [r for r in rows if r not in group]
            rest_wins = sum(r["correct"] for r in rest)
            low, high = wilson(wins, len(group))
            strict_high = wilson(wins, len(group), z)[1]
            rest_low = wilson(rest_wins, len(rest), z)[0] if rest else 1.0
            entries.append({
                "dimension": dimension, "slice": name, "n": len(group), "accuracy": wins / len(group),
                "low": low, "high": high, "rest_accuracy": rest_wins / len(rest) if rest else float("nan"),
                "clearly_worse": len(group) >= MIN_SLICE and bool(rest) and strict_high < rest_low,
            })
    return entries


def words_by_contract(titles: set[str]) -> dict[str, int]:
    if not CUAD.exists():
        return {}
    docs = json.loads(CUAD.read_text(encoding="utf-8"))["data"]
    return {d["title"]: len(d["paragraphs"][0]["context"].split()) for d in docs if d["title"] in titles}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        sys.exit("no rows in that file")
    entries = report(rows, words_by_contract({contract_of(r) for r in rows}))
    overall = sum(r["correct"] for r in rows) / len(rows)
    print(f"{Path(args.file).name}: {len(rows)} questions, overall accuracy {overall:.0%}\n")
    for e in entries:
        flag = "  <- clearly worse than the rest" if e["clearly_worse"] else ""
        print(f"  {e['dimension']:16} {e['slice']:28} n={e['n']:3d}  {e['accuracy']:4.0%} [{e['low']:.0%}, {e['high']:.0%}]{flag}")
    out = ROOT / "eval" / "results" / (Path(args.file).stem.replace("agent_", "slices_") + ".json")
    out.write_text(json.dumps(entries, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
