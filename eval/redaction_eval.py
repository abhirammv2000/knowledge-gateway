"""Measures redaction precision and recall on synthetic, labeled PII.

Run from the repo root:
    PYTHONPATH=src .venv/Scripts/python eval/redaction_eval.py

Method, stated so the numbers can be judged:
- Documents are built from contract-style sentence templates with PII inserted
  by Faker (fixed seed, so the run is reproducible). Gold spans are recorded at
  insertion time, so labels are exact by construction.
- Some sentences contain no PII on purpose (legal boilerplate with company
  names and dates) to expose false positives.
- Recall: a gold span counts as found if a predicted span of the SAME type
  overlaps it. "Any-type recall" ignores the label, since for safety what
  matters most is that the value was removed at all.
- Precision: a predicted span counts as correct if it overlaps a gold span of
  the same type.
- This is synthetic text. It says how the detector behaves on clean, templated
  PII. It does NOT predict accuracy on messy real documents.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from faker import Faker

from gateway.redaction import build_analyzer, redact

SEED = 20260921
N_DOCS = 300
REAL_AREA_CODES = ["303", "720", "970", "212", "415", "206", "312", "617", "404", "512", "602", "305", "214", "713", "619", "202", "503", "615", "702", "813"]

PII_TEMPLATES = [
    "This Agreement is made between Acme Holdings and {PERSON}, reachable at {EMAIL_ADDRESS}.",
    "Notices shall be sent to {PERSON} by email at {EMAIL_ADDRESS} or by phone at {PHONE_NUMBER}.",
    "The Employee, {PERSON}, Social Security Number {US_SSN}, shall report to the Director.",
    "Payment shall be made by card number {CREDIT_CARD} on the first business day of each month.",
    "Wire funds to account {IBAN_CODE}, held in the name of {PERSON}.",
    "Access is restricted to the server at {IP_ADDRESS} and logged for audit.",
    "The guarantor {PERSON} may be contacted on {PHONE_NUMBER} regarding any default.",
]

NEGATIVE_SENTENCES = [
    "The Licensee shall pay all fees within thirty (30) days of receipt of invoice.",
    "This Agreement shall be governed by the laws of the State of Delaware.",
    "Either party may terminate this Agreement upon ninety (90) days written notice.",
    "Acme Holdings and Northwind Traders agree to the terms set out in Schedule A.",
    "The effective date of this Agreement is January 15, 2024.",
    "All intellectual property rights remain with the Licensor.",
]


def make_value(fake: Faker, entity: str) -> str:
    if entity == "PERSON":
        return fake.name()
    if entity == "EMAIL_ADDRESS":
        return fake.email()
    if entity == "PHONE_NUMBER":
        # Real US area codes only. An earlier version used random three-digit
        # codes, most of which do not exist; Presidio validates phone numbers,
        # so that measured the test data, not the detector.
        area = random.choice(REAL_AREA_CODES)
        number = f"{random.randint(2, 9)}{random.randint(0, 9)}{random.randint(0, 9)}-{random.randint(0, 9999):04d}"
        return random.choice([f"{area}-{number}", f"({area}) {number}"])
    if entity == "US_SSN":
        return fake.ssn()
    if entity == "CREDIT_CARD":
        return fake.credit_card_number()
    if entity == "IBAN_CODE":
        return fake.iban()
    if entity == "IP_ADDRESS":
        return fake.ipv4()
    raise ValueError(entity)


def negative_digits_sentence(rng: random.Random) -> str:
    """Non-PII long digit runs (invoice and reference numbers). About 1 in 10
    random digit strings passes Luhn, so these expose the false-positive cost
    of a checksum-based card recognizer."""
    n = rng.randint(12, 16)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(n))
    return rng.choice([
        f"Invoice reference number {digits} is due upon receipt.",
        f"Purchase order {digits} was issued under Schedule B.",
    ])


def build_doc(fake: Faker, rng: random.Random) -> tuple[str, list[tuple[int, int, str]]]:
    sentences = []
    for _ in range(rng.randint(3, 6)):
        roll = rng.random()
        if roll < 0.30:
            sentences.append(("neg", rng.choice(NEGATIVE_SENTENCES)))
        elif roll < 0.45:
            sentences.append(("neg", negative_digits_sentence(rng)))
        else:
            sentences.append(("pii", rng.choice(PII_TEMPLATES)))

    text = ""
    gold: list[tuple[int, int, str]] = []
    for kind, sentence in sentences:
        if text:
            text += " "
        if kind == "neg":
            text += sentence
            continue
        cursor = 0
        while True:
            start = sentence.find("{", cursor)
            if start == -1:
                text += sentence[cursor:]
                break
            end = sentence.index("}", start)
            text += sentence[cursor:start]
            entity = sentence[start + 1 : end]
            value = make_value(fake, entity)
            gold.append((len(text), len(text) + len(value), entity))
            text += value
            cursor = end + 1
    return text, gold


def overlaps(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def evaluate(docs, analyzer) -> dict:
    tp_recall = defaultdict(int)
    gold_count = defaultdict(int)
    any_type_hit = defaultdict(int)
    pred_correct = defaultdict(int)
    pred_count = defaultdict(int)
    false_positives: list[str] = []
    misses: list[str] = []

    for text, gold in docs:
        preds = redact(text, analyzer=analyzer)[0].spans

        for g in gold:
            gold_count[g[2]] += 1
            if any(overlaps(g, p) and p[2] == g[2] for p in preds):
                tp_recall[g[2]] += 1
            else:
                misses.append(f"{g[2]}: {text[g[0]:g[1]]!r}")
            if any(overlaps(g, p) for p in preds):
                any_type_hit[g[2]] += 1

        for p in preds:
            pred_count[p[2]] += 1
            if any(overlaps(p, g) and g[2] == p[2] for g in gold):
                pred_correct[p[2]] += 1
            else:
                false_positives.append(f"{p[2]}: {text[p[0]:p[1]]!r}")

    def ratio(n: int, d: int) -> float | None:
        return round(n / d, 3) if d else None

    entities = sorted(set(gold_count) | set(pred_count))
    total_gold = sum(gold_count.values())
    total_pred = sum(pred_count.values())
    return {
        "total_gold_spans": total_gold,
        "overall_recall": ratio(sum(tp_recall.values()), total_gold),
        "overall_any_type_recall": ratio(sum(any_type_hit.values()), total_gold),
        "overall_precision": ratio(sum(pred_correct.values()), total_pred),
        "false_positive_count": len(false_positives),
        "per_entity": {
            e: {
                "gold": gold_count[e],
                "recall": ratio(tp_recall[e], gold_count[e]),
                "any_type_recall": ratio(any_type_hit[e], gold_count[e]),
                "predicted": pred_count[e],
                "precision": ratio(pred_correct[e], pred_count[e]),
            }
            for e in entities
        },
        "sample_misses": misses[:10],
        "sample_false_positives": false_positives[:10],
    }


def main() -> None:
    random.seed(SEED)
    rng = random.Random(SEED)
    fake = Faker("en_US")
    Faker.seed(SEED)

    docs = [build_doc(fake, rng) for _ in range(N_DOCS)]
    results = {
        "seed": SEED,
        "n_docs": N_DOCS,
        "presidio_default": evaluate(docs, build_analyzer(with_luhn_cards=False)),
        "with_luhn_card_recognizer": evaluate(docs, build_analyzer(with_luhn_cards=True)),
    }
    out = Path(__file__).parent / "results" / "redaction_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for name in ("presidio_default", "with_luhn_card_recognizer"):
        r = results[name]
        cc = r["per_entity"]["CREDIT_CARD"]
        print(f"{name}: recall={r['overall_recall']} any_type_recall={r['overall_any_type_recall']} "
              f"precision={r['overall_precision']} false_positives={r['false_positive_count']}")
        print(f"   CREDIT_CARD gold={cc['gold']} recall={cc['recall']} any_type_recall={cc['any_type_recall']} precision={cc['precision']}")


if __name__ == "__main__":
    main()
