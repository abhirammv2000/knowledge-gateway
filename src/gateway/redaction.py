"""PII redaction applied before text is chunked, embedded or indexed.

Detection is Presidio (spaCy NER plus pattern recognizers). Two things Presidio
does not do for us, and this module adds:

1. Overlap resolution. Presidio can return overlapping spans, for example a URL
   fragment inside an email address. Replacing both corrupts the text, so the
   highest-scoring, then longest, span wins and the rest are dropped.
2. Consistent, reversible tokens. The same value always maps to the same token
   (<PERSON_1>), so retrieval can still tell two different people apart, and a
   TokenVault can restore the originals for an authorized caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult

DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
]

def luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum, the check real payment card numbers satisfy."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class LuhnCardRecognizer(PatternRecognizer):
    """Any 12-19 digit run that passes Luhn.

    Presidio's built-in card recognizer missed a quarter of the (Luhn-valid)
    card numbers in this repo's evaluation, mostly 12, 15 and unusual-prefix
    16 digit numbers. The cost of this broader rule is false positives on
    non-card digit runs that happen to pass Luhn (roughly 1 in 10 random ones),
    which eval/redaction_eval.py measures rather than assumes.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity="CREDIT_CARD",
            patterns=[Pattern("digit run 12-19", r"\b\d{12,19}\b", 0.3)],
            name="LuhnCardRecognizer",
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        return luhn_valid(pattern_text)


def build_analyzer(with_luhn_cards: bool = True) -> AnalyzerEngine:
    analyzer = AnalyzerEngine()
    if with_luhn_cards:
        analyzer.registry.add_recognizer(LuhnCardRecognizer())
    return analyzer


_analyzer: AnalyzerEngine | None = None


def _get_analyzer() -> AnalyzerEngine:
    # Building the analyzer loads a ~400MB spaCy model, so do it once per process.
    global _analyzer
    if _analyzer is None:
        _analyzer = build_analyzer()
    return _analyzer


@dataclass
class TokenVault:
    """Maps tokens back to original values. Never written to the search index."""

    token_to_value: dict[str, str] = field(default_factory=dict)
    _value_to_token: dict[tuple[str, str], str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)

    def token_for(self, entity_type: str, value: str) -> str:
        key = (entity_type, value)
        if key not in self._value_to_token:
            n = self._counters.get(entity_type, 0) + 1
            self._counters[entity_type] = n
            token = f"<{entity_type}_{n}>"
            self._value_to_token[key] = token
            self.token_to_value[token] = value
        return self._value_to_token[key]

    def restore(self, text: str) -> str:
        for token, value in self.token_to_value.items():
            text = text.replace(token, value)
        return text


@dataclass
class RedactionResult:
    text: str
    spans: list[tuple[int, int, str]]  # (start, end, entity_type) in ORIGINAL coordinates


def resolve_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Keep the best span among any that overlap: higher score, then longer."""
    ranked = sorted(results, key=lambda r: (-r.score, -(r.end - r.start), r.start))
    kept: list[RecognizerResult] = []
    for r in ranked:
        if all(r.end <= k.start or r.start >= k.end for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda r: r.start)


def redact(
    text: str,
    vault: TokenVault | None = None,
    entities: list[str] | None = None,
    score_threshold: float = 0.4,
    analyzer: AnalyzerEngine | None = None,
) -> tuple[RedactionResult, TokenVault]:
    vault = vault or TokenVault()
    found = (analyzer or _get_analyzer()).analyze(
        text=text,
        language="en",
        entities=entities or DEFAULT_ENTITIES,
        score_threshold=score_threshold,
    )
    kept = resolve_overlaps(found)

    out: list[str] = []
    cursor = 0
    for r in kept:
        out.append(text[cursor : r.start])
        out.append(vault.token_for(r.entity_type, text[r.start : r.end]))
        cursor = r.end
    out.append(text[cursor:])

    spans = [(r.start, r.end, r.entity_type) for r in kept]
    return RedactionResult(text="".join(out), spans=spans), vault
