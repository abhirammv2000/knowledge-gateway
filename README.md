# knowledge-gateway

A retrieval gateway for enterprise data: legal documents, relational tables and external APIs behind one query interface, with PII redaction before anything is indexed.

**Status: early. Only the redaction layer exists so far.** Everything below the line is planned, not built, and is here so the direction is clear.

## Built

### PII redaction (`src/gateway/redaction.py`)

Detection uses [Presidio](https://github.com/microsoft/presidio) (spaCy NER plus pattern recognizers). The module adds what Presidio does not do on its own:

- **Overlap resolution.** Presidio can return overlapping matches, for example URL fragments inside an email address. Replacing both corrupts the text, so the highest-scoring, then longest, span wins.
- **Consistent, reversible tokens.** The same value always maps to the same token (`<PERSON_1>`), so retrieval can still tell two people apart. A `TokenVault` restores originals for an authorized caller and is never written to the index.

Two behaviors found by testing, not assumed:

- Presidio rejects structurally invalid SSNs (`123-45-6789` is not detected), so evaluation data must use valid-format numbers.
- Presidio's built-in card recognizer left about a quarter of the (Luhn-valid) card numbers in the text. This repo adds a Luhn-checked 12-19 digit recognizer for them.

**Measured** with `eval/redaction_eval.py` (300 synthetic contract-style documents, 1,454 labeled PII spans, fixed seed, exact gold labels), same data for both rows:

| Configuration | Overall recall | Card recall | Card precision | False positives |
|---|---|---|---|---|
| Presidio default | 97.0% | 76.3% | 94.7% | 29 |
| With Luhn card recognizer | 99.0% | 100% | 90.1% | 32 |

The tradeoff is real: the broader rule removes every card number in the test data but flags more non-card digit runs (about 1 in 10 random digit strings passes Luhn). Limits of this result:

- It is synthetic, templated text. It does not predict accuracy on messy real documents.
- 118 card numbers is a small sample for the card rows.
- An earlier run of the evaluation scored phone recall at 86%; that was my generator using non-existent area codes, which Presidio correctly rejects. The generator now uses real ones, and phone recall is 100%.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install presidio-analyzer presidio-anonymizer pytest pytest-asyncio
.venv/Scripts/python -m spacy download en_core_web_lg
.venv/Scripts/python -m pytest
```

## Planned

- Structure-aware chunking for legal contracts, compared against fixed-window chunking on the CUAD dataset (CC BY 4.0)
- Hybrid retrieval with reranking
- Query routing across documents, SQL and an external API, with a guarded read-only text-to-SQL path
- Cost controls (semantic cache, model routing) and tracing
- Exposure over MCP and A2A
