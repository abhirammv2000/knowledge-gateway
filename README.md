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
- Redaction precision and recall have **not been measured yet**. No accuracy claim is made until that evaluation exists.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install presidio-analyzer presidio-anonymizer pytest pytest-asyncio
.venv/Scripts/python -m spacy download en_core_web_lg
.venv/Scripts/python -m pytest
```

## Planned

- Redaction evaluation on synthetic PII (precision and recall per entity type)
- Structure-aware chunking for legal contracts, compared against fixed-window chunking on the CUAD dataset (CC BY 4.0)
- Hybrid retrieval with reranking
- Query routing across documents, SQL and an external API, with a guarded read-only text-to-SQL path
- Cost controls (semantic cache, model routing) and tracing
- Exposure over MCP and A2A
