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
- Phone recall is 100%, but phone precision is only about 93-94%: long non-PII digit runs (invoice and order numbers) are sometimes flagged as phone numbers, with or without the Luhn recognizer.
- An earlier run of the evaluation scored phone recall at 86%; that was my generator using non-existent area codes, which Presidio correctly rejects. The generator now uses real ones, and phone recall is 100%.

### Chunking for legal contracts (`src/gateway/chunking.py`)

Two chunkers over the same text: a 250-word sliding window with 50 words of overlap (baseline), and a structure-aware chunker that splits at clause headings (numbered clauses, ARTICLE/SECTION, ALL-CAPS headings), merges tiny sections, and splits oversized ones at paragraph, then sentence, then word boundaries, re-attaching the heading to later pieces. Heading patterns were chosen from the data: numbered headings appear in 447 of 510 contracts, ALL-CAPS in 310, ARTICLE/SECTION in 112.

**Measured** with `eval/chunking_eval.py` on [CUAD](https://www.atticusprojectai.org/cuad/) (CC BY 4.0): 510 contracts, 6,702 questions with expert-marked clause spans, BM25 retrieval within each contract. A hit means the retrieved chunks cover at least 50% of a gold clause span.

| | Fixed window | Structure-aware |
|---|---|---|
| Hit@5 (equal top-k) | 69.8% | 66.9% |
| Hit at a 1,250-word budget (equal context) | 69.8% | 70.5% |
| Clauses left uncut by a chunk boundary | 94.1% | 97.7% |
| Median chunk size | 250 words | 169 words |

What this does and does not show:

- **My hypothesis was that structure-aware chunking would improve retrieval. It did not.** At equal top-k it was 2.8 points worse (95% interval -3.6 to -2.1). That comparison was unfair, because its chunks are smaller, so top-5 returns less text.
- At an **equal word budget** the difference is +0.7 points with a 95% interval of -0.04 to +1.44, a statistical tie. The budget comparison was added after I saw the top-k result; the 1,250-word primary budget was declared before running it.
- What structure-aware chunking does buy is clause integrity: about 3.7 points fewer clauses cut across a chunk boundary. Whether that helps answer quality with an LLM is not measured here.
- Retrieval is BM25 only. A dense or hybrid retriever may behave differently.

```bash
# CUAD_v1.json (40 MB, CC BY 4.0) into data/raw/, which is gitignored
curl -L -o data/raw/CUAD_v1.json "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"
PYTHONPATH=src .venv/Scripts/python eval/chunking_eval.py
```

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install presidio-analyzer presidio-anonymizer pytest pytest-asyncio faker rank-bm25 numpy
.venv/Scripts/python -m spacy download en_core_web_lg
.venv/Scripts/python -m pytest
```

## Planned

- Hybrid retrieval with reranking
- Query routing across documents, SQL and an external API, with a guarded read-only text-to-SQL path
- Cost controls (semantic cache, model routing) and tracing
- Exposure over MCP and A2A
