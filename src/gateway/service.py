"""What the gateway does for a caller: list contracts, search inside one, redact text.

This is the layer the MCP server calls. The retrieval, chunking and redaction
code it uses is the same code the evaluations measured, put together in the same
order: structure-aware chunks, BM25 and dense search fused with RRF, then the
cross-encoder over the top 10.

Each stage of a search is its own tracing span, so a slow query shows whether the
time went into building the index, the two searches, the fusion or the reranker.
Spans carry sizes and timings only, never the query or any contract text (see
tracing.py).
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

from gateway.chunking import structure_aware_chunks
from gateway.redaction import TokenVault, redact
from gateway.retrieval import (
    Index,
    build_index,
    dense_search,
    reciprocal_rank_fusion,
    rerank,
    sparse_search,
)
from gateway.tracing import get_tracer

DEFAULT_CUAD_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "CUAD_v1.json"

RERANK_POOL = 10  # the pool size the reranking evaluation used
MAX_TOP_K = 20
MAX_LIST = 200
MAX_REDACT_CHARS = 100_000
CACHE_SIZE = 8  # contracts whose index is kept in memory


class ContractNotFound(LookupError):
    pass


class ContractStore:
    """The CUAD contracts, read from the JSON file the evaluations use."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.environ.get("KG_CUAD_PATH") or DEFAULT_CUAD_PATH)
        self._texts: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._texts is None:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"{self.path} not found. Download CUAD_v1.json into data/raw/ (see the README) "
                    "or point KG_CUAD_PATH at it."
                )
            data = json.loads(self.path.read_text(encoding="utf-8"))["data"]
            self._texts = {doc["title"]: doc["paragraphs"][0]["context"] for doc in data}
        return self._texts

    def titles(self) -> list[str]:
        return sorted(self._load())

    def text(self, title: str) -> str:
        try:
            return self._load()[title]
        except KeyError:
            raise ContractNotFound(
                f"no contract titled {title!r}. Use list_contracts to find the exact title."
            ) from None


class GatewayService:
    def __init__(self, store: ContractStore | None = None, embedder: Any = None, reranker: Any = None) -> None:
        self.store = store or ContractStore()
        # None means the real models, loaded on first use (see retrieval.py)
        self._embedder = embedder
        self._reranker = reranker
        self._indexes: OrderedDict[str, tuple[Index, list[str]]] = OrderedDict()

    def list_contracts(self, contains: str = "", limit: int = 25) -> dict[str, Any]:
        with get_tracer().start_as_current_span("kg.list_contracts") as span:
            titles = self.store.titles()
            needle = contains.lower()
            matches = [t for t in titles if needle in t.lower()]
            span.set_attribute("contracts.total", len(titles))
            span.set_attribute("contracts.matched", len(matches))
            limit = min(max(1, limit), MAX_LIST)
            return {"total": len(titles), "matched": len(matches), "titles": matches[:limit]}

    def _index_for(self, title: str) -> tuple[Index, list[str], bool]:
        """The search index for one contract, built on first use and then kept (LRU)."""
        if title in self._indexes:
            self._indexes.move_to_end(title)
            index, texts = self._indexes[title]
            return index, texts, True

        tracer = get_tracer()
        with tracer.start_as_current_span("kg.chunk") as span:
            chunks = structure_aware_chunks(self.store.text(title))
            texts = [c.text for c in chunks]
            span.set_attribute("chunks", len(texts))
        with tracer.start_as_current_span("kg.index_build") as span:
            index = build_index(texts, self._embedder)
            span.set_attribute("chunks", len(texts))

        self._indexes[title] = (index, texts)
        if len(self._indexes) > CACHE_SIZE:
            self._indexes.popitem(last=False)
        return index, texts, False

    def search_contract(
        self, title: str, query: str, top_k: int = 5, use_reranker: bool = True
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query is empty")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

        tracer = get_tracer()
        with tracer.start_as_current_span("kg.search_contract") as span:
            span.set_attribute("query.chars", len(query))
            span.set_attribute("top_k", top_k)
            span.set_attribute("reranked", use_reranker)

            index, texts, cache_hit = self._index_for(title)
            span.set_attribute("index.cache_hit", cache_hit)
            span.set_attribute("index.chunks", len(texts))

            with tracer.start_as_current_span("kg.sparse_search"):
                sparse = sparse_search(index, query)
            with tracer.start_as_current_span("kg.dense_search"):
                dense = dense_search(index, query, self._embedder)
            with tracer.start_as_current_span("kg.fuse"):
                fused = reciprocal_rank_fusion([sparse, dense])

            if use_reranker:
                with tracer.start_as_current_span("kg.rerank") as rspan:
                    pool = fused[:RERANK_POOL]
                    rspan.set_attribute("pool", len(pool))
                    # only the pool is re-scored, anything past it keeps its fused order
                    ordered = rerank(index, query, pool, self._reranker) + fused[RERANK_POOL:]
            else:
                ordered = fused

            return [
                {"rank": rank, "chunk_index": i, "words": len(texts[i].split()), "text": texts[i]}
                for rank, i in enumerate(ordered[:top_k], start=1)
            ]

    def redact_text(self, text: str) -> dict[str, Any]:
        if len(text) > MAX_REDACT_CHARS:
            raise ValueError(f"text is {len(text)} characters, the limit is {MAX_REDACT_CHARS}")

        with get_tracer().start_as_current_span("kg.redact_text") as span:
            # A fresh vault per call that is thrown away, so nothing sent through
            # the server can be turned back into the original value.
            result, _ = redact(text, vault=TokenVault())
            counts: dict[str, int] = {}
            for _, _, entity_type in result.spans:
                counts[entity_type] = counts.get(entity_type, 0) + 1
            span.set_attribute("text.chars", len(text))
            span.set_attribute("entities", len(result.spans))
            return {"text": result.text, "entities": counts, "entity_count": len(result.spans)}
