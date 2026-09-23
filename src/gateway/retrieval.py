"""Sparse (BM25), dense (embedding cosine similarity), hybrid (reciprocal rank
fusion of the two), and a cross-encoder reranking stage on top of any of them.

Models are loaded lazily and cached at module level, the same pattern as
redaction.py's Presidio analyzer, so importing this module never pays the
~4-second model-load cost until a search actually happens, and there's a
single shared instance per process rather than one per call.

Reciprocal rank fusion: score(d) = sum over rankers of 1 / (k + rank_r(d)),
rank_r(d) = 1 for the top result of ranker r. k=60 is the constant from the
original RRF paper (Cormack, Clarke & Buettcher, SIGIR 2009) and is what most
implementations use unchanged; there's no dataset-specific reason to retune it
here, so it's left as the paper's default rather than picked to fit this data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(DEFAULT_RERANKER_MODEL)
    return _reranker


def _tokenize(s: str) -> list[str]:
    # Matches eval/chunking_eval.py's tokenizer so BM25 behavior is identical
    # to what was already measured there — this module isn't introducing a
    # second, different sparse-retrieval implementation.
    import re
    return re.findall(r"\w+", s.lower())


@dataclass
class Index:
    """One contract's (or document's) chunks, ready to search."""
    texts: list[str]
    bm25: BM25Okapi
    embeddings: np.ndarray  # (n_chunks, dim), L2-normalized


def build_index(texts: list[str], embedder: SentenceTransformer | None = None) -> Index:
    if not texts:
        raise ValueError("build_index requires at least one text")
    bm25 = BM25Okapi([_tokenize(t) or ["_"] for t in texts])
    emb = (embedder or get_embedder()).encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return Index(texts=texts, bm25=bm25, embeddings=np.asarray(emb))


def sparse_search(index: Index, query: str) -> list[int]:
    """Chunk indices ranked by BM25 score, best first."""
    scores = index.bm25.get_scores(_tokenize(query))
    return sorted(range(len(index.texts)), key=lambda i: (-scores[i], i))


def dense_search(index: Index, query: str, embedder: SentenceTransformer | None = None) -> list[int]:
    """Chunk indices ranked by cosine similarity, best first.

    Embeddings are L2-normalized at index-build time, so a plain dot product
    against a normalized query vector equals cosine similarity.
    """
    q = (embedder or get_embedder()).encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = index.embeddings @ q
    return sorted(range(len(index.texts)), key=lambda i: (-scores[i], i))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Combine multiple ranked-index lists into one, by RRF score descending.

    An index missing from one ranking (e.g. it scored 0 and BM25Okapi still
    returns it, so in practice every index appears in every input ranking
    here) simply contributes 0 from that ranking rather than raising.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def hybrid_search(index: Index, query: str, embedder: SentenceTransformer | None = None) -> list[int]:
    return reciprocal_rank_fusion([sparse_search(index, query), dense_search(index, query, embedder)])


def rerank(index: Index, query: str, candidate_idx: list[int], reranker: CrossEncoder | None = None) -> list[int]:
    """Re-score a candidate list with a cross-encoder and return it re-sorted.

    Only the candidates passed in are scored — this is meant to sit on top of
    sparse_search/dense_search/hybrid_search's top-N, not to replace a first
    retrieval stage, since scoring every chunk against the query with a
    cross-encoder is the accuracy/latency tradeoff a first stage exists to avoid.
    """
    if not candidate_idx:
        return []
    pairs = [(query, index.texts[i]) for i in candidate_idx]
    scores = (reranker or get_reranker()).predict(pairs, show_progress_bar=False)
    order = sorted(range(len(candidate_idx)), key=lambda j: -scores[j])
    return [candidate_idx[j] for j in order]
