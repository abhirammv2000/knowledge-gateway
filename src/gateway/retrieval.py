"""BM25, dense search, hybrid (the two fused with reciprocal rank fusion), and a cross-encoder reranker on top.

The models load the first time they're used and stay cached, like the Presidio analyzer in redaction.py,
so importing this doesn't cost the ~4 second load. RRF scores a chunk as the sum of 1 / (k + rank) over the
rankers. k=60 is the paper's value and I haven't tuned it to this data.
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
    # same tokenizer as chunking_eval.py, so BM25 behaves like it did in that eval
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
    """Chunk indices ranked by cosine similarity, best first. The vectors are normalized, so a dot product is the cosine."""
    q = (embedder or get_embedder()).encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = index.embeddings @ q
    return sorted(range(len(index.texts)), key=lambda i: (-scores[i], i))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge ranked lists into one by RRF score. An index missing from a list just adds 0 for that list."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def hybrid_search(index: Index, query: str, embedder: SentenceTransformer | None = None) -> list[int]:
    return reciprocal_rank_fusion([sparse_search(index, query), dense_search(index, query, embedder)])


def rerank(index: Index, query: str, candidate_idx: list[int], reranker: CrossEncoder | None = None) -> list[int]:
    """Re-score the candidates with the cross-encoder and return them re-sorted. Only the ones passed in are scored,
    since scoring every chunk is the cost a first stage is there to avoid."""
    if not candidate_idx:
        return []
    pairs = [(query, index.texts[i]) for i in candidate_idx]
    scores = (reranker or get_reranker()).predict(pairs, show_progress_bar=False)
    order = sorted(range(len(candidate_idx)), key=lambda j: -scores[j])
    return [candidate_idx[j] for j in order]
