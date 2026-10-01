"""RRF math, index building and reranker sorting are tested with fake embedder and reranker objects, so most
of this runs in milliseconds. One test at the bottom loads the real models."""
import numpy as np
import pytest

from gateway.retrieval import (
    build_index,
    dense_search,
    hybrid_search,
    rerank,
    reciprocal_rank_fusion,
    sparse_search,
)


class FakeEmbedder:
    """A hand-picked vector per text, so dense_search's ranking is predictable without a model."""
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vecs = np.array([self.vectors[t] for t in texts], dtype=np.float64)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms
        return vecs


class FakeReranker:
    def __init__(self, score_fn):
        self.score_fn = score_fn

    def predict(self, pairs, show_progress_bar=False):
        return [self.score_fn(q, d) for q, d in pairs]


# reciprocal_rank_fusion

def test_rrf_rewards_consistent_placement_over_a_first_then_last_split():
    # idx 1 is 2nd in both lists (1/62 + 1/62 = 0.032258), which beats idx 0, first in one and last in
    # the other (1/61 + 1/64 = 0.032018). Checked by computing it, an earlier version of this test
    # asserted the opposite and was wrong. Steady beats erratic
    fused = reciprocal_rank_fusion([[0, 1, 2, 3], [3, 1, 2, 0]])

    assert fused[0] == 1


def test_rrf_of_identical_rankings_preserves_the_order():
    ranking = [3, 1, 4, 0]
    assert reciprocal_rank_fusion([ranking, ranking]) == ranking


def test_rrf_with_a_single_ranking_returns_it_unchanged():
    assert reciprocal_rank_fusion([[2, 0, 1]]) == [2, 0, 1]


# sparse, dense and hybrid search with fakes

def test_sparse_search_ranks_the_lexically_matching_chunk_first():
    texts = ["This Agreement is governed by Delaware law.", "Fees are due within thirty days.", "Either party may terminate this Agreement."]
    index = build_index(texts, embedder=FakeEmbedder({t: [1.0, 0.0] for t in texts}))

    order = sparse_search(index, "governing law Delaware")

    assert order[0] == 0


def test_dense_search_ranks_by_cosine_similarity_to_the_query():
    texts = ["about cats", "about dogs", "about spreadsheets"]
    vectors = {"about cats": [1.0, 0.0], "about dogs": [0.9, 0.1], "about spreadsheets": [0.0, 1.0]}
    index = build_index(texts, embedder=FakeEmbedder(vectors))
    embedder = FakeEmbedder({**vectors, "feline query": [1.0, 0.0]})

    order = dense_search(index, "feline query", embedder=embedder)

    assert order == [0, 1, 2]


def test_build_index_rejects_empty_input():
    with pytest.raises(ValueError):
        build_index([], embedder=FakeEmbedder({}))


# rerank

def test_rerank_only_scores_the_given_candidates_and_resorts_them():
    texts = ["irrelevant", "very relevant to the query", "somewhat relevant", "also irrelevant"]
    scored_calls = []

    def score_fn(q, d):
        scored_calls.append(d)
        return {"very relevant to the query": 5.0, "somewhat relevant": 1.0}.get(d, -99.0)

    index = build_index(texts, embedder=FakeEmbedder({t: [1.0] for t in texts}))

    result = rerank(index, "the query", candidate_idx=[2, 1], reranker=FakeReranker(score_fn))

    assert result == [1, 2]  # idx 1 ("very relevant") scored highest
    assert set(scored_calls) == {"somewhat relevant", "very relevant to the query"}  # idx 0 and 3 never scored


def test_rerank_of_empty_candidates_is_empty_and_does_not_call_the_model():
    calls = []
    index = build_index(["a"], embedder=FakeEmbedder({"a": [1.0]}))

    result = rerank(index, "q", candidate_idx=[], reranker=FakeReranker(lambda q, d: calls.append(1) or 0.0))

    assert result == []
    assert calls == []


def test_hybrid_search_fuses_sparse_and_dense_rankings():
    texts = ["Delaware governing law clause", "unrelated boilerplate text here", "another unrelated clause about fees"]
    vectors = {texts[0]: [1.0, 0.0], texts[1]: [0.0, 1.0], texts[2]: [0.0, 0.9]}
    query_vec = {"governing law": [1.0, 0.0]}
    index = build_index(texts, embedder=FakeEmbedder(vectors))

    order = hybrid_search(index, "governing law", embedder=FakeEmbedder({**vectors, **query_vec}))

    assert order[0] == 0  # matches on both lexical overlap and embedding similarity


# the real models

def test_real_models_rank_the_governing_law_clause_first_for_a_governing_law_query():
    texts = [
        "This Agreement shall be governed by the laws of the State of Delaware.",
        "The Licensee shall pay all fees within thirty days of invoice.",
        "Either party may terminate this Agreement upon ninety days written notice.",
    ]
    index = build_index(texts)

    dense_order = dense_search(index, "what governing law applies to this contract")
    assert dense_order[0] == 0

    reranked = rerank(index, "what governing law applies to this contract", candidate_idx=list(range(len(texts))))
    assert reranked[0] == 0
