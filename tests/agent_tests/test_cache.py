"""The semantic cache, with hand-made vectors so every similarity is known."""
import numpy as np
import pytest

from agent.cache import SemanticCache

VECTORS = {
    "terminate notice": [1.0, 0.0],
    "how much notice to end it": [0.95, 0.31],  # cosine about 0.95 with the first
    "payment terms": [0.0, 1.0],
}


def embed(text):
    return np.array(VECTORS[text])


class Clock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def cache(clock):
    return SemanticCache(embed, threshold=0.9, ttl_seconds=1000, max_entries=3, clock=clock)


ANSWER = {"found": True, "answer": "Ninety days.", "citations": []}


def test_the_same_question_hits(cache):
    cache.put("terminate notice", "Acme", "model-1", ANSWER)

    hit = cache.get("terminate notice", "Acme", "model-1")

    assert hit.answer == ANSWER and hit.similarity == pytest.approx(1.0, abs=1e-3)


def test_a_question_that_means_nearly_the_same_hits(cache):
    cache.put("terminate notice", "Acme", "model-1", ANSWER)

    hit = cache.get("how much notice to end it", "Acme", "model-1")

    assert hit is not None and hit.matched_question == "terminate notice"


def test_a_different_question_misses(cache):
    cache.put("terminate notice", "Acme", "model-1", ANSWER)

    assert cache.get("payment terms", "Acme", "model-1") is None


def test_the_threshold_decides(clock):
    strict = SemanticCache(embed, threshold=0.99, clock=clock)
    strict.put("terminate notice", "Acme", "m", ANSWER)

    assert strict.get("how much notice to end it", "Acme", "m") is None


def test_another_contract_never_hits(cache):
    cache.put("terminate notice", "Acme", "model-1", ANSWER)

    assert cache.get("terminate notice", "Beta", "model-1") is None


def test_another_model_setup_never_hits(cache):
    cache.put("terminate notice", "Acme", "model-1", ANSWER)

    assert cache.get("terminate notice", "Acme", "model-2") is None


def test_old_entries_expire(cache, clock):
    cache.put("terminate notice", "Acme", "model-1", ANSWER)
    clock.now += 1001

    assert cache.get("terminate notice", "Acme", "model-1") is None


def test_the_oldest_entries_are_dropped_past_the_limit(cache):
    for contract in ("A", "B", "C", "D"):
        cache.put("terminate notice", contract, "m", {"answer": contract})

    assert cache.get("terminate notice", "A", "m") is None
    assert cache.get("terminate notice", "D", "m").answer == {"answer": "D"}
    assert cache.stats()["entries"] == 3


def test_hits_are_counted(cache):
    cache.put("terminate notice", "Acme", "m", ANSWER)
    cache.get("terminate notice", "Acme", "m")
    cache.get("terminate notice", "Acme", "m")
    cache.get("payment terms", "Acme", "m")

    assert cache.stats() == {"entries": 1, "hits": 2}


def test_the_best_match_wins_when_several_are_stored(cache):
    cache.put("how much notice to end it", "Acme", "m", {"answer": "near"})
    cache.put("terminate notice", "Acme", "m", {"answer": "exact"})

    assert cache.get("terminate notice", "Acme", "m").answer == {"answer": "exact"}


def test_entries_survive_a_restart(tmp_path, clock):
    path = tmp_path / "c.db"
    SemanticCache(embed, path, threshold=0.9, clock=clock).put("terminate notice", "Acme", "m", ANSWER)

    assert SemanticCache(embed, path, threshold=0.9, clock=clock).get("terminate notice", "Acme", "m") is not None
