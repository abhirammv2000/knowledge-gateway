import numpy as np
import pytest

from agent import embeddings


def test_an_unknown_embedder_is_refused():
    with pytest.raises(ValueError):
        embeddings.make_embedder("word2vec")


def test_the_openai_embedder_returns_the_vector_litellm_gives(monkeypatch):
    seen = {}

    class Reply:
        data = [{"embedding": [0.1, 0.2, 0.3]}]

    def fake(model, input, timeout):
        seen.update(model=model, input=input, timeout=timeout)
        return Reply()

    import litellm
    monkeypatch.setattr(litellm, "embedding", fake)

    vec = embeddings.make_embedder("openai")("hello")

    assert np.allclose(vec, [0.1, 0.2, 0.3])
    assert seen == {"model": "text-embedding-3-small", "input": ["hello"], "timeout": 15}


def test_every_embedder_has_a_threshold():
    assert set(embeddings.DEFAULT_THRESHOLDS) == {"openai", "minilm"}
    assert all(0.5 < t < 1 for t in embeddings.DEFAULT_THRESHOLDS.values())
