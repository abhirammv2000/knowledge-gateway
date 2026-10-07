"""The embedding model the semantic cache uses to compare questions.

cache_eval.py measured both options on the same question pairs. text-embedding-3-small keeps
paraphrases and different questions further apart than the local MiniLM model, so it is the
default, with a threshold chosen from that measurement. MiniLM needs no network or key and is the
fallback.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# thresholds picked from eval/cache_eval.py, with the guard on. Each is just above the most similar pair of
# different-clause questions the eval found, so no different question hit, and below most paraphrases.
DEFAULT_THRESHOLDS = {"openai": 0.70, "minilm": 0.80}
OPENAI_MODEL = "text-embedding-3-small"


def make_embedder(kind: str) -> Callable[[str], np.ndarray]:
    if kind == "openai":
        import litellm

        def embed(text: str) -> np.ndarray:
            data = litellm.embedding(model=OPENAI_MODEL, input=[text], timeout=15).data
            return np.asarray(data[0]["embedding"], dtype=np.float32)

        return embed
    if kind == "minilm":
        from agent.tools import model_lock
        from gateway.retrieval import get_embedder

        model = get_embedder()

        def embed(text: str) -> np.ndarray:
            with model_lock:
                return model.encode([text], normalize_embeddings=True)[0]

        return embed
    raise ValueError(f"unknown embedder {kind!r}, use 'openai' or 'minilm'")
