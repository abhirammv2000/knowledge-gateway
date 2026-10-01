"""Fakes shared by the service, tracing and MCP tests, so they don't load a real model (the stdio test at the
bottom of test_mcp_server.py is the one that does)."""
import numpy as np
import pytest

from gateway.service import ContractStore, GatewayService

CONTRACTS = {
    "Acme Distributor Agreement": (
        "1. Payment\n\nThe buyer pays each invoice within thirty days of receipt.\n\n"
        "2. Indemnification\n\nThe supplier shall indemnify the buyer against any third party claims "
        "arising from defective goods.\n\n"
        "3. Termination\n\nEither party may terminate this agreement on ninety days written notice."
    ),
    "Beta Software License": (
        "1. Grant\n\nThe licensor grants a non-exclusive licence to use the software.\n\n"
        "2. Fees\n\nThe licensee pays an annual fee in advance."
    ),
    "Gamma Lease": "1. Rent\n\nRent is due on the first day of each month.",
}


class FakeStore(ContractStore):
    def __init__(self, contracts=None):
        self._texts = dict(CONTRACTS if contracts is None else contracts)


class FakeEmbedder:
    """A fixed two-number vector per text."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vecs = np.array([[len(t) % 7 + 1.0, len(t.split()) % 5 + 1.0] for t in texts])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class FakeReranker:
    """Scores a passage by how many of the query's words it has."""

    def predict(self, pairs, show_progress_bar=False):
        return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in pairs]


@pytest.fixture
def service():
    return GatewayService(FakeStore(), FakeEmbedder(), FakeReranker())
