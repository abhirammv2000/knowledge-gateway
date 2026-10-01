import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from conftest import FakeEmbedder, FakeReranker, FakeStore
from gateway import service as service_module
from gateway.service import ContractNotFound, ContractStore, GatewayService
from gateway.tracing import setup_tracing, shutdown_tracing


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    setup_tracing(exporter)
    yield exporter
    shutdown_tracing()


def test_list_contracts_filters_case_insensitively_and_caps(service):
    out = service.list_contracts("LICENSE")
    assert out == {"total": 3, "matched": 1, "titles": ["Beta Software License"]}
    assert len(service.list_contracts("", limit=2)["titles"]) == 2


def test_search_returns_ranked_passages_from_that_contract(service):
    out = service.search_contract("Acme Distributor Agreement", "who indemnifies the buyer", top_k=3)
    assert [r["rank"] for r in out] == list(range(1, len(out) + 1))
    assert all(r["text"] for r in out)
    assert any("indemnify" in r["text"] for r in out)


def test_the_reranker_decides_the_order_when_it_is_on(service):
    # the fake reranker scores by shared words, so the indemnification passage should come first
    out = service.search_contract("Acme Distributor Agreement", "indemnify claims defective", top_k=1)
    assert "indemnify" in out[0]["text"]


def test_search_without_the_reranker_is_repeatable(service):
    a = service.search_contract("Acme Distributor Agreement", "payment", top_k=5, use_reranker=False)
    b = service.search_contract("Acme Distributor Agreement", "payment", top_k=5, use_reranker=False)
    assert a == b


def test_asking_for_more_than_the_rerank_pool_returns_each_passage_once():
    clauses = "\n\n".join(f"{i}. Clause {i}\n\n" + ("word " * 60) for i in range(1, 30))
    svc = GatewayService(FakeStore({"Big": clauses}), FakeEmbedder(), FakeReranker())
    out = svc.search_contract("Big", "word", top_k=service_module.MAX_TOP_K)
    ids = [r["chunk_index"] for r in out]
    assert len(ids) > service_module.RERANK_POOL
    assert len(ids) == len(set(ids))


def test_unknown_contract_and_bad_arguments_are_rejected_clearly(service):
    with pytest.raises(ContractNotFound, match="list_contracts"):
        service.search_contract("Nope", "x")
    with pytest.raises(ValueError, match="empty"):
        service.search_contract("Gamma Lease", "   ")
    with pytest.raises(ValueError, match="top_k"):
        service.search_contract("Gamma Lease", "rent", top_k=0)
    with pytest.raises(ValueError, match="characters"):
        service.redact_text("x" * (service_module.MAX_REDACT_CHARS + 1))


def test_a_missing_data_file_names_the_file_and_the_fix(tmp_path):
    svc = GatewayService(ContractStore(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="KG_CUAD_PATH"):
        svc.list_contracts()


def test_second_search_in_the_same_contract_reuses_the_index(service, spans):
    service.search_contract("Gamma Lease", "rent")
    service.search_contract("Gamma Lease", "rent")
    finished = spans.get_finished_spans()
    searches = [s for s in finished if s.name == "kg.search_contract"]
    assert [s.attributes["index.cache_hit"] for s in searches] == [False, True]
    assert len([s for s in finished if s.name == "kg.index_build"]) == 1


def test_least_recently_used_index_is_dropped(monkeypatch):
    monkeypatch.setattr(service_module, "CACHE_SIZE", 2)
    svc = GatewayService(FakeStore(), FakeEmbedder(), FakeReranker())
    for title in ("Gamma Lease", "Beta Software License", "Acme Distributor Agreement"):
        svc.search_contract(title, "the")
    assert list(svc._indexes) == ["Beta Software License", "Acme Distributor Agreement"]


def test_every_search_stage_has_a_span_under_the_search_span(service, spans):
    service.search_contract("Acme Distributor Agreement", "payment")
    finished = spans.get_finished_spans()
    root = next(s for s in finished if s.name == "kg.search_contract")
    children = {s.name for s in finished if s.parent and s.parent.span_id == root.context.span_id}
    assert {"kg.sparse_search", "kg.dense_search", "kg.fuse", "kg.rerank"} <= children


def test_spans_never_contain_the_query_or_contract_text(service, spans):
    service.search_contract("Acme Distributor Agreement", "zebra unicorn secret indemnify")
    service.redact_text("Contact Maria Gonzalez at maria.gonzalez@example.com")
    for s in spans.get_finished_spans():
        for value in (s.attributes or {}).values():
            text = str(value).lower()
            assert "zebra" not in text and "indemnify" not in text
            assert "gonzalez" not in text and "example.com" not in text


def test_redaction_replaces_personal_data_and_counts_it(service):
    out = service.redact_text("Contact Maria Gonzalez at maria.gonzalez@example.com or maria.gonzalez@example.com.")
    assert "maria.gonzalez@example.com" not in out["text"]
    assert "<EMAIL_ADDRESS_1>" in out["text"] and "<EMAIL_ADDRESS_2>" not in out["text"]  # same value, same token
    assert out["entities"]["EMAIL_ADDRESS"] == 2
    assert out["entity_count"] == sum(out["entities"].values())
