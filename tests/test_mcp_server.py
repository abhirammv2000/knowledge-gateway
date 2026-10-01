"""The MCP tools, called through a real MCP client.

Most tests connect to the server in-process with fake models. The last one starts
the server as a subprocess over stdio with the real models, which is how an MCP
client such as Claude Desktop would run it.
"""
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import StdioServerParameters
from mcp.client import Client

from conftest import FakeEmbedder, FakeReranker, FakeStore
from gateway import mcp_server
from gateway.service import GatewayService

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client_factory(monkeypatch):
    monkeypatch.setattr(mcp_server, "_service", GatewayService(FakeStore(), FakeEmbedder(), FakeReranker()))
    return lambda: Client(mcp_server.mcp)


async def test_the_three_tools_are_listed_and_all_are_read_only(client_factory):
    async with client_factory() as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == {"list_contracts", "search_contract", "redact_text"}
    for t in tools:
        assert t.annotations.read_only_hint is True
        assert t.annotations.open_world_hint is False
        assert t.description  # an agent chooses tools from these descriptions


async def test_list_contracts(client_factory):
    async with client_factory() as client:
        result = await client.call_tool("list_contracts", {"contains": "lease"})
    assert not result.is_error
    assert result.structured_content == {"total": 3, "matched": 1, "titles": ["Gamma Lease"]}


async def test_search_contract_returns_passages_best_first(monkeypatch):
    # clauses under 40 words are merged into one chunk, so give this contract real-sized ones
    clauses = "\n\n".join(f"{i}. Clause {i}\n\n" + ("the party shall comply with this clause. " * 8) for i in range(1, 5))
    store = FakeStore({"Long Agreement": clauses})
    monkeypatch.setattr(mcp_server, "_service", GatewayService(store, FakeEmbedder(), FakeReranker()))
    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool("search_contract", {"title": "Long Agreement", "query": "comply", "top_k": 2})
    assert not result.is_error
    passages = result.structured_content["result"]
    assert len(passages) == 2
    assert [p["rank"] for p in passages] == [1, 2]


async def test_an_unknown_title_gives_the_caller_a_message_it_can_act_on(client_factory):
    async with client_factory() as client:
        result = await client.call_tool("search_contract", {"title": "Nope", "query": "rent"})
    assert result.is_error
    assert "list_contracts" in result.content[0].text


async def test_a_bad_top_k_is_explained(client_factory):
    async with client_factory() as client:
        result = await client.call_tool("search_contract", {"title": "Gamma Lease", "query": "rent", "top_k": 0})
    assert result.is_error
    assert "top_k" in result.content[0].text


async def test_a_wrong_argument_type_is_an_error_not_a_crash(client_factory):
    async with client_factory() as client:
        result = await client.call_tool("search_contract", {"title": "Gamma Lease", "query": "rent", "top_k": "many"})
    assert result.is_error


async def test_redact_text(client_factory):
    async with client_factory() as client:
        result = await client.call_tool("redact_text", {"text": "Write to ana.silva@example.com today."})
    assert not result.is_error
    assert "ana.silva@example.com" not in result.structured_content["text"]
    assert result.structured_content["entities"] == {"EMAIL_ADDRESS": 1}


async def test_the_real_server_over_stdio_with_the_real_models(tmp_path):
    cuad = tmp_path / "cuad.json"
    contract = (
        "1. Payment\n\nThe distributor shall pay each invoice within thirty days of receipt.\n\n"
        "2. Indemnification\n\nThe supplier shall indemnify and hold harmless the distributor against "
        "all third party claims arising from defective products.\n\n"
        "3. Governing Law\n\nThis agreement is governed by the laws of the State of New York."
    )
    cuad.write_text(json.dumps({"data": [{"title": "Test Distributor Agreement", "paragraphs": [{"context": contract}]}]}))
    trace_file = tmp_path / "traces.jsonl"

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "gateway.mcp_server"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "KG_CUAD_PATH": str(cuad),
            "KG_TRACE_FILE": str(trace_file),
        },
    )
    query = "which law governs this agreement"
    async with Client(params) as client:
        listed = await client.call_tool("list_contracts", {})
        found = await client.call_tool("search_contract", {"title": "Test Distributor Agreement", "query": query, "top_k": 1})
        redacted = await client.call_tool("redact_text", {"text": "Reach Maria Gonzalez at maria@example.com"})

    assert listed.structured_content["titles"] == ["Test Distributor Agreement"]
    assert "New York" in found.structured_content["result"][0]["text"]
    assert "maria@example.com" not in redacted.structured_content["text"]

    spans = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    names = {s["name"] for s in spans}
    assert {"kg.list_contracts", "kg.search_contract", "kg.rerank", "kg.redact_text"} <= names
    assert query not in trace_file.read_text(encoding="utf-8")
