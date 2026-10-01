"""MCP server so an agent can search contracts and redact text. Three read-only tools: list_contracts,
search_contract and redact_text.

    PYTHONPATH=src .venv/Scripts/python -m gateway.mcp_server

It runs over stdio, so nothing here can print to stdout (that's the protocol). Logs and the console
trace go to stderr. Written for mcp 2.x, where the class is MCPServer (it was FastMCP in 1.x).
"""
from __future__ import annotations

from typing import Any, Callable

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from gateway.service import ContractNotFound, GatewayService
from gateway.tracing import setup_tracing, shutdown_tracing

mcp = MCPServer(
    "knowledge-gateway",
    instructions=(
        "Search legal contracts and redact personal data. Use list_contracts to find the exact "
        "title of a contract, then search_contract to get the passages that answer a question. "
        "search_contract returns contract text, so it can contain personal data: pass anything "
        "you are going to show or store through redact_text first."
    ),
)

# all three only read
_READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

_service: GatewayService | None = None


def get_service() -> GatewayService:
    global _service
    if _service is None:
        _service = GatewayService()
    return _service


def _run(call: Callable[..., Any], *args: Any) -> Any:
    """Call the service and turn fixable mistakes into ToolErrors, since the SDK hides the message of any other exception."""
    try:
        return call(*args)
    except (ContractNotFound, ValueError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e


@mcp.tool(annotations=_READ_ONLY)
def list_contracts(contains: str = "", limit: int = 25) -> dict[str, Any]:
    """List contract titles, optionally only those containing some text (case-insensitive).

    Returns the number of contracts in total, how many matched, and up to `limit` titles.
    """
    return _run(get_service().list_contracts, contains, limit)


@mcp.tool(annotations=_READ_ONLY)
def search_contract(title: str, query: str, top_k: int = 5, use_reranker: bool = True) -> list[dict[str, Any]]:
    """Find the passages in one contract that best answer a question.

    `title` must be an exact title from list_contracts. Searches the contract's clauses with BM25
    and dense embeddings fused together, then reranks the top 10 with a cross-encoder (turn that
    off with use_reranker=false for a faster, less precise answer). Returns up to `top_k` passages,
    best first, each with its text. The first search in a contract takes a few seconds while its
    index is built, later ones are fast.
    """
    return _run(get_service().search_contract, title, query, top_k, use_reranker)


@mcp.tool(annotations=_READ_ONLY)
def redact_text(text: str) -> dict[str, Any]:
    """Replace personal data (names, emails, phone numbers, card numbers and so on) with tokens.

    The same value always gets the same token within one call, for example <PERSON_1>, so the
    redacted text still shows who is who. The tokens cannot be reversed through this server.
    Returns the redacted text and a count of what was found by type.
    """
    return _run(get_service().redact_text, text)


def main() -> None:
    setup_tracing()  # off unless KG_TRACE_FILE, KG_TRACE_CONSOLE or OTEL_EXPORTER_OTLP_ENDPOINT is set
    try:
        mcp.run(transport="stdio")
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    main()
