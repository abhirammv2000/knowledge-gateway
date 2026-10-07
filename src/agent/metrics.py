"""Prometheus metrics for the agent. Labels are short fixed sets, never user text."""
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("agent_requests_total", "Questions answered, by how they ended.", ["stop_reason"])
DENIED = Counter("agent_denied_total", "Requests refused before any work, by HTTP status.", ["status"])
COST = Counter("agent_cost_usd_total", "Model spend in US dollars, by model that answered.", ["model"])
TOKENS = Counter("agent_tokens_total", "Model tokens used.", ["direction"])
LATENCY = Histogram(
    "agent_request_seconds", "Time to answer one question, including the model calls.",
    buckets=(0.5, 1, 2, 4, 8, 15, 30, 60, 120),
)
TOOL_CALLS = Counter("agent_tool_calls_total", "Tool calls the model made.", ["tool", "ok"])
CACHE = Counter("agent_cache_total", "Semantic cache lookups.", ["result"])  # hit, miss, skipped
FALLBACKS = Counter("agent_fallbacks_total", "Questions where a fallback model answered.")
ACTIVE = Gauge("agent_active_runs", "Questions being worked on right now.")
