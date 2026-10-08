# What is built, and where

A map from the usual AI engineering topics to this repo. "Built" means the code exists and has tests. "Measured" means there is a saved result in `eval/results/`. Anything not done says so.

## Software foundations

| Topic | Where | Status |
|---|---|---|
| Async programming | `loop.py` (timeouts with `asyncio.timeout`), `service.py` (a semaphore caps concurrent runs, blocking work goes to threads) | Built |
| APIs and HTTP | `api.py`: 401, 402, 404, 409, 422, 429, 503 with `Retry-After`, request ids, idempotency keys | Built |
| SQL and databases | SQLite for accounts, memory, cache and idempotency, with small migrations for older files | Built. One instance only. |
| Git and CI | GitHub Actions runs the full test suite from a clean machine | Built |
| Containers | `Dockerfile` (CPU torch, baked models, non-root). It was built and smoke tested. | Built |
| Testing | About 270 tests with no network. I broke the code on purpose to check the important ones fail. | Built |

## Working with models

| Topic | Where | Status |
|---|---|---|
| LLM APIs, three providers | `llm.py` through LiteLLM: Anthropic, OpenAI, Gemini | Built. Only OpenAI has a full accuracy row. |
| Tool calling | `tools.py`: `list_contracts`, `search_contract`, `save_note`, `submit_answer` | Built, measured (35 of 36 trace checks) |
| Structured outputs | `schema.py`: the answer is a Pydantic object, checked before it is used | Built |
| Embeddings | `gateway/retrieval.py` (MiniLM), `agent/embeddings.py` (MiniLM or OpenAI) | Built, measured |
| Context and tokens | per-question token budget, question length cap, five-turn history cap | Built |
| Prompting | `loop.py`: system prompt, nudges, repair prompts | Built. The "search again before saying absent" variant is untested. |

## Retrieval

| Topic | Where | Status |
|---|---|---|
| Chunking | `gateway/chunking.py`, clause-aware | Built, measured |
| Lexical and vector search, hybrid | BM25 and dense fused with RRF | Built, measured |
| Reranking | cross-encoder over the top 10 | Built, measured (+3.7 points Hit@5) |
| Retrieval evaluation | `eval/chunking_eval.py`, `eval/hybrid_retrieval_eval.py` on 6,702 questions | Measured |
| Metadata filtering | search is scoped to one contract | Partial. No other metadata filters. |

## Agents

| Topic | Where | Status |
|---|---|---|
| Tool schemas and state | `tools.py`, `RunState` | Built |
| Retries and failure recovery | router retry and fallback, tool errors returned as readable text | Built, failover time measured |
| Loop prevention | step, tool call, token and time limits, repeated identical calls blocked | Built |
| Permissions | the model has read-only tools. Its one write only proposes. | Built |
| Human approval | the model proposes notes and the user approves them (`/v1/notes`). DevFlow gates risky actions. | Built |
| Idempotency | `idempotency.py`, `Idempotency-Key` on `/v1/ask` | Built |
| Memory | conversations and approved notes, redacted, per key, deletable | Built |
| Long-conversation memory | last five turns kept, older turns folded into a running summary (`service.py`, `memory.py`) | Built, tested with a scripted model. Not run on a live model. |
| Streaming | `POST /v1/ask/stream`: progress events, then the answer once its citations are checked | Built, tested through the test client |
| MCP | the retrieval layer is an MCP server (`gateway/mcp_server.py`). DevFlow uses MCP servers. | Built. The agent calls its tools in process, not through an MCP client. |
| Multi-agent | not built on purpose. One loop was enough, and more agents would add cost and failure modes. | Not done |
| Coding agent | DevFlow, a separate repo | Built there |

## Evaluation

| Topic | Where | Status |
|---|---|---|
| Task success with intervals | `eval/agent_eval.py`, 100 CUAD questions, bootstrap intervals | Measured (GPT-4o, GPT-4o-mini) |
| Trajectory grading | `eval/tool_eval.py` | Measured |
| Cost and latency | `eval/latency_eval.py`, `eval/failover_timing.py` | Measured |
| Cache failure modes | `eval/cache_eval.py` | Measured |
| Model comparison | paired intervals between models | Measured for two models |
| Cheaper-model cascade | `eval/routing_sim.py` | Measured. It lost, so it was not built. |
| Regression gate with real models | would cost money on every change | Not done. Scripted-model tests run in CI instead. |

## Safety

| Topic | Where | Status |
|---|---|---|
| Hallucination control | every citation is checked against what the model was shown, else the agent refuses | Built, measured (no unverified answers in 100 questions) |
| Prompt injection | tagged passages, canary string, citation check, note approval | Built. The injection eval is written but has no saved result. |
| PII | Presidio before the model, in the index and in memory | Built, measured (99.0% recall on synthetic data) |
| Guardrails | all of the above, plus input length and output leak checks | Built |
| Drift | `drift.py`, `admin drift-report` | Built. No live traffic yet. |
| Bias auditing | not done | Not done |

## Production

| Topic | Where | Status |
|---|---|---|
| Authentication and access control | hashed API keys, admin keys, per-key data | Built |
| Rate limits and budgets | per key per minute, per key per day, whole service per day | Built |
| Observability | OpenTelemetry spans without text, Prometheus metrics, counts-only logs, usage ledger | Built |
| Cost monitoring | cost by model and by arm, per request in the ledger | Built |
| Model routing | ordered fallback chain, plus a percentage split between two models | Built |
| A/B testing | sticky arms, separate caches, feedback, `ab-report` | Built. No live traffic yet. |
| Semantic cache | guarded against flipped meanings | Built, measured |
| Provider prompt caching | cache marker on the system prompt for Anthropic only, cached tokens counted | Built. The request shape is checked offline, the saving is not measured. |
| Secrets | environment variables, keys stored as hashes | Built |
| Queues | not built. A semaphore limits concurrency. | Not done |
| Infrastructure as code | not for this project. Another project has Terraform and Helm. | Not done here |
| Public deployment and live demo | | Not done |

## Inference engineering

This agent calls hosted models, so serving internals are not part of it. The related work is in Citera: a QLoRA fine-tune of Llama 3.1 8B (4-bit) served with vLLM on one L4. Not done anywhere in this folder: continuous batching and KV-cache tuning, speculative decoding, tensor parallelism, TensorRT or Triton.
