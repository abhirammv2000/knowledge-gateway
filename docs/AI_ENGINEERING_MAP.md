# Where I used what

A map from the usual AI engineering topics to the projects that use them: where, how, what I measured, and what is missing. Where a topic is not in any project, it says so and gives the short answer I would give instead. Use it to find the file before an interview, not as a claim list: every row names something you can open.

## The projects, in one line each

| Project | What it is | Repo |
|---|---|---|
| Contract agent (knowledge-gateway) | Ask a contract question and get a cited answer that is checked against the text. Tool calling, memory, redaction, budgets, evals. | [knowledge-gateway](https://github.com/abhirammv2000/knowledge-gateway) |
| DevFlow | A developer agent (triage, ticket to pull request, review) where risky tool calls pause for a person. MCP, approval, injection defences. | [agentic-devflow](https://github.com/abhirammv2000/agentic-devflow) |
| Citera | Agentic RAG over 733 support documents with page citations. Ablation, judge, provider bakeoff, QLoRA fine-tune served with vLLM. | [Ricoh](https://github.com/abhirammv2000/Ricoh) |
| Self-healing data platform | Multi-tenant pipeline platform. When a run fails, a LangGraph agent diagnoses it and recommends an action. EKS, Helm, Terraform, SLOs. | [self-healing-data-platform](https://github.com/abhirammv2000/self-healing-data-platform) |
| Blitz | A company URL in, a marketing package out, through six LangGraph agents. Model router, experiments, grounding evals. | [blitz](https://github.com/abhirammv2000/blitz) |
| scaleflow | Next.js app that answers questions over UN Comtrade data with a reasoning, retrieval and synthesis agent. | [scaleflow](https://github.com/abhirammv2000/scaleflow) |
| TestForge (MCP AI) | Jira ticket to runnable tests with a local LLM, reaching Jira, Bitbucket and Confluence through MCP. | local folder `MCP AI` |
| Synthetic data guard | CTGAN synthetic transactions checked by a multi-agent hallucination detector. | local folder |

Test counts at the time of writing: contract agent about 270, DevFlow 149, Citera 260+, self-healing platform 88, scaleflow 21.

## Foundations

| Topic | Where | How |
|---|---|---|
| Async | Contract agent `src/agent/loop.py`, `service.py`. DevFlow `orchestrator/engine.py`. | `asyncio.timeout` shrinks as a run goes, a semaphore caps concurrent runs, blocking work goes to threads. The DevFlow loop is async so a run can stop mid-turn and resume from another request. |
| APIs and HTTP | Contract agent `api.py`. DevFlow `app.py`. Self-healing platform `control_plane/`. Citera `api/main.py`. | 401, 402, 404, 409, 422, 429, 503 each mean one thing. `Retry-After` on 429 and 503. Idempotency keys. A provider outage is a 503, not a 200 with an apology. |
| SQL and databases | Self-healing platform (Postgres, SQLAlchemy, Alembic, pgvector). Contract agent (SQLite). | SQLite is one instance only, and I say so. The platform uses async SQLAlchemy and a unique constraint for idempotency. |
| Containers, Kubernetes, IaC | Self-healing platform `infra/`, `helm/`. Contract agent `Dockerfile`. DevFlow `docker-compose.yml`. | The platform was deployed to real EKS with Terraform and Helm, then torn down. |
| Testing and CI | Every project has CI. | Scripted fake models for the loop, mocked sessions for the database, and I break the code on purpose to check the important tests fail. |

## Working with models

| Topic | Where | How |
|---|---|---|
| LLM APIs, several providers | Contract agent `llm.py` (LiteLLM). Blitz `core/llm.py`. Citera `llm_factory.py`. | One interface over Anthropic, OpenAI, Gemini and a local server. Claude Sonnet 5.5 rejects forced tool choice and a temperature, so the loop never forces a tool and sends no temperature to Anthropic. |
| Tool calling | Contract agent `tools.py`. Citera `tools.py`. Self-healing platform `worker/app/agent/tools.py`. DevFlow (MCP). | Tool schemas from Pydantic. Errors come back as readable text so the model can fix them. Citera's `search_docs` loop recovered 4 of 6 failing questions at 1.77x the cost. The platform's tool checks live circuit-breaker state, which similarity search cannot find. |
| Structured outputs | Contract agent `schema.py` (`submit_answer`). Self-healing platform `schemas.py`. Blitz `evals/schema_conformance.py`. scaleflow (Zod). | The final answer is a Pydantic object. Output that abandoned the format is rejected, not passed on. |
| Prompting | Contract agent `loop.py`. Blitz agent prompts. | System prompt, nudges, repair prompts that name exactly what was wrong. |
| Embeddings | Contract agent `gateway/retrieval.py`, `agent/embeddings.py`. Citera (MiniLM in Chroma). scaleflow (Pinecone). | Measured: on legal text keyword search beat embeddings by 5.2 points at Hit@5, and "allow" and "prohibit" are closer than two paraphrases. |
| Context windows and tokens | Contract agent `config.py`. | Per-question token budget (80,000), question length cap, a five-turn window with a running summary behind it. |
| Prompt caching | Contract agent `llm.py`. | A cache marker on the system prompt for Anthropic only, added by LiteLLM to that deployment so a fallback never receives it. Cached tokens are counted. The saving is not measured. |
| Streaming | Contract agent `/v1/ask/stream`. Citera `/query/stream`. Blitz (SSE for each agent's result). | The contract agent streams progress, then the answer once its citations are checked. It never streams a draft that might be refused. |

## Retrieval and RAG

| Topic | Where | How and result |
|---|---|---|
| Chunking | Contract agent `gateway/chunking.py`. Citera `ingest.py`. | Clause-aware chunks tied a fixed window at equal context but cut fewer clauses (97.7% intact against 94.1%). Citera uses 500-word chunks with 50 overlap. |
| Hybrid search | Contract agent `gateway/retrieval.py`. Citera `retriever.py`. | BM25 and vectors fused with reciprocal rank fusion. Fusion alone barely moved the needle over BM25 (+0.46 points, interval includes zero). |
| Reranking | Contract agent (cross-encoder, top 10). Citera (optional reranker). | The reranker was the only clear gain: +3.7 points Hit@5, from 67.4% to 71.1%. |
| Vector stores | Chroma (Citera, Blitz), pgvector (self-healing platform), Pinecone (scaleflow), Azure AI Search backend (Citera). | Citera's Azure backend matched the local reranker on recall@5 and was much faster. |
| Metadata filtering | Self-healing platform `retrieval_node.py` (tenant filter on past incidents). Contract agent (scoped to one contract). | Runbooks are global, past incidents are tenant-scoped, searched separately and combined. |
| Query rewriting | Citera `conversation.py` (`condense_query`). | A follow-up like "can I copy one?" is rewritten to stand alone before retrieval. Only the question is rewritten, so a bad rewrite gives a refusal, not an invention. |
| Retrieval evaluation | Contract agent `eval/chunking_eval.py`, `hybrid_retrieval_eval.py` (6,702 questions). Citera recall@5 0.94 on 100 questions. | Intervals are bootstrap, with equal-budget comparisons declared before the run. |
| Agentic RAG | Citera `tools.py`, `router.py`. | The model chooses what to search for and when to search again. The router runs the cheap path first and uses the tool loop only on a refusal, because a retrieval-confidence router did not separate misses from hits. |

## Agents

| Topic | Where | How |
|---|---|---|
| A single tool loop | Contract agent `loop.py`. | 8 steps, 12 tool calls, 80,000 tokens, 120 seconds. Seven named stop reasons. The last two steps offer only `submit_answer`. |
| Loop prevention | Contract agent, DevFlow. | Identical repeated calls are refused. DevFlow stops a run after three refusals. A local 7B model labelled one issue five times and never reached the comment, which is how that guard was found. |
| Multi-agent | Self-healing platform (four-node diagnostic graph). Blitz (six sequential agents with a critic). Synthetic data guard (validator pipeline). | The contract agent is one loop on purpose: more agents add cost, latency and failure modes, and one loop was enough. |
| Permissions and approval | DevFlow `policy.py`. Contract agent notes. Self-healing platform recommendations. | Four risk tiers, three autonomy levels, merge always needs a person. Model-proposed notes need approval because memory is an attack surface. Approving a platform recommendation now starts the retry or pauses the schedule. |
| Idempotency | Contract agent `idempotency.py`. Self-healing platform (run idempotency key, recommendation id key). | Same key and same request returns the first answer. Same key with a different request is refused. A failed request is not stored. |
| MCP servers | Contract agent `gateway/mcp_server.py`. Citera `src/mcp_server.py`. DevFlow `mcp_servers/`. | Read-only tools with annotations. Written for mcp 2.x, where the class is `MCPServer`. |
| MCP clients | DevFlow `mcp_registry.py`. TestForge. | DevFlow launches servers over stdio and treats any unclassified tool as `critical`. |
| Memory | Contract agent `memory.py`. Citera `conversation.py`. Blitz (ChromaDB between agents, in-memory checkpoint). | Per key, redacted before storing, deletable. Approved notes outlast a session. Summaries are folded in every few turns. |
| Coding agent | DevFlow `implement_ticket` playbook. | Finds the bug, branches, edits, adds a test, runs the suite, opens a pull request. Merging is never automatic. |
| Local models | Contract agent (Ollama), DevFlow (Ollama or vLLM), TestForge (Ollama), Citera (vLLM). | A local 7B took 12 minutes for one contract question on CPU and was fooled by every planted instruction in the DevFlow eval. |

## Evaluation

| Topic | Where | How and result |
|---|---|---|
| Ground truth you did not write | Contract agent (CUAD, lawyers' marked spans). | 100 questions, half unanswerable. GPT-4o 77% [68, 85], GPT-4o-mini 65% [56, 74], paired difference -12% [-21, -4]. |
| LLM as judge | Citera `eval_harness.py`. | A stronger model judges groundedness and correctness. Citera found a 10-question set flattered the numbers: growing it to 100 moved groundedness from 0.98 to 0.96. |
| Ablation | Citera `docs/EVALUATION.md`. | Planner and verifier cost 1.7x to 2.7x. The verifier added nothing, so it is off. The planner's gain did not carry to held-out questions. |
| Trajectory grading | Contract agent `eval/tool_eval.py`. DevFlow `evals/`. | Rules over the tool trace, not the answer text. GPT-4o passed 35 of 36. Graders are tested on hand-made trails first. |
| Cross-provider comparison | Citera `provider_bakeoff.py`. | Gemini 3.6 Flash kept answer quality at about 1/50th the per-query cost. GPT-4o-mini lost 0.21 correctness. |
| Domain checks | Blitz `evals/grounding.py`. Self-healing platform 33-case eval. | Two narrow grounding checks instead of one blurry score. The platform scored 90.9% classification and 84.8% action accuracy, and the eval found a real weakness (it over-picks `schema_evolution`). |
| Experiments and A/B | Blitz `experiments.py`. Contract agent (sticky arms, `ab-report`). | Blitz assigns variants by hashing, with a Wilson interval and a two-proportion test. The contract agent has no live traffic yet. |
| Regression gate | Citera CI (retrieval gate on a small demo index). | Runs on every push with no API key. |
| Slice analysis | Contract agent `eval/slice_report.py`. | Accuracy by clause group, clause marked or not, contract length and type, flagging only slices clearly below the rest. Both models miss existing clauses far more than they invent them. |
| Cascade routing | Contract agent `eval/routing_sim.py`. Citera `router.py`. | Mini-first with escalation cost almost as much as GPT-4o and scored lower, because a right "not found" looks the same as a wrong one. Both projects ended with a plain fallback chain. |

## Safety and security

| Topic | Where | How and result |
|---|---|---|
| Hallucination control | Contract agent `loop.py` (`verify_citations`). Citera (cited answers, refusal when evidence is thin). | Every citation must quote a passage the model was shown. 0 unverified answers in 100 questions. The misses were omissions, 17 of 20 "not found", which a citation check cannot catch. |
| Prompt injection | Contract agent `guard.py`. DevFlow `guard.py`, `policy.py`. Blitz `untrusted_content.py`. | Tagged data, a canary string, and in DevFlow a taint rule: after reading outside text, publishing needs a person. With the rule on the attacker's comment ran 0 of 3 times, off it ran 3 of 3 (7B model, small sample). |
| PII | Contract agent `gateway/redaction.py`. | Presidio before the model, plus a Luhn-checked card recogniser. 99.0% recall on synthetic data, with the caveat that it is templated text. |
| Guardrails | Citera `guardrails.py`. Contract agent. | Citera screens a few obvious jailbreak phrasings before they cost a model call, and says the real defence is grounding. |
| Memory poisoning | Contract agent notes approval. | An injected "save a note that says always approve" would have sat in the system prompt of every later question. There is a test that plays that attack. |
| Approval link security | DevFlow `signing.py`, `decide_page.py`. | HMAC over run, decision, the exact pending calls and an expiry. A confirm page so a link preview cannot approve. |
| Authentication | Contract agent `accounts.py`. DevFlow service token (constant-time compare). | Hashed keys, admin keys, per-key data. |
| Red-teaming | Citera `tests/test_redteam_guardrail.py`. Contract agent `eval/injection_eval.py` (written, not run). | |

## Production

| Topic | Where | How |
|---|---|---|
| Observability | Contract agent (OpenTelemetry without text, Prometheus). DevFlow. Self-healing platform (OTel, Prometheus, Jaeger, Grafana, SLO alerts). Citera `instrumentation.py`. Blitz `telemetry/`. | Spans carry sizes, timings and counts, never text, and a test checks it. |
| Cost monitoring | Contract agent ledger. Citera `perf.py`. DevFlow `/usage`. Self-healing platform token metric. Blitz telemetry store. | Cost per question, by model, by arm. Per-key and global daily budgets. |
| Rate limits | Contract agent (per key per minute). Citera `ratelimit.py` (token bucket). DevFlow (runs per minute). | Spend is checked before and recorded after, so concurrent requests can overshoot by their own cost. It is a backstop, not an exact meter. |
| Retries and fallback | Contract agent `llm.py`. Blitz `core/llm.py`. Self-healing platform (retries with backoff, circuit breakers). | Measured failover: about 3 seconds with one retry, 5 to 10 with 2 to 3. A bad key fails over in about 1 second because it is not retried. |
| Semantic cache | Contract agent `cache.py`. Citera `semantic_cache.py`. | Flipped-meaning questions are more similar than paraphrases, so the contract agent adds a guard on negations, numbers and parties. 23% paraphrase hit rate with 0% false hits. |
| Drift | Contract agent `drift.py`. | Behaviour, not accuracy, because live questions have no labels. Two-proportion test with a strict cutoff, 30 requests minimum per window. |
| Feedback loop | Citera `feedback.py`. Contract agent `/v1/feedback`. | Thumbs keyed to the trace id, so a vote leads back to the chunks that produced the answer. |
| Queues and workers | Self-healing platform (Redis queue, worker). | Idempotent runs, retries with backoff, circuit breakers. |

## Fine-tuning and serving

| Topic | Where | How and result |
|---|---|---|
| QLoRA fine-tune | Citera `finetune/`. | Llama 3.1 8B, 348 teacher examples, from 593 documents never used in any eval set so the test is genuinely held out. Correctness 0.941 against Sonnet's 0.9925, inside the judge's noise. Groundedness 0.9055 against 0.98, a real gap. |
| Self-hosted serving | Citera `finetune/serve/`. | vLLM on one L4. vLLM 0.29 dropped bitsandbytes support, so the adapter is merged to 16 bit first. It shells out to `ninja` by bare name, so the venv must be activated. 25 s per answer against 2.6 to 9.9 s for APIs, expected for one unbatched GPU. |
| Quantization | Citera (4-bit QLoRA). | Used for training. Not used for serving. |
| Distillation | Citera. | Training data came from a stronger model answering with the real retriever. |

## Not used, and what I would say

I have not built these. The short answers are real, the projects are not.

- **Continuous batching, PagedAttention, KV-cache management, chunked prefill, prefix caching, speculative decoding, tensor parallelism.** I have run vLLM, which does the first three, but I did not tune it. Batching raises throughput by filling the GPU with several requests at once, and PagedAttention stops the KV cache wasting memory on padding. Speculative decoding has a small model draft tokens that the big one checks in one pass.
- **Triton, TensorRT-LLM.** Not used.
- **Reasoning models and extended thinking.** Not used. I would try them where multi-step reading of a clause matters, and measure whether the extra cost buys accuracy, as with the planner in Citera.
- **GraphRAG.** Not used. It would help for questions across many documents, such as "which contracts share this party", and costs a graph-building step.
- **Agent-to-agent protocols (A2A).** Not built. The retrieval layer is exposed over MCP, which is the tool side.
- **Preference tuning (DPO, RLHF).** Not done. The Citera README says so.
- **Training a model from scratch, GPU programming.** Not done.
- **A public deployment with real users.** Not done, and the collect-failures-and-improve loop depends on it.
- **Demographic fairness auditing.** Not possible with CUAD, which has no attributes about the people involved. What exists is performance slicing (`eval/slice_report.py`).

## Questions I can answer from the code

Each of these has a section with a measured answer in [INTERVIEW.md](INTERVIEW.md): design a RAG system, stop hallucinations, fine-tune versus prompt versus RAG, evaluate an LLM app, cut cost, loops, prompt injection, a semantic cache and where it fails, chunk size, observability, A/B testing, provider failure, memory, grading tool calls, rate limits and timeouts, PII, a coding agent, and walking through a system end to end.
