# Contract review agent

Ask a question about a legal contract and get an answer that quotes it. Every quote is checked against the contract text, and personal data (names, emails, phone numbers) is removed before anything reaches a language model.

It is an LLM agent with tool calling, built on [Knowledge Gateway](docs/GATEWAY.md), a retrieval layer measured on 510 real contracts from the CUAD dataset. It is an API with API keys, rate limits and spending limits, plus a small demo page.

Example (the shape of a request and response):

```
POST /v1/ask   {"question": "How much notice is needed to terminate?", "contract": "<exact title>"}

{ "found": true, "verified": true, "confidence": "high",
  "answer": "Either party may terminate on 30 days' written notice...",
  "citations": [{"source_id": "S1", "contract": "...", "quote": "Either party may terminate this Agreement upon 30 days", "passage": "..."}],
  "usage": {"cost_usd": 0.012, "seconds": 4.1, "model": "gpt-4o", "tool_calls": 1, "fallback_used": false} }
```

## What was measured

100 questions from CUAD, where lawyers marked which clauses match each of 41 categories. Half have a marked clause and half don't. Right means the agent cited a passage covering at least half of a marked span, or said nothing was found when nothing was marked. Temperature 0. Intervals are 95% bootstrap.

| Model | Accuracy | Found the clause when it exists | Said "not found" when absent | Cost per question | Median time |
|---|---|---|---|---|---|
| GPT-4o | 77% [68, 85] | 60% | 94% | $0.0118 | 4.0 s |
| GPT-4o-mini | 65% [56, 74] | 38% | 92% | $0.0016 | 7.5 s |

GPT-4o-mini is 7 times cheaper and 12 points less accurate (paired difference -12% [-21, -4]). It is also slower, because it makes twice as many tool calls (2.6 against 1.2).

- **Tool calls, graded from the trace.** 36 questions (exact title given, only a company name given, contract that does not exist), checked by rules and not by reading the answer. GPT-4o passes 35, with no failed calls and no made-up titles.
- **Where the time goes.** For GPT-4o, 79% of a question is waiting on the model, 16% is retrieval and 5% is everything else. Building a contract's search index adds about 0.6 s.
- **The semantic cache.** With the local MiniLM embeddings, questions that flip the meaning ("allow" and "prohibit", "30 days" and "90 days") are *more* similar to each other than true paraphrases are, so no threshold separates them. OpenAI's embeddings separate them better, and a guard that refuses a match when numbers, negations or parties differ stops the flips. Measured hit rate on paraphrases at the chosen threshold is 23%, with no false hits on 405 different-question pairs and 15 flipped pairs. The sets are small. Details in [docs/AGENT_EVALUATION.md](docs/AGENT_EVALUATION.md).

What the numbers do not say: the CUAD labels are one lawyer's judgement. The agent sometimes finds a related clause nobody marked, which counts as wrong here. 17 of GPT-4o's 20 misses were "said not found", mostly on header facts such as the document name and dates, usually after a single search.

## How it works

1. The question is redacted, so names and emails become tokens like `<PERSON_1>`.
2. The model gets tools: `list_contracts`, `search_contract` and `save_note`, and ends by calling `submit_answer`. Search uses the Knowledge Gateway: BM25 and embeddings fused, then a cross-encoder reranker.
3. Every passage is redacted before the model sees it, and wrapped as data. Text inside a contract is never treated as an instruction.
4. `submit_answer` is checked. Each citation must name a passage the model was shown and quote it word for word. A wrong citation is sent back for repair. If it can't be fixed the agent refuses and does not answer.
5. The result is remembered per conversation. The model can propose notes about a matter, but a note only reaches later prompts after the user approves it, so text planted in a contract cannot write itself into the system prompt. Only redacted text is ever stored.

What keeps it safe to run:

- **Limits per question:** 8 steps, 12 tool calls, 80,000 tokens and 120 seconds. A tool call the model repeats exactly is not run again, and after two repeats it may only submit an answer.
- **Model failures:** the router retries once, then falls back to the next provider. If the first provider's account has no credit or the key is bad, the answer still comes from the fallback.
- **Keys and spending:** API keys are stored as hashes. Each key has a per-minute rate limit and a daily dollar budget, and the whole service has a daily dollar cap.
- **Observability:** OpenTelemetry spans (sizes and timings, never text), Prometheus metrics at `/metrics` for admin keys, and one log line per question with counts only.
- **Cache:** a semantic cache with a guard, off for follow-up questions. For Anthropic the system prompt and tool definitions are also marked for the provider's prompt cache, and cached input tokens are counted.
- **Long conversations:** the last five turns are kept word for word. Older ones are folded into a running summary, one model call every three turns, redacted like everything else.
- **Retries:** send an `Idempotency-Key` header and a repeated request gets the first answer back instead of a second run and a second charge.
- **Drift:** `python -m agent.admin drift-report` compares the last days with the days before (answer rate, found rate, failures, fallbacks, time, cost) and flags real shifts.
- **A/B test:** `AGENT_AB_MODEL` and `AGENT_AB_PERCENT` send a share of new conversations to a second model. `python -m agent.admin ab-report` compares the two on cost, latency, failures and thumbs.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt && python -m spacy download en_core_web_lg
# put CUAD_v1.json (40 MB, CC BY 4.0) in data/raw/

export OPENAI_API_KEY=...            # and ANTHROPIC_API_KEY, GEMINI_API_KEY for the fallbacks
export AGENT_PRIMARY_MODEL=openai/gpt-4o AGENT_FALLBACK_MODELS=openai/gpt-4o-mini
PYTHONPATH=src python -m agent.admin create-key --name me --budget 2 --rpm 20     # prints the key once
PYTHONPATH=src uvicorn agent.api:app_factory --factory --port 8080                  # open http://localhost:8080
```

For a local open-weight model with no API key, start Ollama and set `AGENT_PRIMARY_MODEL=ollama_chat/qwen2.5-coder:7b` and `AGENT_FALLBACK_MODELS=` (empty), with `AGENT_REQUEST_TIMEOUT=900`. The contracts never leave your machine. It works, but slowly: one question took 12 minutes on a CPU, and the 7B model set `found=false` while quoting the clause in its answer, so I have no accuracy row for it. Hosted models are the realistic choice.

Or `docker build -t contract-agent .` and run it with a volume on `/data`. Settings are environment variables, listed in `src/agent/config.py`. The default primary model is `anthropic/claude-sonnet-5-5`.

| Endpoint | |
|---|---|
| `POST /v1/ask` | ask a question (`contract`, `session_id` optional) |
| `POST /v1/ask/stream` | the same as server-sent events: `started`, `step` and `tool` as it works, then `answer` |
| `GET /v1/contracts?contains=` | find an exact title |
| `POST /v1/feedback` | thumbs up or down for a conversation |
| `GET /v1/notes`, `POST /v1/notes/{id}/approve`, `DELETE /v1/notes/{id}` | review the notes the model proposed |
| `GET /v1/usage` | what this key spent today |
| `DELETE /v1/sessions/{id}`, `DELETE /v1/data` | forget a conversation, or everything stored for the key |
| `GET /metrics` | Prometheus (admin key) |

## Tests and evals

`python -m pytest -q` runs about 270 tests with no network and no API keys: the agent loop against a scripted model, the security properties (no raw personal data in anything sent to a model, a made-up quote is rejected, a key over budget is stopped), the stores and the API. I checked that the main ones fail when the code is broken on purpose.

The evals call real models and cost money, so they are scripts in `eval/`: `agent_eval.py` (accuracy), `tool_eval.py` (tool calls), `cache_eval.py` (cache), `latency_eval.py`, `injection_eval.py` and `failure_eval.py`. Each refuses to save a run where the provider failed, so a table of errors can't pass for a result.

## More

- [docs/AGENT_EVALUATION.md](docs/AGENT_EVALUATION.md): methods, results and what went wrong along the way
- [docs/AI_ENGINEERING_MAP.md](docs/AI_ENGINEERING_MAP.md): every AI engineering topic mapped to the project, file and result that uses it, across all my projects, with what is missing
- [docs/CONCEPTS.md](docs/CONCEPTS.md): which AI engineering topics are built, measured or not done, and where
- [docs/INTERVIEW.md](docs/INTERVIEW.md): the usual AI engineering questions, answered from this repo
- [docs/GATEWAY.md](docs/GATEWAY.md): the retrieval layer, redaction and MCP server

MIT licensed. The contracts are from [CUAD](https://www.atticusprojectai.org/cuad/) (CC BY 4.0).
