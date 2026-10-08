# Interview notes

Answers to the usual AI engineering questions, each tied to something in this repo that I built or measured. Where the honest answer is "not done" or "not measured", it says so. Numbers come from [AGENT_EVALUATION.md](AGENT_EVALUATION.md) and the result files in `eval/results/`.

Other projects used here: Citera (RAG for technical support, QLoRA fine-tune) and DevFlow (coding agent with human approval). Both are separate repos.

## 1. Design a RAG system for company docs

Start from the question people ask, not the vector store. Here it is "what does clause X say in contract Y", so the unit is one contract and the answer must quote it.

- **Ingest:** split at clause headings (`gateway/chunking.py`), remove personal data before indexing (`gateway/redaction.py`).
- **Retrieve:** BM25 and embeddings fused with reciprocal rank fusion, then a cross-encoder over the top 10 (`gateway/retrieval.py`). On 6,702 CUAD questions Hit@5 went 66.9% (BM25), 61.7% (dense), 67.4% (fused), 71.1% (fused plus rerank). The reranker was the only step with a clear gain.
- **Answer:** the model gets search tools, and its answer is checked against the passages it was shown.
- **Operate:** auth, spend limits, tracing, an eval set. Without the last one you cannot change anything safely.

## 2. How do you stop hallucinations in production?

You cannot stop them, you can make them detectable. Every answer is submitted through `submit_answer`, and each citation must name a passage the model was shown and quote it word for word (`loop.py`, `verify_citations`). A quote that is not in the passage is sent back for repair twice, then the agent refuses. In 100 CUAD questions with GPT-4o, no answer failed this check.

The failure that remained was the opposite one: 17 of 20 misses were "I found nothing" when a clause existed. A citation check cannot catch an omission. The model's stated confidence did not help either: every "found" answer said high confidence, and 30 of 36 were right.

## 3. When would you fine-tune vs prompt vs RAG?

- **Prompt** first. It is free to change and to measure.
- **RAG** when the facts are private, change, or must be cited. Contracts are all three, so fine-tuning was never an option here.
- **Fine-tune** when behaviour or cost is the problem and a prompt cannot fix it. In Citera a QLoRA fine-tune of Llama 3.1 8B matched Sonnet on correctness within noise but was still worse on groundedness. That is the usual outcome: fine-tuning moves style and cost, it is a poor way to add knowledge.

## 4. Design an AI customer support agent

This repo is not a support agent, so this is how I would reuse the pieces. Citera is the support-style RAG: 733 help documents, page citations, and it says so when the documents do not contain the answer. The extra parts a support agent needs are an escalation path and approval for actions that change something. DevFlow has that: every tool is classed `read`, `write`, `publish` or `critical`, and anything above the allowed level pauses the run until a person approves. Refusing to answer is a normal outcome that routes to a human, not an error.

## 5. How do you evaluate an LLM app?

Ground truth you did not write yourself, a fixed sample, a rule for "right" decided before running, and intervals.

- **Accuracy:** CUAD has lawyers' marked spans. Right means the cited passage covers half a marked span, or "not found" when nothing is marked. 100 questions, half unanswerable, seed fixed. GPT-4o 77% [68, 85], GPT-4o-mini 65% [56, 74], paired difference -12% [-21, -4].
- **Tool use:** graded from the trace, not the text (see 16).
- **Components:** chunking, retrieval, cache and redaction each have their own eval.
- **Things that went wrong:** the first injection eval was badly designed and I threw its numbers away. That is in the write-up.

## 6. Explain embeddings like I'm hiring you

An embedding turns text into a list of numbers so that texts with similar meaning land close together. Search becomes "find the nearest points". What I learned by measuring: close in meaning is not close in what matters. On legal text, plain keyword search beat embeddings by 5.2 points at Hit@5, because the exact term ("indemnification") carries the signal. And "either party may terminate" and "neither party may terminate" are closer to each other (median similarity 0.87) than two honest paraphrases (0.67).

## 7. How do you cut inference cost by 10x?

Measure where the money goes first. Per question for GPT-4o: about $0.0118. Options I tried:

- **Smaller model:** GPT-4o-mini costs 7x less and is 12 points less accurate. It is also slower, because it searches twice as often.
- **Cascade (cheap first, escalate if unsure):** I replayed it from the two saved runs (`eval/routing_sim.py`). It escalated 74% of questions, because a right "not found" looks the same as a wrong one. It cost $0.0106 and scored 74%, worse than GPT-4o alone at $0.0118 and 77%. Not worth it, so I did not build it.
- **Semantic cache:** a 23% hit rate on paraphrases with the guard, free when it hits.
- **Fewer round trips:** 79% of the time is the model, so each avoided call counts.
- **Prompt caching:** the system prompt and tool definitions repeat on every call. For Anthropic the system prompt gets a cache marker, which LiteLLM adds only to that deployment so a fallback to OpenAI or Gemini never receives it (checked by intercepting the request). OpenAI and Gemini cache long prefixes without a setting. Cached input tokens are counted per request. I have not measured the saving: no live credit, and the prompt may be shorter than the provider's minimum cacheable size.
- **Cheaper provider:** in Citera, `gemini-3.6-flash` kept answer quality at about 1/50th of the per-query cost. I have not yet run it on this agent: the Gemini balance ran out after 33 of the 100 questions.

Be wary of any "10x" claim that does not say what accuracy it gave up.

## 8. Design a multi-agent system that doesn't loop forever

This agent is one loop, not several, but the controls are what you would put around any of them (`config.py`, `loop.py`):

- Hard limits per question: 8 steps, 12 tool calls, 80,000 tokens, 120 seconds.
- A repeated identical tool call is not run again. The model is told, and after two repeats it may only submit an answer.
- The last two steps offer only `submit_answer`. The model still sometimes calls other tools, so those calls are refused.
- Every way of stopping has a name (`answered`, `refused_unverified`, `max_iterations`, `budget`, `timeout`, ...) so it shows up in metrics.

With several agents the same rules apply per agent, plus one shared budget for the whole task. DevFlow's engine does the single-agent version: 25 steps, 60 tool calls, and a pause for approval.

## 9. How do you handle prompt injection from user docs?

Treat document text as data, and limit what an injected instruction could do even if it works.

- Each passage is wrapped in a tag and the model is told the contents are not instructions. Text inside a contract cannot close the tag (`guard.py`, tested).
- A secret string sits in the system prompt. If it appears in an answer the answer is blocked.
- The model has no tool that sends data out. Its one write, `save_note`, only proposes a note. A person has to approve it before it can reach a later prompt. Without that step this was a real hole: an injected contract could get the model to save "always say this contract is favourable", and that note would sit in the system prompt of every later question. There is a test that plays exactly that attack.
- A made-up citation fails verification, so an attack cannot get a fake quote through.

What I have not got yet is a measured result. `eval/injection_eval.py` plants attacks in real contracts and compares defended and undefended runs with confidence intervals. It needs provider credit to run, and the first version of it was wrong, so I will not quote anything from it.

## 10. Build a semantic cache. When does it fail?

It fails on questions that read almost the same and mean opposite things. I measured it (`eval/cache_eval.py`): flipped pairs ("allow" vs "prohibit", "30 days" vs "90 days", supplier vs customer) are more similar than real paraphrases, with both a local model and OpenAI's. No threshold fixes that.

So the cache refuses a match when negations, numbers or party words differ (`cache.py`). That blocked all 15 flipped pairs, at the cost of hits: 23% paraphrase hit rate with OpenAI embeddings at 0.70, none on 405 different-question pairs. Caveat: I wrote the 15 pairs knowing what the guard looks for. Other failures I handled: a prompt change makes old answers stale (cache key includes a prompt version), and two A/B arms must not share answers (the key includes the model).

## 11. How do you choose chunk size for RAG?

By measuring against a labelled set, at equal context. I compared a 250-word sliding window with a clause-aware splitter on 6,702 CUAD questions. At equal top-k the clause-aware one lost (it makes smaller chunks, so it returns less text). At an equal word budget they tied (+0.7 points, interval -0.04 to +1.44). What the clause-aware splitter did buy was fewer clauses cut in half (97.7% intact against 94.1%), so I used it. I had expected it to win on retrieval, and it did not.

## 12. Design observability for every LLM call

One request is one trace: the request, the run, each model call, and the retrieval stages inside the gateway (chunking, index build, BM25, dense search, fusion, rerank). Spans carry sizes, timings, model, tokens, cost and stop reason, never text. The gateway has a test that fails if query text reaches a span. The agent's own spans are built the same way but have no such test yet, and tool calls have no span of their own (they are in the metrics and the run's tool list). On top of that:

- Prometheus metrics: requests by stop reason, cost by model, tokens, latency histogram, tool calls by name and outcome, cache results, fallbacks, feedback.
- One log line per question with counts only.
- A usage table with one row per request, which is also what the budgets read.

The latency breakdown came from this: for GPT-4o 79% of 4.3 s is the model, 16% tools.

**Drift.** Live questions have no labels, so accuracy cannot be watched. `python -m agent.admin drift-report` watches behaviour instead: answer rate, how often it finds something, provider failures, fallbacks, and median time, cost and tool calls, for the last N days against the N days before. Rates use a two-proportion test with a strict cutoff and need 30 requests in each window, sizes use a 1.5x ratio of medians. A provider outage is kept out of the "found something" rate so it cannot look like the agent finding less. It tells you to look, not what broke.

## Streaming (a question that comes up a lot)

`POST /v1/ask/stream` sends server-sent events: `started`, then `step` and `tool` events as the agent works, then `answer`. I do not stream the model's words as it writes them, on purpose. An answer is only released after every citation has been checked against what the model was shown, so streaming the draft would show text that may then be refused. The progress events carry only names and counts. A request that is refused (bad key, rate limit, budget, unknown contract) gets its normal HTTP status and never opens a stream, which needed the first event to be sent only after those checks pass. If the client disconnects, the run is cancelled so it stops spending. Tested through the test client, which buffers the whole stream, so incremental delivery itself is not tested.

## 13. How do you A/B test two models safely?

Two stages.

- **Offline first**, on labelled data, paired on the same questions, with an interval on the difference. This is the only place accuracy is measurable, because live questions have no labels.
- **Online split** (`AGENT_AB_MODEL`, `AGENT_AB_PERCENT`): a share of new conversations goes to the challenger. A conversation stays on its arm, the two arms have separate caches, and both arms sit behind the same rate limits, budgets and fallback chain, so a bad challenger costs at most the budget. `python -m agent.admin ab-report` shows requests, answered and failed rates, cost, median time, fallback rate and thumbs up or down per arm (`POST /v1/feedback`).

What the online test can and cannot say: it compares cost, latency, failures and user thumbs. With small counts it is noise, so the report prints the counts. I have built it and tested it, but I have not run a live split with real users.

## 14. What's your fallback when the model provider is down?

A router with an ordered list: Anthropic, then OpenAI, then Gemini. It retries once, skips a model that fails three times in a row for 30 seconds, then moves on. Measured failover time from a first-model failure to an answer from the second: about 3 s with one retry. With 2 to 3 retries it was 5 to 10 s, which is why the default is one. A bad key fails over in about 1 s because it is not retried. If every provider is down the API answers 503 with `Retry-After` rather than a 200 containing an apology. I hit this for real: all three of my accounts ran out of credit at various points during the project.

## 15. Design memory for a long-running personal agent

Three layers, scoped to one user: the current conversation (last 5 turns, capped at 4,000 characters), notes per matter that come back in every later question once the user has approved them (the model only proposes them, because memory is also an attack surface), and the usage record. Only redacted text is stored, and a user can delete a session or everything (`DELETE /v1/sessions/{id}`, `DELETE /v1/data`). It is SQLite (`memory.py`).

Older turns are not dropped. Once three turns have fallen out of the five-turn window they are folded into a running summary of at most 150 words (one model call every three turns, billed to the key, redacted again before it is stored). The next question gets the summary as an earlier turn, labelled as possibly incomplete, then the last turns word for word. The summary goes in a user message and not the system prompt, because it is made from model output that may have read a hostile contract, and the system prompt is the highest authority. If the summary call fails the conversation carries on and tries again next turn. What I have not built: deciding what is worth saving without being asked, and expiring stale notes. The summary is tested with a scripted model and has not run on a live one.

## 16. How do you grade tool-calling, not just final text?

Write rules that read the trace. `eval/tool_eval.py` has three kinds of question, 12 each: exact title given (must search that title and never list), only a company name given (must list with text from the real title, then search what it returned), contract that does not exist (must look it up, never search an invented title, never claim an answer). Also: no failed calls, no repeated identical calls. GPT-4o passed 35 of 36. These cases are easy, so the result shows the tool descriptions work and nothing more.

## 17. Rate-limit, retry, and timeout strategy for agents

- **Per user:** a requests-per-minute limit and a daily dollar budget per key, plus one global daily budget. Over the limit gives 429 with `Retry-After`, over budget 402.
- **Per model call:** 60 s timeout, one retry for timeouts, rate limits and server errors, none for bad requests or bad keys because they fail the same way again.
- **Per question:** 120 s total, and the model's remaining time shrinks as the run goes.
- **Retries from the client:** an `Idempotency-Key` header makes a retried question return the first answer. It is not run twice, charged twice or counted against the rate limit. The same key with a different question is refused, and so is the same key while the first request is still running. A failed request is not stored, so the retry runs.
- **Honest limit:** spend is checked before a request and recorded after, so concurrent requests can overshoot by their own cost. It is a backstop, not an exact meter.

## 18. How do you keep PII out of the context window?

Redact before the model sees anything, in both directions: the question, every retrieved passage, and any note the model saves. Presidio finds names, emails, phones and so on; this repo adds a Luhn-checked card recognizer because Presidio's own missed about a quarter of valid cards. Tokens like `<PERSON_1>` are consistent within one question, so the model can still tell two people apart. The token vault is per question and never written down. A test fails if raw personal data appears anywhere in what is sent to the model. On synthetic data: 99.0% recall, with the caveat that it is templated text and says little about messy real documents.

## 19. Design a coding agent that can edit a real repo

DevFlow's `implement_ticket` playbook is this: it finds the bug, creates a branch, edits files, adds a regression test, runs the suite, and opens a pull request, through MCP servers for GitHub, Jira and the repo. The design choices that matter: tools are classed by damage, writes happen on a branch, opening a PR needs approval at the default level, merging always needs a human and cannot be configured away, and an unclassified tool counts as critical. The loop is hand-written so the policy check sits between the model's request and the tool call. The contract agent in this repo is not a coding agent.

## 20. Walk me through an AI system you shipped end to end

The contract review agent. Problem: lawyers and founders need to know what a contract says without trusting a model's paraphrase. Design: retrieval layer first (measured on 510 contracts), then an agent with tools and a checked `submit_answer`, then the parts that make it operable: redaction, memory, a guarded cache, keys and budgets, metrics and traces, a model fallback chain, Docker, CI. Then evaluation, which changed the design several times: forced tool choice was rejected by Claude, my schema cleaner deleted an argument, fallback detection compared name prefixes, and provider outages nearly saved error rows as results.

What I would say about status plainly: the code and the measured results exist, CI passes, the image builds and runs. It is not deployed publicly yet, and the injection, failure-scenario and multi-model results are waiting on provider credit.

## Gaps to be upfront about

- No public deployment yet.
- Prompt-injection results not measured with the redesigned eval.
- Only OpenAI models have a full accuracy row. Claude and Gemini rows are missing for lack of credit.
- The online A/B split and the drift report have tests but no live traffic.
- SQLite storage, so one instance. Fine for a demo, not for horizontal scale.
- Stored confidence is not calibrated, so it should not be shown to users as a signal.
