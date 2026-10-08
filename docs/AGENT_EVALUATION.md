# How the agent was evaluated

What each eval measures, what it found, and what went wrong along the way. Result files are in `eval/results/`. Every eval that calls a model refuses to save a run where the provider failed.

## The question set

CUAD has 510 contracts, 41 clause categories, and lawyers' marks on which text matches each category. `eval/agent_eval.py` takes a fixed sample of 100 (contract, category) pairs from 20 contracts, half with a marked clause and half without, using CUAD's own question for each ("Highlight the parts of this contract related to X that should be reviewed by a lawyer"). The sample is reproducible from the seed.

Scoring is the same rule the earlier retrieval evals used. If a clause is marked, the agent is right when it says it found something, all its citations verified, and a cited passage covers at least half of a marked span. If nothing is marked, it is right when it says nothing was found. Temperature is 0.

## Accuracy

| Model | Accuracy [95% CI] | Recall (clause exists) | Refusal (clause absent) | Precision of "found" | $/question | Median | p95 | Tool calls |
|---|---|---|---|---|---|---|---|---|
| GPT-4o | 77% [68, 85] | 60% | 94% | 83% | $0.0118 | 4.0 s | 9.8 s | 1.2 |
| GPT-4o-mini | 65% [56, 74] | 38% | 92% | 73% | $0.0016 | 7.5 s | 16.5 s | 2.6 |

Paired on the same 100 questions, GPT-4o-mini minus GPT-4o is -12% [-21, -4]. Cheaper and clearly worse. It is slower too, because it searches and re-searches about twice as often.

**Reading the misses (GPT-4o).** Of 20 missed clauses, 17 were "said not found", 3 were "found something, but not the marked clause", and none failed citation checking. Misses cluster on header facts (Document Name, Agreement Date, Parties, Effective Date and Expiration Date) and on a few narrow categories (Volume Restriction, Minimum Commitment, Warranty Duration), each with only one or two questions in the sample. 76% of questions got exactly one search, so the model often searched once, saw no obvious match and stopped. When it said "found", it was right 30 times out of 36, every time with high confidence, so its stated confidence says nothing about whether it is right.

Not yet tried: telling the model to search again with different words before it says something is absent. That is the obvious fix for the misses above, and it needs a rerun to measure.

## Would a cheap-first cascade save money?

The obvious way to cut cost: run GPT-4o-mini first and call GPT-4o only when the answer looks unsure. `eval/routing_sim.py` replays this from the two saved runs, which used the same 100 questions at temperature 0. It assumes a rerun would give the same answers.

| Setup | Accuracy | $/question | Questions sent to GPT-4o |
|---|---|---|---|
| GPT-4o-mini only | 65% | $0.0016 | 0% |
| GPT-4o only | 77% | $0.0118 | 100% |
| Mini first, GPT-4o unless mini found and verified a clause | 74% | $0.0106 | 74% |
| Mini first, GPT-4o only on errors or unverified answers | 65% | $0.0016 | 0% |

The cascade loses. Half the questions have no marked clause, so a correct "not found" looks the same as a wrong one, and 74% of questions get escalated. The result costs almost as much as GPT-4o and is less accurate. Mini's mistakes are not errors or failed citations, so the second rule never fires. The model's confidence does not separate them either, which is why I did not build a cascade. The router stays a plain fallback chain.

## Where it does worse

`eval/slice_report.py` splits a saved run by clause group, whether a clause is marked, contract length and contract type, and flags a slice only when its interval sits clearly below everything else (the interval widens with the number of slices, so a lucky small slice is not reported). No model is called. Results are in `eval/results/slices_*.json`.

| Slice | GPT-4o | GPT-4o-mini |
|---|---|---|
| Clause terms (78 questions) | 83% [74, 90] | 73% [62, 82] |
| Header facts: name, parties, dates, renewal (22) | 55% [35, 73] | **36% [20, 57], clearly worse** |
| A clause is marked (50) | **60% [46, 72], clearly worse** | **38% [26, 52], clearly worse** |
| No clause marked (50) | 94% [84, 98] | 92% [81, 97] |
| Short, medium, long contracts (35, 35, 30) | 80%, 80%, 70% | 71%, 60%, 63% |

- The main gap is the same for both models: when a clause exists they miss it far more often than they invent one. That is the "said not found" finding again, seen from the other side.
- For the smaller model, header facts are a second weak spot.
- Long contracts score 10 points lower for GPT-4o, but 30 questions cannot separate that from luck, so it is not flagged.
- No contract type is flagged for either model. With five questions per type that is a limit of the data, not a clean bill.
- CUAD says nothing about the people or companies involved, so this is performance slicing and not a fairness audit across demographic groups. That would need data this set does not have.

## Tool calls

`eval/tool_eval.py` grades the tool trace with rules, not the answer text. Three kinds of question, 12 each:

- **exact title given:** it must search that title, never list contracts, and never fail a call
- **only a company name given:** it must list contracts with text from the real title, then search the exact title it got back
- **contract that does not exist:** it must look it up, never search a made-up title, and not claim an answer

GPT-4o passed 35 of 36. The one failure was listing contracts when the exact title had been given. No failed calls, no made-up titles, and an average of 1.1, 2.0 and 1.5 tool calls. These cases are straightforward, so this shows the tool descriptions work. It does not show the agent is good at hard ones.

## The semantic cache

A cache that matches questions by meaning can serve the wrong answer to a question that reads almost the same. `eval/cache_eval.py` measures this with real embeddings on three groups of pairs: 30 paraphrases (should hit), 405 questions about different clauses (must not hit), and 15 flipped pairs, such as "allow" and "prohibit", "30 days" and "90 days", or supplier and customer (must not hit).

| Embedding model | Paraphrases | Different clause | Flipped |
|---|---|---|---|
| MiniLM (local), similarity median | 0.63 | 0.27 | **0.88** |
| text-embedding-3-small, similarity median | 0.67 | 0.32 | **0.87** |

Flipped questions are more similar to each other than true paraphrases are, with both models. No threshold fixes that. Without a guard, anything that catches paraphrases also serves the wrong answer to the flips.

So the cache has a guard. Two questions only match if the words that can flip an answer are the same: negations and prohibitions, numbers, and which party is meant. The guard is blunt and it only costs hits, never wrong answers.

| Setting, with the guard | Paraphrase hit | Different clause hit | Flipped hit |
|---|---|---|---|
| text-embedding-3-small at 0.70 (default) | 23% | 0% | 0% |
| MiniLM at 0.80 | 10% | 0% | 0% |

The 0% on flipped pairs comes with a caveat: I wrote the 15 pairs knowing what the guard looks for. A flip the guard has no word for would get through. The groups are small too, so a 23% hit rate is a rough figure. The nearest false pair is 0.657, so 0.70 leaves a margin of about 0.04.

## Latency

`eval/latency_eval.py` runs questions one at a time and splits the wait. For GPT-4o on 16 questions: 4.3 s on average, of which 3.4 s (79%) is the model, 0.7 s (16%) is tool calls and 0.2 s is everything else. Building a contract's search index adds about 0.6 s the first time. Retrieval is not where the time goes, so the levers are the model and the number of round trips.

**Failover time.** When the first provider fails, the router retries before moving to the next one, and each retry adds about 2 s of backoff. `eval/failover_timing.py` measured how long it takes to get an answer from the second model:

| First model's failure | 2 to 3 retries | 1 retry | no retries |
|---|---|---|---|
| timeout | 7.1 s | 3.3 s | 0.7 s |
| rate limit | 9.7 s | 3.1 s | 0.7 s |
| server error | 5.3 s | 3.1 s | 0.7 s |
| bad key | 1.1 s | 0.9 s | 0.8 s |

One retry is the default, so a one-off blip is still absorbed while a real outage fails over in about 3 s. This was a single run per cell.

## What went wrong

These shaped the design, so they are written down.

- **Claude Sonnet 5.5 refuses forced tool choice.** My first design forced the model to call `submit_answer` on the last step. LiteLLM raised an error for the primary model. The loop now offers only `submit_answer` on the last two steps and never forces it.
- **Models ignore that restriction.** Gemini called a tool that was not on the list it was given, and GPT-4o sent `submit_answer` twice in one reply. The loop refuses tools that are not allowed at that step and uses only the first submit.
- **My schema cleaner deleted an argument.** It stripped every key named "title", including the real `title` parameter of `search_contract`. I caught it by printing the schemas before the first model call.
- **Fallback detection was wrong.** I compared the model name's prefix, and `gpt-4o-mini` starts with `gpt-4o`, so a fallback inside one family looked like the primary. It now compares deployment ids.
- **Provider failures were saved as results.** The Anthropic account ran out of credit during the first full run, and 100 error rows went into the results file. I found them, deleted them, and made every eval refuse to save a run where the provider failed. Later the Gemini and OpenAI balances ran out as well.
- **The first prompt-injection eval was badly designed.** Its control row was inflated by how I picked the questions, and the temperature was left at OpenAI's default of 1. I redesigned it, and I am not reporting numbers from the first version.
- **Newer Claude models reject the temperature parameter.** It is set to 0 for the other providers and omitted for Anthropic.

## A local open-weight model

`qwen2.5-coder:7b` through Ollama runs the whole loop (tools, redaction, citation check), at no cost per token and with the contracts staying on the machine. I ran two live questions on a CPU. The first took 726 s for 4 steps and 8,288 tokens, and it answered with the termination clause, correctly cited and verified, but set `found=false`. The second was stopped to free the machine. That is one sample, so it says the agent runs on a local model and says nothing about accuracy. A full 100 questions would take about a day on this hardware, so the open-weight row is not in the table.

## Not done yet

- Prompt-injection results with the redesigned eval (`eval/injection_eval.py`)
- Claude and Gemini rows in the accuracy comparison. Their API balances ran out during the project.
- The "search again before saying it is absent" experiment
- A saved run of the six failure scenarios in `eval/failure_eval.py`. They passed when I ran them against real providers, but that run's output was overwritten by a later run with no credit, and I did not keep it.
- A public deployment
