# Tuning log

Every accuracy claim traces back to a row here. Method: run the 50-question eval,
look at every ✗ in `results.json`, form a hypothesis about *why* it failed, change
**one variable**, re-run, record. Changing two things at once means you can't
attribute the gain — so we never do.

Levers, in the order worth trying:
1. `chunk_size` / `chunk_overlap` (chunk.py) — too big dilutes relevance, too small loses context. **Re-index after changing** (delete `qdrant_storage/`).
2. `top_k` in `search` (store.py) — retrieving too few chunks starves the agent.
3. `separators` (chunk.py) — splitting mid-function makes chunks meaningless.
4. System prompt (agent.py) — e.g. force `search_code` before answering, push it to `read_file` when chunks aren't enough.

| Run | Change (one variable) | Accuracy | Notes |
|-----|----------------------|----------|-------|
| 0 | naive RAG baseline (`--naive`) | 100% (50/50) | retrieve-once, no tools |
| 1 | agent baseline (multi-hop, defaults: 1000/150, top_k=5) | 100% (50/50) | |
| 2 | `MIN_SCORE = 0.32` floor in `search_code` (tools.py) | 100% (50/50) | **reverted** — fixed one bug, caused another |
| 3 | report similarity score to the agent instead of filtering | 100% (50/50) | fixes both |
| 4 | new 67-question set, `--runs 3`, agent | 50/50 clean (partial) | stopped at Q44, out of API credits; see below |
| 4, finished | same set, `--runs 3`, agent and naive, current code | agent 67/67, naive 67/67 | cross-file 12/12 for both; set still too easy |

Runs 0 and 1 hit 100% on the first pass, so there was nothing to tune — no ✗ rows to form a
hypothesis from. The benchmark is saturated: every question is answerable from a single
retrieval, which is why naive RAG matches the agent. The honest conclusion is that this
question set doesn't separate the two approaches; making it do so would mean writing harder,
cross-file questions where retrieve-once starts to miss. Recorded here rather than inventing
a tuning gap that didn't happen.

## Run 2 — a failure the benchmark missed

Found by hand, not by the eval: asking a freshly indexed `itsdangerous` "what ORM ships
inside this project?" sometimes answered "SQLAlchemy". There is no ORM in that library.

Cause: `search_code` returned `top_k` chunks unconditionally, and cosine search always
returns *something*. Measured on `itsdangerous`, absent concepts topped out around 0.28
("ORM" → `.pre-commit-config.yaml` at 0.21; "Django" → `signer.py` at 0.23, which genuinely
contains the string `django-concat`), while real matches started around 0.37. The agent was
being handed plausible-looking noise and occasionally believed it.

First attempt: drop hits below 0.32 and return "no code matches" instead. That fixed the ORM
answer and broke a working one. "What are the main classes here?" started replying that the
codebase contains no classes, because the agent searches for bare syntax like `class `, and
that query scores *lower* than the absent concepts do:

| query | top score | actually in the repo? |
|-------|-----------|----------------------|
| `class ` | 0.179 | yes |
| `main classes` | 0.199 | yes |
| `ORM` | 0.207 | no |
| `SQLAlchemy` | 0.264 | no |
| `Serializer class` | 0.313 | yes |

The premise was wrong. Absolute cosine similarity tracks how a query is *phrased*, not
whether the answer exists, so no cutoff separates "repo has nothing on this" from "that was a
badly worded query". The apparent gap that motivated 0.32 came from comparing one
well-phrased query against a few absent concepts — too small a sample to generalise from.

Run 3, what actually works: return every hit, but prefix the result with the top similarity
and say plainly when it is weak. The agent can then tell strong evidence from weak and
re-query instead of concluding from noise, which was the real failure in both directions.
Verified 3/3 on both questions; the classes question now runs 5–10 hops.

Accuracy stayed 50/50 through all three runs, which is the part worth recording: **the eval
scored a clean 100% while the agent was actively broken in both directions.** Questions 49
and 50 are the same shape of trap and passed throughout. A benchmark at 100% is not evidence
the system is correct, only that the questions are too easy — the strongest argument yet for
writing harder cross-file ones.

## Run 4: harder benchmark, partial

The question set grew from 50 to 67: twelve cross-file questions (each answer traced through
the source by hand), itsdangerous as a fourth repo, and the two hand-found failures from run 2
as regression tests. Each question now runs three times and only passes if all three do.

Before the run, requests, click and flask were re-indexed with the fixed chunking (exact
`read_file`, `.toml`/`.rst` indexed, lockfiles skipped).

The run stopped at question 44 when the OpenAI account ran out of credits. Every call after
that returned `429 insufficient_quota`. Those attempts are excluded. (This section used to say
the harness now stops cleanly on quota and auth errors. It didn't: it only caught
`BadRequestError`, so a quota error crashed the run, and results were written once at the end,
so everything already paid for was lost. Fixed in the finished run below.)

Clean results, 50 questions (requests, click, flask Q35 to Q43): every question passed all
three runs, including 7/7 cross-file. Mean 1.8 tool calls per answer.

A smoke test on itsdangerous before the full run found a new failure: question 62 ("what are
the main classes here and which files define them?") makes the agent call `read_file` on file
after file until the conversation passes gpt-4o-mini's 128k-token limit and the API call
fails. Nothing bounds the agent's context. Tracked as issue #15; the harness counts a context
overflow as a wrong answer.

Not yet known: whether naive RAG fails the cross-file questions. If it doesn't, they are still
too easy and the set needs questions that span more files.

## Run 4, finished

Before spending anything, the harness got the fixes the partial run showed it needed: results
save after every question, `RateLimitError` and `AuthenticationError` stop the run with what's
done kept (selftest covers it), and each answer records its measured dollar cost from token
usage. `--ids`, `--out` and `--max-usd` select questions, pick the output file and cap spend.

A pilot on two cross-file questions (Q58 flask, Q62 itsdangerous) measured about $0.002 per
agent answer and $0.001 per naive answer, judge included. At that rate everything fit well under
budget, so this run covers all 67 questions for both, not just the gap:

- Agent on the 17 questions the partial run never reached (Q44 to Q50, Q58 to Q67).
- Naive RAG on all 67.
- Agent on the first 50 again. The partial run's per-question results were never saved, and
  the agent has changed since (paged `read_file`, step limit, `find_symbol`), so the old 50/50
  wasn't comparable to anything. `eval/results.json` now holds one consistent run of all 67.

Pass = all 3 runs correct:

| | Naive RAG | Agent |
|---|---|---|
| All 67 | 67/67 | 67/67 |
| requests / click / flask / itsdangerous | 21/21, 20/20, 19/19, 7/7 | 21/21, 20/20, 19/19, 7/7 |
| cross_file | 12/12 | 12/12 |
| factual / locational / explanatory | 21/21, 14/14, 12/12 | 21/21, 14/14, 12/12 |
| should_say_idk | 8/8 | 8/8 |
| Flaky | none | none |
| Cites an expected file | 99/162 | 135/162 |
| Cites an expected file, cross-file | 18/36 | 34/36 |
| Mean tool calls per answer | 0 | 2.5 (3.7 cross-file, max 10) |
| Mean seconds per answer | 2.3 | 5.8 (8.8 cross-file) |
| Cost per answer, judge included | $0.0006 | $0.0019 |

Naive RAG passes every cross-file question, all three runs. So the set is still too easy, which
is what this run was meant to find out. Retrieve-once with top_k=5 pulls enough of the right
code for gpt-4o-mini to answer questions I wrote as "needs two or more files". The agent's only
measured edge is citations: it names an expected file in 34 of 36 cross-file answers against
18 for naive, for about three times the cost and two and a half times the latency. The judge
doesn't grade citations, so that edge doesn't move accuracy. Next step for the benchmark is
cross-file questions whose answer needs details from three or more files that never land in
one top-5 retrieval, or grading the citations directly.

Q62 ("what are the main classes?"), the question that overflowed the context in the smoke test,
now takes one `find_symbol` call on all three runs.

One failure, a crash, not a wrong answer: the agent rerun of the first 50 died on Q8 with
`SystemError: AST constructor recursion depth mismatch (before=48, after=43)` from `ast.parse`
inside `find_symbol`. The agent had called `find_symbol` twice in one turn, langgraph runs those
in threads, and CPython 3.11.7 keeps the AST constructor's recursion counter in shared state
(cpython gh-106905), so the two parses raced. The two different depths in the message are the
two threads. `find_symbol` now parses under a lock. The incremental save kept Q1 to Q7; the
rerun resumed at Q8 and finished clean.

Measured cost, from token usage at list prices (gpt-4o-mini $0.15/$0.60, gpt-4o $2.50/$10 per
1M input/output tokens; query embeddings left out, they're fractions of a cent):

| Step | Answers | Cost |
|---|---|---|
| Pilot, agent and naive, 2 questions | 4 | $0.006 |
| Agent, 17 gap questions | 51 | $0.120 |
| Naive, 67 questions | 201 | $0.124 |
| Agent, Q1 to Q7 before the crash | 21 | $0.025 |
| Agent, Q8 to Q57 | 129 | $0.242 |
| Total | 406 | $0.517 |

Plus the partial Q8 attempt that crashed, which wasn't measured (a fraction of a cent). The
judge is 42% of the agent's cost and 69% of naive's.
