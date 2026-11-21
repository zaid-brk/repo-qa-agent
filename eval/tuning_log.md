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
