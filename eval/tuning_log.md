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
| 2 | `MIN_SCORE = 0.32` floor in `search_code` (tools.py) | 100% (50/50) | fixes a hallucination the 50 questions never caught |

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

Fix: drop hits below 0.32 and return "no code matches" instead. Absent concepts now return
nothing, so the agent searches again rather than reasoning over junk — hop counts on that
question went from 3 to 4–6.

Accuracy stayed 50/50, which is the point worth recording: **the eval did not catch this and
did not detect the fix.** Questions 49 and 50 are the same shape of trap and passed both
before and after. A benchmark scoring 100% is not evidence there are no bugs; it is evidence
the questions are too easy. Another argument for harder cross-file questions.
