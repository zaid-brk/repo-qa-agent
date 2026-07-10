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

Both runs hit 100% on the first pass, so there is nothing to tune — no ✗ rows to form a
hypothesis from. The benchmark is saturated: every question is answerable from a single
retrieval, which is why naive RAG matches the agent. The honest conclusion is that this
question set doesn't separate the two approaches; making it do so would mean writing harder,
cross-file questions where retrieve-once starts to miss. Recorded here rather than inventing
a tuning gap that didn't happen.
