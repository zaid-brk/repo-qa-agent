---
title: Repo Q&A Agent
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# GitHub Repo Q&A Agent (RAG)

I built a web app where you paste a GitHub repo URL, it indexes the code, and then you can
ask questions like "where is authentication handled?" or "what does raise_for_status do?"
and get answers grounded in the actual source, citing the file and line. Under the hood it's
retrieval-augmented generation (RAG) upgraded to an agent that can search, list files, and
read whole files across multiple hops, plus a 67-question evaluation harness that measures
how accurate it actually is.

**Try it:** https://repo-qa-agent-web.onrender.com (paste your OpenAI key, index a GitHub repo, ask about it)

**Live API:** https://repo-qa-agent.onrender.com/docs (FastAPI backend on Render, Qdrant Cloud as the vector store). Render's free tier sleeps when idle, so the first request can take ~50s to wake up. It's bring-your-own-key: `/index` and `/ask` need your OpenAI key in an `X-OpenAI-Key` header (the web app has a field for it), and the server never uses a key of its own.

## Demo

[Watch the demo on Loom](https://www.loom.com/share/12029311091b493395c86c3842e175d7)

(It's worth noting that the codebase has been edited and improved since this demo was filmed - new demo will be out soon)

## How it works

The LLM has never seen your repo, so the pipeline gives it the right pieces at the right time:

```
GitHub URL ──> ingest (shallow clone) ──> chunk (code-aware splits)
                                              │
                                              v
                              embed (text-embedding-3-small)
                                              │
                                              v
                                   Qdrant (vector store)
                                              ^
                search_code / find_symbol / list_files / read_file
                                              │
question ──────────────────────────> ReAct agent (gpt-4o-mini) ──> answer + path:line links
```

1. **Ingest:** only `https://github.com/<owner>/<repo>` URLs are accepted, since the URL goes
   to `git clone` on a public server. Shallow-clone it and keep code, config and docs files,
   skipping binaries, `node_modules`, lockfiles and anything over 200KB (generated files cost
   embedding money and drown search results in noise).
2. **Chunk:** split files into ~1000-char pieces with 150 overlap. The splitter prefers
   `\nclass ` and `\ndef ` boundaries so whole functions stay together, a real improvement
   over default splitting for code. Each chunk keeps its offset in the file, so files can be
   rebuilt exactly and every hit knows its line numbers.
3. **Embed + store:** each chunk becomes a 1536-dim vector in Qdrant. Cosine similarity
   makes "how are retries handled?" land near retry code even with zero shared keywords.
4. **Agent:** instead of retrieve-once-and-hope, a ReAct agent decides what to do: search,
   see it needs the full file, read it, search again. That's multi-hop retrieval, and it's
   what handles questions whose answers span several files. For exact names ("where is
   `Session` defined?") it has `find_symbol`, which parses the indexed Python files with
   `ast` and returns each definition's file and line, since embeddings are bad at syntax.
   Answers cite `path:line`, and the API links each citation to those lines on GitHub at
   the exact commit that was indexed. Two guard rails keep it inside the model's context:
   `read_file` returns long files in 20k-character pages, and the agent stops after 25
   steps and says it ran out instead of crashing.

Qdrant runs in local file mode (`qdrant_storage/`), so there's no Docker or server to set
up while developing. For deployment it swaps to Qdrant Cloud with one line. Each repo gets
its own collection so several repos can be indexed and evaluated side by side, and `/ask`
takes the repo per request, so two visitors asking about different repos don't interfere.

## The eval harness (the part I care about most)

Anyone can wire up an agent. Knowing whether it's *right* is the part I care about, so I wrote
67 questions with known correct answers across four real codebases, requests, click, flask
and itsdangerous, by reading their source and checking every answer. Five question types:

- **Factual:** "what is the default number of retries?"
- **Locational:** "which file defines the Session class?"
- **Explanatory:** "how are Flask's client-side sessions secured?"
- **Cross-file:** "trace a timeout passed to requests.get() down to urllib3" (needs two or more files)
- **Should-say-I-don't-know:** "what ORM ships inside Flask?" (there isn't one; this catches hallucination)

Explanatory answers can't be string-matched, so a stronger model (gpt-4o) grades each answer
against the ground truth: the LLM-as-judge pattern. The judge can be wrong too, which is a
known limitation of the pattern; using a judge stronger than the agent keeps it reliable enough.

`eval/run_eval.py` runs the whole thing and reports accuracy overall, per repo and per question
type. `--runs 3` asks every question three times and only counts it if all three pass, since a
wrong answer that shows up one time in three is still a bug. It also checks whether each answer
cites the right file, and records tool calls, seconds and measured dollar cost per answer.
Results save after every question, and a quota or auth error stops the run instead of being
scored as a wrong answer. `--ids`, `--out` and `--max-usd` run a subset, write it to its own
file and stop at a spend cap.
`--naive` runs the same questions through plain single-shot RAG, so the agent's multi-hop
advantage is measured, not assumed. Tuning happens one variable at a time and every run is
recorded in `eval/tuning_log.md`. That log is the evidence behind any accuracy number.

## Results

### Original 50 questions

Run on OpenAI (`gpt-4o-mini` agent, `text-embedding-3-small`, `gpt-4o` judge), all 50
questions across `requests`, `click`, and `flask`:

| Setup | Accuracy (50 questions) |
|-------|------------------------|
| Naive RAG (retrieve once) | 100% (50/50) |
| Agent (multi-hop) | 100% (50/50) |

Both approaches answer every question, including the should-say-I-don't-know traps (the agent
correctly replies "I don't know" to "what ORM ships inside Flask?" and "what is the production
database password?" instead of hallucinating). Per repo: requests 17/17, click 17/17, flask 16/16.

**Honest read of this:** the benchmark is saturated. These questions are answerable from a
single retrieval, so the multi-hop agent doesn't beat naive RAG here; both hit the ceiling.
That means the interesting engineering (the tool-calling layer, multi-hop retrieval) doesn't
show up as a number on *this* question set. A fair next step would be harder questions that
force cross-file reasoning, where retrieve-once should start to fail. I'm reporting the real
100/100 rather than manufacturing a gap that isn't there.

It was worse than saturated: it scored 50/50 while the agent was answering two real questions
wrong (see `eval/tuning_log.md`, run 2). That's why the benchmark grew.

### Harder 67-question benchmark

All 67 questions, 3 runs each, pass = all 3 runs correct. Same models as above, judged by
gpt-4o (`eval/results.json`, `eval/results_naive.json`):

| | Naive RAG | Agent |
|---|---|---|
| Passed all 3 runs | 67/67 | 67/67 |
| Cross-file questions | 12/12 | 12/12 |
| Flaky questions | none | none |
| Answer cites an expected file | 99/162 (61%) | 135/162 (83%) |
| Cites an expected file, cross-file only | 18/36 | 34/36 |
| Mean tool calls per answer | 0 | 2.5 (3.7 on cross-file) |
| Mean seconds per answer | 2.3 | 5.8 |
| Measured cost per answer, judge included | $0.0006 | $0.0019 |

Naive RAG passes the cross-file questions too, so the harder set is still too easy to separate
the two on accuracy. The only measured difference is that the agent names the file the answer
lives in far more often, at about three times the cost and two and a half times the latency.
The cross-file questions need to span more files, or ask for details that never appear in a
single top-5 retrieval, before accuracy shows whether multi-hop retrieval earns its cost.

The eval found two bugs on the way. On itsdangerous, "what are the main classes?" made the
agent read whole files until it overflowed the model's 128k-token context (issue #15); that's
fixed by the paging and step limit above, and `find_symbol` now answers it in one call. And
two `find_symbol` calls in parallel could crash the agent on Python 3.11.7 (a CPython race in
`ast.parse`); `find_symbol` now parses under a lock.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # paste your OpenAI key into .env

python selftest.py                                    # offline checks, no key needed
python index_repo.py https://github.com/psf/requests # index a repo
python agent.py                                       # one test question
python eval/run_eval.py --runs 3                      # the 67-question benchmark
```

The web app:

```bash
uvicorn main:app --reload          # backend on :8000 (docs at /docs)
cd frontend && npm install && npm run dev   # chat UI on :3000
```

Note: Qdrant local mode allows one process at a time, stop the API server before running
CLI scripts like `index_repo.py` or the eval.

## Files

| File | What it does |
|------|--------------|
| `ingest.py` | Shallow-clones a repo and collects its code files. |
| `chunk.py` | Code-aware chunking (splits at class/def boundaries, 1000/150). |
| `store.py` | Embeddings + Qdrant storage and cosine search; per-repo collections. |
| `index_repo.py` | One command: clone → chunk → embed → store. |
| `answer.py` | Naive RAG baseline: retrieve once, answer from those chunks only. |
| `tools.py` | The agent's tools: `search_code`, `find_symbol`, `list_files`, `read_file`. |
| `agent.py` | The ReAct agent that chains those tools for multi-hop answers. |
| `eval/ground_truth.json` | 67 hand-verified questions across requests, click, flask, itsdangerous. |
| `eval/run_eval.py` | Benchmark runner with an LLM judge; `--naive` for the baseline. |
| `eval/tuning_log.md` | Every tuning change and its measured effect, one variable at a time. |
| `main.py` | FastAPI backend: `/index` and `/ask`. |
| `frontend/` | Minimal Next.js chat UI. |
| `llm.py` | Model choices in one place (agent, judge, embeddings). |
| `selftest.py` | Offline checks for everything that doesn't need an API key. |
| `.github/workflows/keep-alive.yml` | Pings `/health` three times a week so Qdrant doesn't suspend the free cluster. |
