# GitHub Repo Q&A Agent (RAG)

I built a web app where you paste a GitHub repo URL, it indexes the code, and then you can
ask questions like "where is authentication handled?" or "what does raise_for_status do?"
and get answers grounded in the actual source, with the files cited. Under the hood it's
retrieval-augmented generation (RAG) upgraded to an agent that can search, list files, and
read whole files across multiple hops — plus a 50-question evaluation harness that measures
how accurate it actually is.

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
                            search_code / list_files / read_file
                                              │
question ──────────────────────────> ReAct agent (gpt-4o-mini) ──> answer + cited files
```

1. **Ingest** — shallow-clone the repo, keep only code files (skip binaries, node_modules, etc.).
2. **Chunk** — split files into ~1000-char pieces with 150 overlap. The splitter prefers
   `\nclass ` and `\ndef ` boundaries so whole functions stay together — a real improvement
   over default splitting for code.
3. **Embed + store** — each chunk becomes a 1536-dim vector in Qdrant. Cosine similarity
   makes "how are retries handled?" land near retry code even with zero shared keywords.
4. **Agent** — instead of retrieve-once-and-hope, a ReAct agent decides what to do: search,
   see it needs the full file, read it, search again. That's multi-hop retrieval, and it's
   what handles questions whose answers span several files.

Qdrant runs in local file mode (`qdrant_storage/`), so there's no Docker or server to set
up while developing. For deployment it swaps to Qdrant Cloud with one line. Each repo gets
its own collection so several repos can be indexed and evaluated side by side.

## The eval harness (the part I care about most)

Anyone can wire up an agent. Knowing whether it's *right* is the actual engineering. I wrote
50 questions with known correct answers across three real codebases — requests, click, and
flask — by reading their source and verifying every answer. Four question types:

- **Factual** — "what is the default number of retries?"
- **Locational** — "which file defines the Session class?"
- **Explanatory** — "how are Flask's client-side sessions secured?"
- **Should-say-I-don't-know** — "what ORM ships inside Flask?" (there isn't one; this catches hallucination)

Explanatory answers can't be string-matched, so a stronger model (gpt-4o) grades each answer
against the ground truth — the LLM-as-judge pattern. The judge can be wrong too, which is a
known limitation of the pattern; using a judge stronger than the agent keeps it reliable enough.

`eval/run_eval.py` runs the whole thing and reports overall and per-repo accuracy.
`--naive` runs the same questions through plain single-shot RAG, so the agent's multi-hop
advantage is measured, not assumed. Tuning happens one variable at a time and every run is
recorded in `eval/tuning_log.md` — that log is the evidence behind any accuracy number.

## Results

*Pending a full run.* The harness and every script are done and tested offline; the numbers
require an OpenAI key (indexing all three repos plus a full 50-question eval run costs well
under $1). The table below gets filled from `eval/run_eval.py` output:

| Setup | Accuracy (50 questions) |
|-------|------------------------|
| Naive RAG (retrieve once) | _pending run_ |
| Agent (multi-hop) | _pending run_ |
| Agent after tuning | _pending run_ |

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # paste your OpenAI key into .env

python selftest.py                                    # offline checks, no key needed
python index_repo.py https://github.com/psf/requests # index a repo
python agent.py                                       # one test question
python eval/run_eval.py                               # the 50-question benchmark
```

The web app:

```bash
uvicorn main:app --reload          # backend on :8000 (docs at /docs)
cd frontend && npm install && npm run dev   # chat UI on :3000
```

Note: Qdrant local mode allows one process at a time — stop the API server before running
CLI scripts like `index_repo.py` or the eval.

## Files

| File | What it does |
|------|--------------|
| `ingest.py` | Shallow-clones a repo and collects its code files. |
| `chunk.py` | Code-aware chunking (splits at class/def boundaries, 1000/150). |
| `store.py` | Embeddings + Qdrant storage and cosine search; per-repo collections. |
| `index_repo.py` | One command: clone → chunk → embed → store. |
| `answer.py` | Naive RAG baseline: retrieve once, answer from those chunks only. |
| `tools.py` | The agent's tools: `search_code`, `list_files`, `read_file`. |
| `agent.py` | The ReAct agent that chains those tools for multi-hop answers. |
| `eval/ground_truth.json` | 50 hand-verified questions across requests, click, flask. |
| `eval/run_eval.py` | Benchmark runner with an LLM judge; `--naive` for the baseline. |
| `eval/tuning_log.md` | Every tuning change and its measured effect, one variable at a time. |
| `main.py` | FastAPI backend: `/index` and `/ask`. |
| `frontend/` | Minimal Next.js chat UI. |
| `llm.py` | Model choices in one place (agent, judge, embeddings). |
| `selftest.py` | Offline checks for everything that doesn't need an API key. |

## Notes for myself

Things I should be able to explain without looking: what an embedding is and why cosine
distance, why chunks overlap and why the separators list starts with `\nclass `/`\ndef `,
how the ReAct loop decides to call a tool, the difference between naive RAG and the agent,
how I built ground truth and why the "should say I don't know" questions matter, and the
limits of LLM-as-judge.
