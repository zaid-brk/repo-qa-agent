"""Phase 6 — the eval harness. This is what makes the project measurable.

Hand-written questions with known answers (ground_truth.json) across four
repos, including cross-file questions that need more than one retrieval. The agent answers each; a STRONGER model (the judge) grades whether the
answer matches the ground truth — string matching can't grade explanatory
answers, so we use the LLM-as-judge pattern.

Usage (from the project root, with each repo's collection indexed):
    python eval/run_eval.py                 # full 50-question run with the agent
    python eval/run_eval.py --naive         # same questions through naive RAG (baseline comparison)
    python eval/run_eval.py --repo requests # one repo only (faster tuning loop)
    python eval/run_eval.py --runs 3        # ask each question 3 times; pass = all 3 pass

Besides accuracy it reports flaky questions (pass some runs, fail others), whether
the answer cites an expected file (questions with expected_files), and the mean
tool calls and seconds per answer, so agent vs naive compares on cost too.

Repos are indexed automatically on first run. NOTE: after changing chunking
(chunk.py), delete qdrant_storage/ so the repos get re-chunked and re-indexed —
otherwise you're evaluating the old chunks.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import from project root

from openai import BadRequestError

from llm import get_judge
from store import client, collection_for, set_active_repo

# The benchmark repos. Small (~10-20k lines each) so indexing costs pennies.
REPOS = {
    "requests": "https://github.com/psf/requests",
    "click": "https://github.com/pallets/click",
    "flask": "https://github.com/pallets/flask",
    "itsdangerous": "https://github.com/pallets/itsdangerous",
}

JUDGE_PROMPT = """You are grading a code-assistant's answer.
Question: {q}
Expected (ground truth): {expected}
Actual answer: {actual}

Is the actual answer correct and consistent with the expected answer?
Reply with exactly one word: CORRECT or INCORRECT."""


def grade(judge, q, expected, actual) -> bool:
    verdict = judge.invoke(JUDGE_PROMPT.format(q=q, expected=expected, actual=actual)).content
    return "CORRECT" in verdict.upper()


def ensure_indexed(repo: str):
    """Index the repo on first use so the eval is one command."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_for(repo) not in existing:
        from index_repo import index_repo
        print(f"[index] {repo} not indexed yet, indexing...")
        index_repo(REPOS[repo])


def cites(answer: str, files: list[str]) -> bool:
    """Did the answer name at least one of the files the answer lives in?"""
    return any(f.lower() in answer.lower() for f in files)


def pct(n, d):
    return f"{n}/{d} = {n / d:.1%}" if d else "n/a"


def run(naive: bool = False, only_repo: str | None = None, runs: int = 1):
    if naive:
        from answer import answer_question
        ask = lambda q: (answer_question(q), [])   # one retrieval, no tool calls
    else:
        from agent import ask_traced as ask        # multi-hop, returns the tool trace

    data = json.load(open(Path(__file__).parent / "ground_truth.json"))
    if only_repo:
        data = [d for d in data if d["repo"] == only_repo]

    judge = get_judge()
    results = []
    active = None
    for item in data:
        if item["repo"] != active:                   # questions are grouped by repo
            ensure_indexed(item["repo"])
            set_active_repo(item["repo"])
            active = item["repo"]
        attempts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            try:
                actual, trace = ask(item["question"])
                ok = grade(judge, item["question"], item["expected"], actual)
            # The agent overflowing its context window is a wrong answer, not a reason
            # to stop. Quota, auth and network errors still stop the run: scoring
            # those as wrong answers would produce a meaningless accuracy number.
            except BadRequestError as e:
                actual, trace, ok = f"ERROR {type(e).__name__}: {str(e)[:300]}", [], False
            secs = time.perf_counter() - t0
            attempts.append({
                "actual": actual,
                "correct": ok,
                "cites_expected_file": cites(actual, item["expected_files"]) if "expected_files" in item else None,
                "tool_calls": len(trace),
                "seconds": round(secs, 1),
            })
        n_ok = sum(a["correct"] for a in attempts)
        results.append({**item, "correct": n_ok == runs, "flaky": 0 < n_ok < runs, "runs": attempts})
        mark = "✓" if n_ok == runs else ("~" if n_ok else "✗")
        print(f"[{item['id']:>2}] {mark} {n_ok}/{runs} ({item['repo']}, {item['type']}) {item['question'][:60]}")

    def summary(rows):
        return pct(sum(r["correct"] for r in rows), len(rows))

    label = "naive RAG" if naive else "agent"
    print(f"\nAccuracy ({label}, pass = all {runs} run(s) correct): {summary(results)}")
    for key in ("repo", "type"):
        for val in dict.fromkeys(r[key] for r in results):
            print(f"  {val}: {summary([r for r in results if r[key] == val])}")
    flaky = [r["id"] for r in results if r["flaky"]]
    if runs > 1:
        print(f"Flaky (passed some runs, not all): {flaky or 'none'}")
    att = [a for r in results for a in r["runs"]]
    cited = [a["cites_expected_file"] for a in att if a["cites_expected_file"] is not None]
    print(f"Cites an expected file: {pct(sum(cited), len(cited))}")
    print(f"Mean tool calls per answer: {sum(a['tool_calls'] for a in att) / len(att):.1f}")
    print(f"Mean seconds per answer: {sum(a['seconds'] for a in att) / len(att):.1f}")

    out = Path(__file__).parent / ("results_naive.json" if naive else "results.json")
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"Per-question results -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--naive", action="store_true", help="run the Phase 4 naive RAG instead of the agent")
    p.add_argument("--repo", choices=REPOS, help="limit to one repo (faster tuning loop)")
    p.add_argument("--runs", type=int, default=1, help="ask each question N times")
    args = p.parse_args()
    run(naive=args.naive, only_repo=args.repo, runs=args.runs)
