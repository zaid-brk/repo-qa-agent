"""Phase 6 — the eval harness. This is what makes the project measurable.

50 hand-written questions with known answers (ground_truth.json) across three
repos. The agent answers each; a STRONGER model (the judge) grades whether the
answer matches the ground truth — string matching can't grade explanatory
answers, so we use the LLM-as-judge pattern.

Usage (from the project root, with each repo's collection indexed):
    python eval/run_eval.py                 # full 50-question run with the agent
    python eval/run_eval.py --naive         # same questions through naive RAG (baseline comparison)
    python eval/run_eval.py --repo requests # one repo only (faster tuning loop)

Repos are indexed automatically on first run. NOTE: after changing chunking
(chunk.py), delete qdrant_storage/ so the repos get re-chunked and re-indexed —
otherwise you're evaluating the old chunks.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import from project root

from llm import get_judge
from store import client, collection_for, set_active_repo

# The three benchmark repos. Small (~10-20k lines each) so indexing costs pennies.
REPOS = {
    "requests": "https://github.com/psf/requests",
    "click": "https://github.com/pallets/click",
    "flask": "https://github.com/pallets/flask",
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


def run(naive: bool = False, only_repo: str | None = None):
    if naive:
        from answer import answer_question as ask   # Phase 4 baseline: retrieve once, answer
    else:
        from agent import ask                        # Phase 5 agent: multi-hop

    data = json.load(open(Path(__file__).parent / "ground_truth.json"))
    if only_repo:
        data = [d for d in data if d["repo"] == only_repo]

    judge = get_judge()
    results = []
    correct = 0
    per_repo = {}
    active = None
    for item in data:
        if item["repo"] != active:                   # questions are grouped by repo
            ensure_indexed(item["repo"])
            set_active_repo(item["repo"])
            active = item["repo"]
        actual = ask(item["question"])
        ok = grade(judge, item["question"], item["expected"], actual)
        correct += ok
        per_repo.setdefault(item["repo"], []).append(ok)
        results.append({**item, "actual": actual, "correct": ok})
        print(f"[{item['id']:>2}] {'✓' if ok else '✗'} ({item['repo']}) {item['question'][:60]}")

    print(f"\nAccuracy: {correct}/{len(data)} = {correct / len(data):.1%}  "
          f"({'naive RAG' if naive else 'agent'})")
    for repo, oks in per_repo.items():
        print(f"  {repo}: {sum(oks)}/{len(oks)} = {sum(oks) / len(oks):.1%}")

    out = Path(__file__).parent / ("results_naive.json" if naive else "results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"Per-question results -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--naive", action="store_true", help="run the Phase 4 naive RAG instead of the agent")
    p.add_argument("--repo", choices=REPOS, help="limit to one repo (faster tuning loop)")
    args = p.parse_args()
    run(naive=args.naive, only_repo=args.repo)
