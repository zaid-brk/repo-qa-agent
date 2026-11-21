"""Phase 7 — HTTP API around the agent, so a frontend can call it.

Run: uvicorn main:app --reload   then open http://localhost:8000/docs

CORS is open (*) because the Next.js dev server runs on a different port and
browsers block cross-origin calls unless the API allows them. Fine for a demo;
lock allow_origins down to the real frontend URL in production.
"""
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from index_repo import index_repo
from agent import ask_traced
from store import collection_for, get_active_repo, get_repo_source, indexed_repos, repo_name_from_url
from tools import _all_points

app = FastAPI(title="Repo Q&A Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class IndexReq(BaseModel):
    github_url: str


class AskReq(BaseModel):
    question: str
    repo: str | None = None   # the repo this caller indexed; None = the active repo


@app.post("/index")
def index(req: IndexReq):
    try:
        chunks = index_repo(req.github_url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "indexed", "chunks": chunks, "repo": repo_name_from_url(req.github_url)}


@app.post("/ask")
def ask_endpoint(req: AskReq):
    if req.repo is not None and req.repo not in indexed_repos():
        raise HTTPException(400, f"Repo '{req.repo}' is not indexed. Index it first.")
    answer, trace = ask_traced(req.question, req.repo)
    return {"answer": link_citations(answer, req.repo or get_active_repo()), "trace": trace}


def link_citations(answer: str, repo: str) -> str:
    """Turn cited indexed paths (`src/x.py:120`, src/x.py:3-5) into links to the
    indexed commit on GitHub. Fenced code and code spans with more than a path
    in them are left alone; so is a repo indexed before commits were recorded."""
    source = get_repo_source(repo)
    if not source:
        return answer
    url, sha = source
    files = {p.payload["file"] for p in _all_points(collection_for(repo))}
    path = "|".join(re.escape(f) for f in sorted(files, key=len, reverse=True))
    cite = rf"(?<![\w/.-])({path})(?::(\d+)(?:-(\d+))?)?(?![\w/])"

    def link(text, f, a, b):
        anchor = f"#L{a}" + (f"-L{b}" if b else "") if a else ""
        return f"[{text}]({url}/blob/{sha}/{f}{anchor})"

    def sub(m):
        if m.group("fence"):
            return m.group(0)
        if m.group("span") is not None:
            inner = re.fullmatch(cite, m.group("span"))
            return link(m.group(0), *inner.groups()) if inner else m.group(0)
        return link(m.group(0), *m.groups()[-3:])

    return re.sub(rf"(?P<fence>```.*?```)|`(?P<span>[^`\n]+)`|{cite}", sub, answer, flags=re.S)


@app.get("/status")
def status():
    """Which repo questions currently go to — the UI shows it so you always know."""
    try:
        return {"repo": get_active_repo()}
    except RuntimeError as e:
        return {"repo": None, "detail": str(e)}
