"""Phase 7 — HTTP API around the agent, so a frontend can call it.

Run: uvicorn main:app --reload   then open http://localhost:8000/docs

CORS is open (*) because the Next.js dev server runs on a different port and
browsers block cross-origin calls unless the API allows them. Fine for a demo;
lock allow_origins down to the real frontend URL in production.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from index_repo import index_repo
from agent import ask_traced
from store import get_active_repo

app = FastAPI(title="Repo Q&A Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class IndexReq(BaseModel):
    github_url: str


class AskReq(BaseModel):
    question: str


@app.post("/index")
def index(req: IndexReq):
    chunks = index_repo(req.github_url)
    return {"status": "indexed", "chunks": chunks}


@app.post("/ask")
def ask_endpoint(req: AskReq):
    answer, trace = ask_traced(req.question)
    return {"answer": answer, "trace": trace}


@app.get("/status")
def status():
    """Which repo questions currently go to — the UI shows it so you always know."""
    try:
        return {"repo": get_active_repo()}
    except RuntimeError as e:
        return {"repo": None, "detail": str(e)}
