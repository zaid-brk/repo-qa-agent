"""Phase 5 — the three tools the agent can call.

Naive RAG retrieves 5 chunks and stops. These tools let the agent keep digging:
search again with a better query, list what files exist, or read a whole file
when chunks aren't enough. That's multi-hop retrieval.
"""
import ast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from qdrant_client.models import Filter, FieldCondition, MatchValue

from chunk import join_chunks
from store import search, client, collection_for, get_active_repo


# Cosine search always returns top_k, however poor the match, so the agent used
# to get five plausible-looking chunks even for a concept the repo never mentions
# and would sometimes believe them. A score floor looked like the fix but isn't:
# on itsdangerous "ORM" scores 0.21 and "SQLAlchemy" 0.26 (both absent) while
# "class " scores 0.18 and "Serializer class" 0.31 (both present). Absolute score
# tracks how the query is phrased, not whether the answer exists, so any cutoff
# that hides the absent concepts also hides real code.
#
# Reporting the score instead lets the agent tell strong evidence from weak and
# re-query rather than concluding from noise — which was the actual failure in
# both directions.
WEAK_MATCH = 0.30

# One whole-file read can be 25k+ tokens (click's core.py), and the agent may make
# about a dozen tool calls before langgraph's step limit stops it, so uncapped
# reads overflowed gpt-4o-mini's 128k context (issue #15). 20k chars is ~5k
# tokens, which keeps a dozen reads well under the limit.
MAX_READ_CHARS = 20_000


def _collection(config: RunnableConfig) -> str:
    """The repo this call is about. The API passes it per request (issue #12), so
    one visitor indexing a repo doesn't change what everyone else is asking about;
    the CLI and eval leave it out and get the active repo."""
    repo = (config or {}).get("configurable", {}).get("repo") or get_active_repo()
    return collection_for(repo)


@tool
def search_code(query: str, config: RunnableConfig) -> str:
    """Search the codebase for snippets relevant to a natural-language query."""
    hits = search(query, top_k=5, collection=_collection(config))
    if not hits:
        return "No chunks are stored for this repo."
    note = (
        f"Best cosine similarity {hits[0]['score']:.2f} (0-1). "
        + (
            "That is a weak match: either this repo has nothing on the topic, or the "
            "query is phrased badly. Try re-wording it before concluding anything."
            if hits[0]["score"] < WEAK_MATCH
            else "That is a solid match."
        )
    )
    body = "\n\n".join(
        f"--- {h['file']} (similarity {h['score']:.2f}) ---\n{h['text']}" for h in hits
    )
    return f"{note}\n\n{body}"


def _all_points(collection: str):
    """Every stored chunk in the collection."""
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True)
        yield from points
        if offset is None:
            break


@tool
def list_files(config: RunnableConfig) -> str:
    """List all files that have been indexed from the repo."""
    return "\n".join(sorted({p.payload["file"] for p in _all_points(_collection(config))}))


MAX_SYMBOLS = 200


@tool
def find_symbol(config: RunnableConfig, name: str = "") -> str:
    """Find where Python classes and functions are defined, with file and line.
    Matches any definition whose name contains `name` (case-insensitive); methods
    show as Class.method. Leave `name` empty to list every class in the repo.
    Use this, not search_code, for "where is X defined" or "what classes exist"."""
    # ponytail: rebuilds every .py file per call from the stored chunks (no re-index
    # needed). Fine for repos this size; cache per collection if it gets slow.
    chunks: dict[str, list] = {}
    for p in _all_points(_collection(config)):
        if p.payload["file"].endswith(".py"):
            chunks.setdefault(p.payload["file"], []).append(p.payload)
    found = []
    for path, parts in sorted(chunks.items()):
        parts.sort(key=lambda c: c.get("chunk_index", 0))
        try:
            tree = ast.parse(join_chunks([(c.get("start"), c["text"]) for c in parts]))
        except (SyntaxError, ValueError):
            continue
        stack = [(tree, "")]
        while stack:
            node, prefix = stack.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_class = isinstance(child, ast.ClassDef)
                    hit = name.lower() in child.name.lower() if name else is_class
                    if hit:
                        kind = "class" if is_class else "def"
                        found.append((path, child.lineno, f"{kind} {prefix}{child.name}"))
                    stack.append((child, f"{prefix}{child.name}."))
    if not found:
        return (f"No Python class or function matching '{name}'. "
                "Try a shorter name, or search_code for non-Python files.")
    found.sort()
    lines = [f"{path}:{line} {what}" for path, line, what in found]
    if len(lines) > MAX_SYMBOLS:
        lines = lines[:MAX_SYMBOLS] + [f"[{len(found) - MAX_SYMBOLS} more. Use a more specific name.]"]
    return "\n".join(lines)


@tool
def read_file(file_path: str, config: RunnableConfig, start: int = 0) -> str:
    """Read one indexed file by its path. Long files come back in pages: pass the
    start offset given at the end of a page to read the next one."""
    collection = _collection(config)
    points, _ = client.scroll(
        collection, limit=1000, with_payload=True,
        # Filter model, not a raw dict — local file-mode Qdrant doesn't parse dicts.
        scroll_filter=Filter(must=[FieldCondition(key="file", match=MatchValue(value=file_path))]),
    )
    # Reassemble the file from its chunks, in order, without the overlap between them.
    chunks = sorted(points, key=lambda p: p.payload.get("chunk_index", 0))
    text = join_chunks([(c.payload.get("start"), c.payload["text"]) for c in chunks])
    if not text:
        return f"File not found: {file_path}"
    end = start + MAX_READ_CHARS
    if end >= len(text):
        return text[start:]
    return (f"{text[start:end]}\n\n[Truncated: showed characters {start}-{end} of {len(text)}. "
            f"Call read_file with start={end} to continue.]")
