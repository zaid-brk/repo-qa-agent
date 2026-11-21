"""Phase 5 — the three tools the agent can call.

Naive RAG retrieves 5 chunks and stops. These tools let the agent keep digging:
search again with a better query, list what files exist, or read a whole file
when chunks aren't enough. That's multi-hop retrieval.
"""
from langchain_core.tools import tool
from qdrant_client.models import Filter, FieldCondition, MatchValue

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


@tool
def search_code(query: str) -> str:
    """Search the codebase for snippets relevant to a natural-language query."""
    hits = search(query, top_k=5)
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


@tool
def list_files() -> str:
    """List all files that have been indexed from the repo."""
    # Scroll through stored points and collect unique file names.
    collection = collection_for(get_active_repo())
    files = set()
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True)
        for p in points:
            files.add(p.payload["file"])
        if offset is None:
            break
    return "\n".join(sorted(files))


@tool
def read_file(file_path: str) -> str:
    """Read the full contents of one indexed file by its path."""
    collection = collection_for(get_active_repo())
    points, _ = client.scroll(
        collection, limit=1000, with_payload=True,
        # Filter model, not a raw dict — local file-mode Qdrant doesn't parse dicts.
        scroll_filter=Filter(must=[FieldCondition(key="file", match=MatchValue(value=file_path))]),
    )
    # Reassemble the file from its chunks, in order.
    chunks = sorted(points, key=lambda p: p.payload.get("chunk_index", 0))
    return "\n".join(c.payload["text"] for c in chunks) or f"File not found: {file_path}"
