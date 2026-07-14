"""Phase 3 — embed chunks and store them in Qdrant; search by meaning.

Qdrant runs in local file mode (everything persisted under ./qdrant_storage)
when QDRANT_URL isn't set, so there's no Docker or server to manage while
developing. Set QDRANT_URL + QDRANT_API_KEY (Qdrant Cloud) to switch to a
remote cluster — needed for deployment, since HF Spaces has no persistent disk.

One collection per repo (chunks_requests, chunks_click, ...) so several repos
can be indexed and evaluated side by side without mixing their vectors.
The "active" repo — the one questions currently go to — is remembered in a
small text file so separate CLI runs agree on it.

Note: local mode allows one process at a time (it holds a file lock). Stop the
API server before running index_repo.py / run_eval.py from the CLI.
"""
import atexit
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

from llm import get_embeddings, EMBED_DIM

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))
else:
    client = QdrantClient(path="qdrant_storage")
atexit.register(client.close)   # clean shutdown; avoids a noisy destructor warning at exit

ACTIVE_FILE = Path("active_repo.txt")


def repo_name_from_url(github_url: str) -> str:
    """'https://github.com/psf/requests' -> 'requests'."""
    return github_url.rstrip("/").removesuffix(".git").split("/")[-1]


def collection_for(repo_name: str) -> str:
    return "chunks_" + repo_name.replace("-", "_")


def set_active_repo(repo_name: str):
    ACTIVE_FILE.write_text(repo_name)


def get_active_repo() -> str:
    if not ACTIVE_FILE.exists():
        raise RuntimeError("No repo indexed yet — run: python index_repo.py <github_url>")
    return ACTIVE_FILE.read_text().strip()


def ensure_collection(name: str):
    """Create the collection if it doesn't exist. EMBED_DIM = vector size (1536)."""
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        # tools.py filters by "file" (read_file) — Qdrant Cloud requires an explicit
        # payload index to filter on a field; local file-mode Qdrant doesn't.
        client.create_payload_index(
            collection_name=name, field_name="file", field_schema=PayloadSchemaType.KEYWORD,
        )


def store_chunks(chunks: list[dict], collection: str) -> int:
    """Embed a batch of chunks and upsert them into Qdrant.

    One embed_documents call per batch (not per chunk) — one HTTP round-trip per
    chunk would be slow and hit rate limits; batching 100 texts is dramatically faster.
    """
    ensure_collection(collection)
    texts = [c["text"] for c in chunks]
    vectors = get_embeddings().embed_documents(texts)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": c["text"], **c["metadata"]},
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


def search(query: str, top_k: int = 5, collection: str | None = None):
    """Embed the query, return the top_k most similar chunks.

    COSINE distance = the angle between vectors: chunks about the same topic point
    in nearly the same direction (score near 1). Standard choice for text embeddings.
    """
    if collection is None:
        collection = collection_for(get_active_repo())
    qvec = get_embeddings().embed_query(query)
    hits = client.query_points(collection_name=collection, query=qvec, limit=top_k).points
    return [{"score": h.score, **h.payload} for h in hits]
