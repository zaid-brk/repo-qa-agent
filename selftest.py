"""Offline sanity checks — no API key, no network, no cost.

Run: python selftest.py
Covers the plumbing that doesn't need an LLM: file collection, chunking,
collection naming, and Qdrant storage/search with fake vectors. If this passes,
the only untested pieces are the OpenAI calls themselves.
"""
import tempfile
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ingest import collect_files
from chunk import chunk_file
from store import repo_name_from_url, collection_for


def test_collect_files():
    d = Path(tempfile.mkdtemp())
    (d / "keep.py").write_text("print('hi')")
    (d / "skip.png").write_bytes(b"\x89PNG")
    (d / "node_modules").mkdir()
    (d / "node_modules" / "junk.js").write_text("var x = 1")
    got = dict(collect_files(str(d)))
    assert got == {"keep.py": "print('hi')"}, got
    print("✓ collect_files keeps code, skips binaries and node_modules")


def test_chunking():
    text = "def f():\n    pass\n\n" * 200          # ~3800 chars -> several chunks
    chunks = chunk_file("big.py", text)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 1000 for c in chunks)
    assert chunks[0]["metadata"] == {"file": "big.py", "chunk_index": 0}
    small = chunk_file("small.py", "x = 1")
    assert len(small) == 1
    print(f"✓ chunking: {len(chunks)} chunks, all <=1000 chars, metadata intact")


def test_collection_naming():
    assert repo_name_from_url("https://github.com/psf/requests") == "requests"
    assert repo_name_from_url("https://github.com/pallets/click.git") == "click"
    assert collection_for("my-repo") == "chunks_my_repo"
    print("✓ collection naming")


def test_qdrant_roundtrip():
    """Store and search with fake 4-dim vectors — proves the Qdrant plumbing
    without paying for embeddings."""
    c = QdrantClient(":memory:")
    c.create_collection("t", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    c.upsert("t", [
        PointStruct(id=str(uuid.uuid4()), vector=[1, 0, 0, 0], payload={"text": "auth code", "file": "auth.py", "chunk_index": 0}),
        PointStruct(id=str(uuid.uuid4()), vector=[0, 1, 0, 0], payload={"text": "db code", "file": "db.py", "chunk_index": 0}),
    ])
    hits = c.query_points("t", query=[0.9, 0.1, 0, 0], limit=1).points
    assert hits[0].payload["file"] == "auth.py"
    print("✓ qdrant store + cosine search roundtrip")


def test_imports_without_key():
    """Every module must import with no OPENAI_API_KEY — models are lazy."""
    import answer, agent, tools, main  # noqa: F401
    print("✓ all modules import without an API key")


if __name__ == "__main__":
    test_collect_files()
    test_chunking()
    test_collection_naming()
    test_qdrant_roundtrip()
    test_imports_without_key()
    print("\nAll offline checks passed.")
