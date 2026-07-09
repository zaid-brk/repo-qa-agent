"""Phase 2 — split files into overlapping chunks.

An LLM can't read a 10,000-line file at once, and embeddings work best on small
focused pieces. Overlap matters: if a chunk boundary cuts a function in half,
neither half makes sense alone — sharing 150 chars with the neighbor keeps
context across the seam.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunk_size = characters per chunk; overlap = shared chars between neighbors.
# 1000/150 is a solid starting point for code. Tuned in Phase 6 (see eval/tuning_log.md).
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    # Try to split at code boundaries first (whole classes/functions stay together),
    # falling back to blank lines, then lines, then words. Real improvement over the
    # default separators for code.
    separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
)


def chunk_file(rel_path: str, text: str):
    """Return list of dicts: {text, metadata} ready for embedding.

    The metadata (which file, which position) is what lets answers cite sources
    and lets the agent (Phase 5) know which file to open next.
    """
    chunks = splitter.split_text(text)
    return [
        {"text": chunk, "metadata": {"file": rel_path, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    sample = "def hello():\n    print('hi')\n\n" * 100
    out = chunk_file("test.py", sample)
    print(f"{len(out)} chunks; first chunk:\n{out[0]['text'][:200]}")
