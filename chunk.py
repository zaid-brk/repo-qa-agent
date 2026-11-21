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
    # Keep whitespace so every chunk is a verbatim slice of the file. That's what
    # lets join_chunks rebuild the file exactly from stored chunks.
    strip_whitespace=False,
)


def chunk_file(rel_path: str, text: str):
    """Return list of dicts: {text, metadata} ready for embedding.

    The metadata (which file, which position) is what lets answers cite sources
    and lets the agent (Phase 5) know which file to open next.
    """
    out, end = [], 0
    for i, chunk in enumerate(splitter.split_text(text)):
        # Neighbours overlap by up to chunk_overlap chars, so the next chunk starts
        # within that window. The splitter occasionally drops a blank line between
        # chunks; starting the slice at the previous end absorbs it.
        found = text.find(chunk, max(0, end - splitter._chunk_overlap))
        start = end if found == -1 else min(found, end)
        stop = max(end, start + len(chunk))
        out.append({"text": text[start:stop],
                    "metadata": {"file": rel_path, "chunk_index": i, "start": start}})
        end = stop
    if out and end < len(text):   # same for trailing blank lines it drops at the end
        out[-1]["text"] += text[end:]
    return out


def join_chunks(chunks: list[tuple[int | None, str]]) -> str:
    """Rebuild a file from (start, text) pairs in chunk order, dropping the overlap.

    Chunks indexed before start offsets existed have start=None and are just
    concatenated, which repeats the overlap. Re-index to fix those.
    """
    out = ""
    for start, text in chunks:
        out += text if start is None else text[len(out) - start:]
    return out


if __name__ == "__main__":
    sample = "def hello():\n    print('hi')\n\n" * 100
    out = chunk_file("test.py", sample)
    print(f"{len(out)} chunks; first chunk:\n{out[0]['text'][:200]}")
