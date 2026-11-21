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
from chunk import chunk_file, join_chunks
from store import repo_name_from_url, collection_for


def test_collect_files():
    d = Path(tempfile.mkdtemp())
    (d / "keep.py").write_text("print('hi')")
    (d / "skip.png").write_bytes(b"\x89PNG")
    (d / "node_modules").mkdir()
    (d / "node_modules" / "junk.js").write_text("var x = 1")
    (d / "pyproject.toml").write_text("[project.scripts]")
    (d / "index.rst").write_text("Docs")
    (d / "package-lock.json").write_text("{}")
    (d / "huge.json").write_text("x" * 300_000)
    got = dict(collect_files(str(d)))
    assert got == {"keep.py": "print('hi')", "pyproject.toml": "[project.scripts]",
                   "index.rst": "Docs"}, got
    print("✓ collect_files keeps code/config/docs, skips binaries, node_modules, lockfiles, huge files")


def test_chunking():
    text = "def f():\n    pass\n\n" * 200          # ~3800 chars -> several chunks
    chunks = chunk_file("big.py", text)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 1000 for c in chunks)
    assert chunks[0]["metadata"] == {"file": "big.py", "chunk_index": 0, "start": 0}
    small = chunk_file("small.py", "x = 1")
    assert len(small) == 1
    print(f"✓ chunking: {len(chunks)} chunks, all <=1000 chars, metadata intact")


def test_join_chunks_rebuilds_files_exactly():
    """read_file rebuilds a file from overlapping chunks. Must equal the original:
    no repeated overlap at the seams, nothing dropped between chunks."""
    texts = [p.read_text() for p in Path(__file__).parent.glob("*.py")]
    texts += ["def f():\n    pass\n\n" * 200, "x = 1\n" * 300 + "\n\n\n", "a b " * 3000, "x" * 5000, ""]
    for t in texts:
        cs = chunk_file("f.py", t)
        assert join_chunks([(c["metadata"]["start"], c["text"]) for c in cs]) == t
    print(f"✓ join_chunks rebuilds {len(texts)} files exactly")


def test_failed_clone_keeps_old_index():
    """A bad URL must not wipe the repo's existing vectors."""
    import index_repo as ir
    calls = []
    def bad_clone(url, dest):
        raise RuntimeError("clone failed")
    orig = ir.clone_repo, ir.reset_collection
    ir.clone_repo, ir.reset_collection = bad_clone, calls.append
    try:
        ir.index_repo("https://github.com/x/y")
    except RuntimeError:
        pass
    finally:
        ir.clone_repo, ir.reset_collection = orig
    assert calls == [], calls
    print("✓ failed clone leaves the existing index alone")


def _fake_store(files: dict[str, str]):
    """Point tools.py at an in-memory Qdrant holding these files, chunked as indexing would."""
    import tools
    c = QdrantClient(":memory:")
    c.create_collection("chunks_fake", vectors_config=VectorParams(size=1, distance=Distance.COSINE))
    c.upsert("chunks_fake", [
        PointStruct(id=str(uuid.uuid4()), vector=[1.0], payload={"text": ch["text"], **ch["metadata"]})
        for path, text in files.items() for ch in chunk_file(path, text)
    ])
    tools.client, tools.get_active_repo = c, lambda: "fake"
    return tools


def test_read_file_pages_big_files():
    """One read_file call must not dump a huge file into the agent's context
    (issue #15). Big files come back in capped pages with a hint to continue;
    following the hints rebuilds the file exactly. Small files are untouched."""
    import re
    big = "".join(f"line {i}: some code here\n" for i in range(4000))   # ~100k chars
    small = "x = 1\n"
    tools = _fake_store({"big.py": big, "small.py": small})

    assert tools.read_file.invoke({"file_path": "small.py"}) == small

    pages, start = [], 0
    while True:
        out = tools.read_file.invoke({"file_path": "big.py", "start": start})
        assert len(out) <= 21_000, len(out)
        m = re.search(r"start=(\d+)", out)
        if not m:
            pages.append(out)
            break
        pages.append(out[:out.rindex("\n\n[Truncated")])
        start = int(m.group(1))
    assert "".join(pages) == big
    assert len(pages) == 6, len(pages)   # 102,890 chars / 20,000 per page
    print(f"✓ read_file pages a 100k-char file in {len(pages)} capped pages, small files exact")


def test_runaway_agent_stays_in_context():
    """Worst case for issue #15: a model that reads the same huge file every turn
    and never answers. The loop must end on its own with a reply, and the context
    the model is sent must stay well under gpt-4o-mini's 128k tokens (~4 chars
    per token, so 400k chars is a ~100k-token budget)."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    import agent

    seen = []

    class AlwaysReads(BaseChatModel):
        @property
        def _llm_type(self):
            return "always-reads"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(sum(len(str(m.content)) for m in messages))
            call = {"name": "read_file", "args": {"file_path": "huge.py"}, "id": f"c{len(seen)}"}
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])

    _fake_store({"huge.py": "x = 1  # padding\n" * 30_000})   # ~510k chars
    agent.get_llm, agent._agent = AlwaysReads, None
    try:
        answer, trace = agent.ask_traced("read everything")
    finally:
        agent._agent = None
    assert answer == agent.OUT_OF_STEPS, answer
    assert max(seen) < 400_000, max(seen)
    print(f"✓ runaway agent stopped after {len(trace)} reads, peak context {max(seen):,} chars")


def test_search_code_shows_line_ranges():
    """Each hit says which lines it covers, so answers can cite path:line."""
    text = "".join(f"line {i}\n" for i in range(1, 400))     # each line names itself
    tools = _fake_store({"src/a.py": text})
    hits = [{"score": 0.9, **p.payload} for p in tools.client.scroll("chunks_fake", limit=100)[0]]
    hits.sort(key=lambda h: h["chunk_index"])
    orig = tools.search
    tools.search = lambda query, top_k=5, collection=None: hits[1:3]
    try:
        out = tools.search_code.invoke({"query": "x"})
    finally:
        tools.search = orig
    for h in hits[1:3]:
        first, last = h["text"].split()[1], h["text"].rstrip().split()[-1]
        assert f"--- src/a.py:{first}-{last} (similarity 0.90) ---" in out, (first, last, out[:300])
    print("✓ search_code hits show their line ranges")


def test_ask_links_citations_to_github():
    """Cited indexed paths become links to that exact commit and line; code,
    unknown paths and repos with no recorded commit are left alone."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from fastapi.testclient import TestClient
    import agent, main, store

    said = ("See `src/a.py:12` and src/a.py:3-5, also `src/a.py`, docs/missing.py:4 and xsrc/a.py.\n"
            "```\nsrc/a.py:9\n```\n`src/a.py:12 Signer`")

    class Cites(BaseChatModel):
        @property
        def _llm_type(self):
            return "cites"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=said))])

    tools = _fake_store({"src/a.py": "x = 1\n"})
    orig = store.client, agent.get_llm
    store.client = tools.client
    agent.get_llm, agent._agent = Cites, None
    try:
        api = TestClient(main.app)
        ask = lambda: api.post("/ask", json={"question": "q", "repo": "fake"}).json()["answer"]
        assert ask() == said        # indexed before commits were recorded: no links
        store.set_repo_source("fake", "https://github.com/o/r", "abc123")
        b = "https://github.com/o/r/blob/abc123/src/a.py"
        assert ask() == (f"See [`src/a.py:12`]({b}#L12) and [src/a.py:3-5]({b}#L3-L5), also "
                         f"[`src/a.py`]({b}), docs/missing.py:4 and xsrc/a.py.\n"
                         "```\nsrc/a.py:9\n```\n`src/a.py:12 Signer`"), ask()
    finally:
        store.client, agent.get_llm = orig
        agent._agent = None
    print("✓ /ask links cited files to the indexed commit and line")


def test_find_symbol():
    """Exact definitions with line numbers, which semantic search is bad at
    ("class " scored below unrelated concepts on itsdangerous, issue #9)."""
    signer = "import hmac\n\n\nclass Signer:\n    def sign(self, value):\n        return value\n\n\ndef want_bytes(s):\n    return s\n"
    serializer = "class Serializer:\n    pass\n\n\nclass URLSafeSerializer(Serializer):\n    def dumps(self, obj):\n        pass\n"
    tools = _fake_store({"src/signer.py": signer, "src/serializer.py": serializer,
                         "broken.py": "def oops(:\n", "README.md": "class Signer in docs"})
    find = lambda name="": tools.find_symbol.invoke({"name": name})
    assert find("Signer") == "src/signer.py:4 class Signer", find("Signer")
    assert find("sign") == "src/signer.py:5 def Signer.sign\nsrc/signer.py:4 class Signer", find("sign")  # exact first
    assert find("serializer") == ("src/serializer.py:1 class Serializer\n"
                                  "src/serializer.py:5 class URLSafeSerializer"), find("serializer")
    assert find("want_bytes") == "src/signer.py:9 def want_bytes", find("want_bytes")
    assert find() == ("src/serializer.py:1 class Serializer\nsrc/serializer.py:5 class URLSafeSerializer\n"
                      "src/signer.py:4 class Signer"), find()
    assert find("nothing_here").startswith("No Python class or function"), find("nothing_here")
    print("✓ find_symbol gives exact definitions with line numbers")


def test_index_rejects_non_github_urls():
    """/index must only clone https://github.com/<owner>/<repo>. Anything else
    is a 400 and git is never run."""
    import index_repo as ir
    from fastapi.testclient import TestClient
    import main

    cloned = []
    class Stop(Exception):
        pass
    def fake_clone(url, dest):
        cloned.append(url)
        raise Stop
    orig = ir.clone_repo
    ir.clone_repo = fake_clone
    try:
        client = TestClient(main.app, raise_server_exceptions=False)
        bad = [
            "file:///etc", "ssh://git@github.com/psf/requests", "git@github.com:psf/requests.git",
            "http://github.com/psf/requests", "https://gitlab.com/psf/requests",
            "https://github.com/psf", "https://github.com/psf/requests/tree/main",
            "https://github.com.evil.com/psf/requests", "https://user@github.com/psf/requests",
            "--upload-pack=touch /tmp/x", "https://github.com/psf/requests\n", "",
        ]
        for url in bad:
            r = client.post("/index", json={"github_url": url})
            assert r.status_code == 400, (url, r.status_code)
        assert cloned == [], cloned
        for url in ["https://github.com/psf/requests", "https://github.com/pallets/click.git",
                    "https://github.com/a-b/c_d.e/"]:
            client.post("/index", json={"github_url": url})
        assert len(cloned) == 3, cloned
    finally:
        ir.clone_repo = orig
    print("✓ /index rejects anything but https://github.com/<owner>/<repo>")


def test_missing_repo_is_a_clear_400():
    """A repo git can't clone (typo, private) is the caller's mistake: 400 with
    a message the UI can show, not a bare 500."""
    import subprocess, ingest
    from fastapi.testclient import TestClient
    import main
    def fail(*a, **k):
        raise subprocess.CalledProcessError(128, a[0])
    orig = ingest.subprocess.run
    ingest.subprocess.run = fail
    try:
        r = TestClient(main.app, raise_server_exceptions=False).post(
            "/index", json={"github_url": "https://github.com/x/does-not-exist"})
    finally:
        ingest.subprocess.run = orig
    assert r.status_code == 400 and "clone" in r.json()["detail"], (r.status_code, r.text)
    print("✓ missing repo gives a 400 with a readable message")


def test_ask_uses_the_callers_repo():
    """/ask must answer from the repo the caller names, not whichever repo was
    indexed last by anyone (issue #12)."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from fastapi.testclient import TestClient
    import agent, main, store, tools

    class ListsThenAnswers(BaseChatModel):
        @property
        def _llm_type(self):
            return "lists-then-answers"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            last = messages[-1]
            if isinstance(last, ToolMessage):   # answer with what list_files saw
                msg = AIMessage(content=last.content)
            else:
                msg = AIMessage(content="", tool_calls=[{"name": "list_files", "args": {}, "id": "c1"}])
            return ChatResult(generations=[ChatGeneration(message=msg)])

    c = QdrantClient(":memory:")
    for repo in ["alpha", "beta"]:
        c.create_collection(f"chunks_{repo}", vectors_config=VectorParams(size=1, distance=Distance.COSINE))
        c.upsert(f"chunks_{repo}", [PointStruct(id=str(uuid.uuid4()), vector=[1.0],
                 payload={"text": "x = 1", "file": f"{repo}.py", "chunk_index": 0, "start": 0})])
    orig = store.client, tools.client, tools.get_active_repo, agent.get_llm
    store.client = tools.client = c
    tools.get_active_repo = lambda: "beta"       # someone else indexed beta last
    agent.get_llm, agent._agent = ListsThenAnswers, None
    try:
        api = TestClient(main.app)
        r = api.post("/ask", json={"question": "what files?", "repo": "alpha"})
        assert r.status_code == 200 and r.json()["answer"] == "alpha.py", r.text
        r = api.post("/ask", json={"question": "what files?"})   # no repo: old behaviour
        assert r.json()["answer"] == "beta.py", r.text
        r = api.post("/ask", json={"question": "what files?", "repo": "nope"})
        assert r.status_code == 400, r.text
    finally:
        store.client, tools.client, tools.get_active_repo, agent.get_llm = orig
        agent._agent = None
    print("✓ /ask answers from the caller's repo, not the global one")


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
    test_join_chunks_rebuilds_files_exactly()
    test_failed_clone_keeps_old_index()
    test_read_file_pages_big_files()
    test_runaway_agent_stays_in_context()
    test_search_code_shows_line_ranges()
    test_ask_links_citations_to_github()
    test_find_symbol()
    test_index_rejects_non_github_urls()
    test_missing_repo_is_a_clear_400()
    test_ask_uses_the_callers_repo()
    test_collection_naming()
    test_qdrant_roundtrip()
    test_imports_without_key()
    print("\nAll offline checks passed.")
