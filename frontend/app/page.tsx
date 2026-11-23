"use client";
import { useEffect, useState } from "react";
import { Markdown } from "./markdown";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Hop = { tool: string; input: Record<string, string> };
type Result = { question: string; answer: string; trace: Hop[] };

const CODE_FILE = /[\w./-]+\.(?:py|pyi|js|jsx|ts|tsx|go|rs|java|c|cpp|h|md|json|ya?ml)\b/g;

/** Files the answer stands on: what the agent opened, plus paths it cites. */
function citedFiles(r: Result): string[] {
  const fromTrace = r.trace
    .filter((h) => h.tool === "read_file")
    .map((h) => Object.values(h.input)[0]);
  return [...new Set([...fromTrace, ...(r.answer.replace(/\]\([^)]*\)/g, "]").match(CODE_FILE) ?? [])])];
}

export default function Home() {
  const [repoUrl, setRepoUrl] = useState("");
  const [indexNote, setIndexNote] = useState("");
  // The repo this visitor indexed, sent with every question so another
  // visitor indexing a different repo doesn't change our answers.
  const [repo, setRepo] = useState<string | undefined>();
  // Bring your own key: the API spends the visitor's OpenAI key, never the
  // owner's. Kept in this browser only so it survives a reload.
  const [apiKey, setApiKey] = useState("");

  // Remember the repo across reloads. Storage can be blocked (private mode),
  // in which case we just fall back to the server's active repo.
  useEffect(() => {
    try {
      setApiKey(localStorage.getItem("openai_key") ?? "");
      const saved = localStorage.getItem("repo");
      if (saved) {
        setRepo(saved);
        setIndexNote(`Asking about ${saved}.`);
      }
    } catch {}
  }, []);
  const [q, setQ] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [indexing, setIndexing] = useState(false);

  function saveKey(k: string) {
    setApiKey(k);
    try {
      if (k) localStorage.setItem("openai_key", k);
      else localStorage.removeItem("openai_key");
    } catch {}
  }

  const headers = { "Content-Type": "application/json", "X-OpenAI-Key": apiKey.trim() };

  async function indexRepo() {
    if (!repoUrl || indexing || !apiKey.trim()) return;
    setIndexing(true);
    setResult(null);
    setIndexNote("Cloning, splitting into chunks, embedding…");
    try {
      const res = await fetch(`${API}/index`, {
        method: "POST",
        headers,
        body: JSON.stringify({ github_url: repoUrl }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        setRepo(data.repo);
        try {
          localStorage.setItem("repo", data.repo);
        } catch {}
      }
      setIndexNote(res.ok ? `Indexed ${data.chunks} chunks. Ask away.` : data.detail ?? "Indexing failed.");
    } catch {
      setIndexNote(`Could not reach the API at ${API}.`);
    }
    setIndexing(false);
  }

  async function ask(question: string) {
    if (!question || busy || !apiKey.trim()) return;
    setQ("");
    setBusy(true);
    setResult(null);
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers,
        body: JSON.stringify({ question, repo }),
      });
      const data = await res.json().catch(() => ({}));
      setResult({ question, answer: data.answer ?? data.detail ?? "Something went wrong.", trace: data.trace ?? [] });
    } catch {
      setResult({
        question,
        answer: `Could not reach the API at ${API}. Start it with: uvicorn main:app --reload`,
        trace: [],
      });
    }
    setBusy(false);
  }

  return (
    <main className="shell">
      <header className="masthead">
        <h1 className="wordmark">
          Repo <em>Q&amp;A</em> Agent
        </h1>
      </header>

      {/* Numbered because the order is real: no key, no indexing; nothing can
          be asked until a repo has been indexed. */}
      <section className="step">
        <h2 className="step-head">
          <span className="step-n">1</span> Add your OpenAI key
        </h2>
        <div className="bar">
          <input
            type="password"
            aria-label="OpenAI API key"
            placeholder="sk-..."
            autoComplete="off"
            spellCheck={false}
            value={apiKey}
            onChange={(e) => saveKey(e.target.value)}
          />
        </div>
        <p className="note">
          <span>
            Each question costs about a cent on your key. It goes to this demo&apos;s API only for your
            requests and is never stored there. It stays saved in this browser until you clear the
            field. <a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer">Get a key</a>
          </span>
        </p>
      </section>

      <section className="step">
        <h2 className="step-head">
          <span className="step-n">2</span> Point it at a repo
        </h2>
        <div className="bar">
          <input
            aria-label="GitHub repository URL"
            placeholder="https://github.com/pallets/itsdangerous"
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && indexRepo()}
          />
          <button onClick={indexRepo} disabled={!repoUrl || indexing || !apiKey.trim()}>
            {indexing ? "Indexing" : "Index"}
          </button>
        </div>
        {indexNote && (
          <p className="note">
            {indexing && <span className="pulse" />}
            {indexNote}
          </p>
        )}
      </section>

      <section className="step">
        <h2 className="step-head">
          <span className="step-n">3</span> Ask about it
        </h2>
        <div className="bar">
          <input
            aria-label="Question about the codebase"
            placeholder="Where is the signing logic?"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask(q)}
          />
          <button onClick={() => ask(q)} disabled={busy || !q || !apiKey.trim()}>
            {busy ? "Working" : "Ask"}
          </button>
        </div>
      </section>

      {busy && (
        <p className="thinking">
          <span className="pulse" />
          searching, reading, deciding what it still needs…
        </p>
      )}

      {result && !busy && (
        <section className="result">
          <div>
            <h2 className="rail-head">Retrieval trace</h2>
            {result.trace.map((h, i) => (
              <div key={i} className="hop" style={{ animationDelay: `${i * 90}ms` }}>
                <span className="hop-n">{i + 1}</span>
                <div className="hop-tool">{h.tool}</div>
                <div className="hop-arg">{Object.values(h.input)[0] ?? "—"}</div>
              </div>
            ))}
            <p className="hop-count">
              {result.trace.length} {result.trace.length === 1 ? "hop" : "hops"} before answering
            </p>
          </div>

          <div>
            <h2 className="answer-head">Answer</h2>
            <p className="question">{result.question}</p>
            <Markdown text={result.answer} />
            {citedFiles(result).length > 0 && (
              <div className="cites">
                <h3 className="rail-head">Grounded in</h3>
                <ul className="cite-list">
                  {citedFiles(result).map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      )}
    </main>
  );
}
