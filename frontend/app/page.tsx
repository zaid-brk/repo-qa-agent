"use client";
import { useEffect, useState } from "react";
import { Markdown } from "./markdown";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Hop = { tool: string; input: Record<string, string> };
type Result = { question: string; answer: string; trace: Hop[] };

const SAMPLES = [
  "Which file defines the Flask class and what does its run method do?",
  "How does the @app.route decorator end up handling a request? Walk me through the files involved.",
  "What ORM ships inside Flask?",
];

const CODE_FILE = /[\w./-]+\.(?:py|pyi|js|jsx|ts|tsx|go|rs|java|c|cpp|h|md|json|ya?ml)\b/g;

/** Files the answer stands on: what the agent opened, plus paths it cites. */
function citedFiles(r: Result): string[] {
  const fromTrace = r.trace
    .filter((h) => h.tool === "read_file")
    .map((h) => Object.values(h.input)[0]);
  return [...new Set([...fromTrace, ...(r.answer.match(CODE_FILE) ?? [])])];
}

export default function Home() {
  const [repo, setRepo] = useState<string | null>(null);
  const [repoUrl, setRepoUrl] = useState("");
  const [indexNote, setIndexNote] = useState("");
  const [q, setQ] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [showIndex, setShowIndex] = useState(false);

  useEffect(() => {
    fetch(`${API}/status`)
      .then((r) => r.json())
      .then((d) => setRepo(d.repo))
      .catch(() => setRepo(null));
  }, []);

  async function indexRepo() {
    if (!repoUrl) return;
    setIndexNote("Cloning and embedding — a small repo takes a minute or two.");
    try {
      const res = await fetch(`${API}/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: repoUrl }),
      });
      const data = await res.json();
      setIndexNote(`Indexed ${data.chunks} chunks. Ask away.`);
      setRepo(repoUrl.replace(/\/$/, "").split("/").pop() ?? null);
    } catch {
      setIndexNote(`Could not reach the API at ${API}.`);
    }
  }

  async function ask(question: string) {
    if (!question || busy) return;
    setQ("");
    setBusy(true);
    setResult(null);
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setResult({ question, answer: data.answer, trace: data.trace ?? [] });
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
        <div>
          <h1 className="wordmark">
            Repo <em>Q&amp;A</em> Agent
          </h1>
          <p className="tagline">
            Ask a question in English. The agent searches the repo, reads what it needs, and
            answers from the source — showing every file it used.
          </p>
        </div>
        <span className="repo-pill">
          <span className={repo ? "dot" : "dot off"} />
          {repo ? (
            <>
              answering from <b>{repo}</b>
            </>
          ) : (
            "no repo indexed"
          )}
        </span>
      </header>

      <div className="bar">
        <input
          aria-label="Question about the codebase"
          placeholder="How are sessions secured?"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask(q)}
        />
        <button onClick={() => ask(q)} disabled={busy || !q}>
          {busy ? "Working" : "Ask"}
        </button>
      </div>

      <div className="samples">
        <span className="label">Try</span>
        {SAMPLES.map((s) => (
          <button key={s} className="ghost" disabled={busy} onClick={() => ask(s)}>
            {s.length > 58 ? s.slice(0, 56) + "…" : s}
          </button>
        ))}
      </div>

      <div className="index-row">
        <button className="ghost" onClick={() => setShowIndex((v) => !v)}>
          {showIndex ? "Hide" : "Index a different repo"}
        </button>
        {indexNote && <p className="note">{indexNote}</p>}
      </div>

      {showIndex && (
        <div className="bar">
          <input
            aria-label="GitHub repository URL"
            placeholder="https://github.com/pallets/click"
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && indexRepo()}
          />
          <button onClick={indexRepo} disabled={!repoUrl}>
            Index
          </button>
        </div>
      )}

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
