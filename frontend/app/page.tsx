"use client";
import { useState } from "react";

const API = "http://localhost:8000";

export default function Home() {
  const [repoUrl, setRepoUrl] = useState("");
  const [indexStatus, setIndexStatus] = useState("");
  const [q, setQ] = useState("");
  const [messages, setMessages] = useState<{ role: string; text: string }[]>([]);
  const [busy, setBusy] = useState(false);

  async function indexRepo() {
    setIndexStatus("Indexing... (a small repo takes a minute or two)");
    try {
      const res = await fetch(`${API}/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: repoUrl }),
      });
      const data = await res.json();
      setIndexStatus(`Indexed ${data.chunks} chunks. Ask away.`);
    } catch {
      setIndexStatus("Indexing failed — is the backend running on :8000?");
    }
  }

  async function send() {
    if (!q || busy) return;
    const question = q;
    setQ("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: question }]);
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setMessages((m) => [...m, { role: "assistant", text: data.answer }]);
    } catch {
      setMessages((m) => [...m, { role: "assistant", text: "Error — is the backend running?" }]);
    }
    setBusy(false);
  }

  return (
    <main className="max-w-2xl mx-auto p-6">
      <h1 className="text-xl font-bold mb-4">Repo Q&A Agent</h1>

      <div className="flex gap-2 mb-2">
        <input
          className="flex-1 border p-2 rounded"
          placeholder="https://github.com/psf/requests"
          value={repoUrl}
          onChange={(e) => setRepoUrl(e.target.value)}
        />
        <button className="bg-black text-white px-4 rounded" onClick={indexRepo}>
          Index
        </button>
      </div>
      <p className="text-sm text-gray-500 mb-6">{indexStatus}</p>

      {messages.map((m, i) => (
        <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
          <p className="my-2 p-2 rounded bg-gray-100 text-black inline-block whitespace-pre-wrap text-left">
            {m.text}
          </p>
        </div>
      ))}
      {busy && <p className="text-sm text-gray-500">thinking...</p>}

      <div className="flex gap-2 mt-4">
        <input
          className="flex-1 border p-2 rounded"
          placeholder="Where is authentication handled?"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button className="bg-black text-white px-4 rounded" onClick={send} disabled={busy}>
          Ask
        </button>
      </div>
    </main>
  );
}
