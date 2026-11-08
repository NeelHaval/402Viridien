import { useEffect, useState } from "react";
import "./WellDescription.css";

type Note = { fallback?: boolean; error?: string; note: string; warning: string; model: string | null; cached: boolean; result_hash: string };
type State = { user: object | null; auth_configured: boolean; auth_required: boolean; gemini_configured: boolean };

export function WellDescription({ id, polygon, ready }: { id?: string; polygon?: number[][]; ready: boolean }) {
  const [session, setSession] = useState<State | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    void fetch("/api/session", { signal: controller.signal, cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("Could not check note availability.");
        setSession(await response.json());
      }).catch((e: Error) => { if (e.name !== "AbortError") setError(e.message); });
    return () => controller.abort();
  }, []);

  async function generate() {
    if (busy) return;
    setCopied(false); setBusy(true); setError(""); setNote(null);
    try {
      const response = await fetch(polygon ? "/api/report/description" : `/api/wells/${encodeURIComponent(id!)}/description`, {
        method: "POST", signal: AbortSignal.timeout(18000),
        ...(polygon ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify({ polygon }) } : {}),
      });
      const data = await response.json();
      if (response.status === 401) setSession((s) => s ? { ...s, user: null } : s);
      if (!response.ok) throw new Error(data.error?.message || "The well note is unavailable.");
      setNote(data);
    } catch (e) {
      setError((e as Error).name === "TimeoutError" ? "The note took too long. Try again." : (e as Error).message);
    } finally { setBusy(false); }
  }

  return <section className="well-description" aria-label="Saved-result note">
    <h2>{polygon ? "Area note" : "Scan note"} <span>Gemini</span></h2>
    {!ready ? <p>The scan is not complete. A note cannot be generated.</p>
      : !session ? <p role="status">{error || "Checking note availability…"}</p>
      : !session.gemini_configured ? <p role="status">Gemini notes are disabled. The server API key is not set.</p>
      : session.auth_required && !session.auth_configured ? <p role="status">Gemini notes are disabled. Sign-in setup is not complete.</p>
      : session.auth_required && !session.user ? <p><a href="/auth/login">Sign in with Auth0</a> to generate this note.</p>
      : <>
        {<button onClick={() => void generate()} disabled={busy}>
          {busy ? "Preparing note…" : "Summarise"}
        </button>}
        {error && <p role="alert">{error}</p>}
        <div aria-live="polite" aria-busy={busy}>
          {note && <>{note.fallback && <p role="alert">{note.error}</p>}<p>{note.note}</p><small>{note.warning}</small><button onClick={() => { void navigator.clipboard.writeText(note.note).then(() => setCopied(true)).catch(() => setError("Copy failed. Select the summary text and copy it.")); }}>{copied ? "Copied" : "Copy"}</button>
            <details><summary>Note source</summary><small>Saved result {note.result_hash.slice(0, 12)} · {note.model}{note.cached ? " · cached" : ""}</small></details></>}
        </div>
      </>}
  </section>;
}
