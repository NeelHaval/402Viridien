import { type ReactNode, useCallback, useEffect, useState } from "react";
import "./Providers.css";
type Session = { user: { name?: string; email?: string } | null; auth_required: boolean; auth_configured: boolean };
export function Providers({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [failure, setFailure] = useState("");
  const load = useCallback(async () => {
    setFailure("");
    try {
      const response = await fetch("/api/session", { cache: "no-store" });
      if (!response.ok) throw new Error("Could not check sign-in. Select Retry.");
      setSession(await response.json());
    } catch (e) { setFailure((e as Error).message); }
  }, []);
  useEffect(() => {
    void load();
    const expired = () => { setSession(null); void load(); };
    window.addEventListener("session-expired", expired);
    return () => window.removeEventListener("session-expired", expired);
  }, [load]);
  if (session?.auth_required && !session.user) return <div className="provider-login">
    <section className="provider-card glass"><h1>Strata</h1>
      <p>{failure || (!session ? "Checking sign-in…" : "Sign in to open your well map.")}</p>
      {failure ? <button onClick={() => void load()}>Retry</button> : session && (session.auth_configured
        ? <a href="/auth/login">Sign in with Auth0</a> : <p role="status">Sign-in setup is not complete.</p>)}
    </section>
  </div>;
  return <>{children}{!session && <nav className="provider-tools glass" aria-label="Account">
    <span role="status">{failure || "Checking sign-in…"}</span>
    {failure && <button onClick={() => void load()}>Retry</button>}
  </nav>}{session?.auth_configured && <nav className="provider-tools glass" aria-label="Account">
    {session.user ? <form action="/auth/logout" method="post"><span>{session.user.name || "Signed in"}</span>
      <button type="submit">Sign out</button></form> : <a href="/auth/login">Sign in for Gemini notes</a>}
  </nav>}</>;
}
