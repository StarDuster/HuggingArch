import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { loadAuth, oauthAuthorizeUrl, signInWithPassword, signUpWithPassword } from "../auth";

function nextUrl(): string {
  const p = new URLSearchParams(window.location.search);
  return p.get("next") || "/";
}

export function AuthLogin() {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [kind, setKind] = useState<"err" | "ok" | "info" | "">("");
  const [disabled, setDisabled] = useState(false);

  useEffect(() => {
    document.title = "Sign in · HuggingArch";
    return () => { document.title = "HuggingArch"; };
  }, []);

  useEffect(() => {
    void loadAuth().then((a) => {
      if (!a.authEnabled) setDisabled(true);
      if (a.user) window.location.href = nextUrl();
    });
  }, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setMsg("");
    setKind("");
    try {
      if (mode === "signup") {
        await signUpWithPassword(email.trim(), password, username.trim() || undefined, nextUrl());
        setKind("ok");
        setMsg("Check your email to verify your account.");
      } else {
        await signInWithPassword(email.trim(), password);
        window.location.href = nextUrl();
        return;
      }
    } catch (err) {
      setKind("err");
      setMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const oauth = async (provider: "github" | "google") => {
    try {
      window.location.href = await oauthAuthorizeUrl(provider, nextUrl());
    } catch (err) {
      setKind("err");
      setMsg(err instanceof Error ? err.message : String(err));
    }
  };

  const tab = (id: "signin" | "signup", label: string) => (
    <button
      type="button"
      className={`flex-1 pb-3 text-sm font-medium border-b-2 ${mode === id ? "border-amber-500 text-amber-700" : "border-transparent text-gray-500 hover:text-gray-700"}`}
      onClick={() => { setMode(id); setMsg(""); setKind(""); }}
    >
      {label}
    </button>
  );

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12 bg-gray-50">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <a href="/" className="inline-flex items-center gap-2 text-2xl font-semibold text-gray-900 hover:text-amber-600">
            <img src="/static/logo.png" alt="" className="w-8 h-8" />
            HuggingArch
          </a>
          <p className="mt-2 text-sm text-gray-600">Sign in to save composed models &amp; track analyses.</p>
        </div>
        {disabled ? (
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-800">
            Auth is not configured on this server (no SUPABASE env vars). Sign-in is unavailable.{" "}
            <a href="/" className="underline font-medium">Back to app</a>.
          </div>
        ) : (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex border-b border-gray-200 mb-5">
              {tab("signin", "Sign in")}
              {tab("signup", "Create account")}
            </div>
            <button type="button" className="w-full flex items-center justify-center gap-2 px-4 py-2.5 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 mb-2" onClick={() => void oauth("github")}>
              Continue with GitHub
            </button>
            <button type="button" className="w-full flex items-center justify-center gap-2 px-4 py-2.5 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 mb-4" onClick={() => void oauth("google")}>
              Continue with Google
            </button>
            <div className="flex items-center gap-3 my-4">
              <div className="flex-1 h-px bg-gray-200" />
              <span className="text-xs text-gray-400">or</span>
              <div className="flex-1 h-px bg-gray-200" />
            </div>
            <form className="space-y-3" onSubmit={(e) => void submit(e)}>
              <label className="block text-xs font-medium text-gray-600">
                Email
                <input type="email" required autoComplete="email" className="mt-1 w-full px-3 py-2 text-sm border border-gray-300 rounded-lg" value={email} onChange={(e) => setEmail(e.target.value)} />
              </label>
              <label className="block text-xs font-medium text-gray-600">
                <span className="flex items-center justify-between">
                  Password
                  {mode === "signin" && (
                    <a href="/auth/reset-password.html" className="text-xs text-amber-600 hover:underline font-normal">Forgot?</a>
                  )}
                </span>
                <input type="password" required minLength={mode === "signup" ? 8 : undefined} autoComplete={mode === "signup" ? "new-password" : "current-password"} className="mt-1 w-full px-3 py-2 text-sm border border-gray-300 rounded-lg" value={password} onChange={(e) => setPassword(e.target.value)} />
              </label>
              {mode === "signup" && (
                <label className="block text-xs font-medium text-gray-600">
                  Username <span className="text-gray-400">(optional)</span>
                  <input type="text" autoComplete="username" className="mt-1 w-full px-3 py-2 text-sm border border-gray-300 rounded-lg" value={username} onChange={(e) => setUsername(e.target.value)} />
                </label>
              )}
              <button type="submit" disabled={busy} className="w-full px-4 py-2.5 bg-amber-500 hover:bg-amber-600 text-white text-sm font-medium rounded-lg disabled:opacity-60">
                {busy ? (mode === "signup" ? "Creating…" : "Signing in…") : (mode === "signup" ? "Create account" : "Sign in")}
              </button>
            </form>
            {msg && (
              <p className={`mt-4 text-sm rounded-lg px-3 py-2 ${kind === "err" ? "bg-red-50 text-red-700 border border-red-200" : kind === "ok" ? "bg-green-50 text-green-700 border border-green-200" : "bg-amber-50 text-amber-800 border border-amber-200"}`}>
                {msg}
              </p>
            )}
          </div>
        )}
        <p className="text-center text-xs text-gray-400 mt-6">
          <Link to="/" className="hover:text-amber-600">← Back to app</Link>
        </p>
      </div>
    </div>
  );
}
