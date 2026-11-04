import { useEffect, useState } from "react";
import { consumeAuthHash, requestPasswordReset, updatePassword } from "../auth";

export function AuthReset() {
  const [mode, setMode] = useState<"request" | "set">("request");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [kind, setKind] = useState<"err" | "ok" | "">("");

  useEffect(() => {
    document.title = "Reset password · HuggingArch";
    void consumeAuthHash().then(({ session, type }) => {
      if (type === "recovery" || session) setMode("set");
    });
    return () => { document.title = "HuggingArch"; };
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12 bg-gray-50">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <a href="/" className="inline-flex items-center gap-2 text-2xl font-semibold text-gray-900 hover:text-amber-600">
            <img src="/static/logo.png" alt="HuggingArch" className="w-8 h-8" />
            HuggingArch
          </a>
        </div>
        {mode === "request" ? (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h1 className="text-lg font-semibold text-gray-900 mb-1">Reset password</h1>
            <p className="text-sm text-gray-600 mb-5">Enter your email and we&apos;ll send a reset link.</p>
            <form className="space-y-3" onSubmit={(e) => {
              e.preventDefault();
              void (async () => {
                setBusy(true);
                setMsg("");
                try {
                  await requestPasswordReset(email.trim());
                  setKind("ok");
                  setMsg("Check your email for a reset link.");
                } catch (err) {
                  setKind("err");
                  setMsg(err instanceof Error ? err.message : String(err));
                } finally {
                  setBusy(false);
                }
              })();
            }}>
              <input type="email" required autoComplete="email" placeholder="you@example.com" className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg" value={email} onChange={(e) => setEmail(e.target.value)} />
              <button type="submit" disabled={busy} className="w-full px-4 py-2.5 bg-amber-500 hover:bg-amber-600 text-white text-sm font-medium rounded-lg disabled:opacity-60">
                {busy ? "Sending…" : "Send reset link"}
              </button>
            </form>
          </div>
        ) : (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h1 className="text-lg font-semibold text-gray-900 mb-1">Set a new password</h1>
            <form className="space-y-3 mt-4" onSubmit={(e) => {
              e.preventDefault();
              void (async () => {
                setBusy(true);
                setMsg("");
                try {
                  await updatePassword(password);
                  setKind("ok");
                  setMsg("Password updated. Redirecting…");
                  window.setTimeout(() => { window.location.href = "/"; }, 1200);
                } catch (err) {
                  setKind("err");
                  setMsg(err instanceof Error ? err.message : String(err));
                } finally {
                  setBusy(false);
                }
              })();
            }}>
              <input type="password" required minLength={8} placeholder="New password" className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg" value={password} onChange={(e) => setPassword(e.target.value)} />
              <button type="submit" disabled={busy} className="w-full px-4 py-2.5 bg-amber-500 hover:bg-amber-600 text-white text-sm font-medium rounded-lg disabled:opacity-60">
                {busy ? "Saving…" : "Save"}
              </button>
            </form>
          </div>
        )}
        {msg && (
          <div className={`mt-4 text-sm rounded-lg px-3 py-2 ${kind === "err" ? "bg-red-50 text-red-700 border border-red-200" : "bg-green-50 text-green-700 border border-green-200"}`}>{msg}</div>
        )}
        <p className="text-center text-xs text-gray-400 mt-6">
          <a href="/auth/login.html" className="hover:text-amber-600">← Back to sign in</a>
        </p>
      </div>
    </div>
  );
}
