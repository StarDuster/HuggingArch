import { useEffect, useState } from "react";
import { consumeAuthHash, loadAuth } from "../auth";

function nextUrl(): string {
  const p = new URLSearchParams(window.location.search);
  return p.get("next") || "/";
}

export function AuthVerify() {
  const [err, setErr] = useState("");

  useEffect(() => {
    document.title = "Verify · HuggingArch";
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      window.location.href = nextUrl();
    };
    const t = window.setTimeout(() => {
      if (!done) setErr("Timed out waiting for verification.");
    }, 6000);
    // What the link carried, read before the exchange can scrub it.
    const hash = new URLSearchParams((window.location.hash || "").replace(/^#\/?/, ""));
    const hadToken = hash.has("access_token");
    const linkError = hash.get("error_description") || hash.get("error");
    void (async () => {
      const { session } = await consumeAuthHash();
      if (session) {
        finish();
        return;
      }
      const a = await loadAuth();
      if (a.user) {
        finish();
        return;
      }
      if (done) return;
      done = true;
      window.clearTimeout(t);
      setErr(linkError
        ? `The sign-in link was refused: ${linkError.replace(/\+/g, " ")}. Request a new one.`
        : hadToken
          ? "The sign-in link could not be completed — it may have expired or already been used. Request a new one."
          : "This page completes sign-in from the link in your email, and this address carries no sign-in token. Open the link from the email again, or sign in.");
    })();
    return () => {
      window.clearTimeout(t);
      document.title = "HuggingArch";
    };
  }, []);

  if (err) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4 bg-gray-50">
        <div className="max-w-md w-full bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center">
          <div className="w-12 h-12 mx-auto rounded-full bg-red-100 flex items-center justify-center text-red-600">✕</div>
          <p className="mt-4 text-sm text-red-700">{err}</p>
          <a href="/auth/login.html" className="inline-block mt-4 text-sm text-amber-600 hover:underline">Back to sign in</a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-gray-50">
      <div className="max-w-md w-full bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center">
        <div className="w-12 h-12 mx-auto rounded-full border-4 border-amber-200 border-t-amber-500 animate-spin" />
        <p className="mt-4 text-sm text-gray-600">Completing sign-in…</p>
      </div>
    </div>
  );
}
