/** Browser auth for the React chrome.
 *
 * The session lives in two httpOnly cookies (`ha_at` / `ha_rt`) set by the
 * backend (`/api/auth/*`); JavaScript never sees a token. Signup, password
 * recovery and the OAuth authorize redirect still talk to Supabase GoTrue
 * directly — those flows are unauthenticated by definition. Same-origin
 * fetches carry the cookies automatically.
 */

const LEGACY_STORAGE_KEY = "huggingarch.auth.token";

/** Custom header the backend requires on /api/auth/* mutations — proof the
 *  call came from our SPA, not a cross-site form (login CSRF guard). */
const SESSION_HEADER = { "X-HA-Auth": "1" } as const;

export type AuthUser = {
  id?: string;
  email?: string;
  user_metadata?: { avatar_url?: string; full_name?: string; username?: string };
};

export type AuthProfile = {
  username?: string | null;
  avatar_url?: string | null;
  full_name?: string | null;
  bio?: string | null;
  github_handle?: string | null;
};

export type AuthState = {
  authEnabled: boolean;
  user: AuthUser | null;
  profile: AuthProfile | null;
  isAdmin: boolean;
};

type MeResponse = {
  authEnabled: boolean;
  user: AuthUser | null;
  profile: AuthProfile | null;
  isAdmin: boolean;
};

function siteConfig(): { authEnabled: boolean; url: string; anon: string } {
  const cfg = window.SITE_CONFIG;
  return {
    authEnabled: cfg?.authEnabled !== false,
    url: (cfg?.supabase?.url || "").replace(/\/$/, ""),
    anon: cfg?.supabase?.anon_key || "",
  };
}

async function ensureSiteConfig(): Promise<void> {
  if (typeof window.SITE_CONFIG?.authEnabled === "boolean") return;
  try {
    const t = await (await fetch("/site-config.js")).text();
    const m = t.match(/authEnabled:\s*(true|false)/);
    const url = t.match(/url:\s*'([^']*)'/) || t.match(/url:\s*"([^"]*)"/);
    const anon = t.match(/anon_key:\s*'([^']*)'/) || t.match(/anon_key:\s*"([^"]*)"/);
    window.SITE_CONFIG = {
      authEnabled: m ? m[1] === "true" : true,
      supabase: { url: url?.[1] || "", anon_key: anon?.[1] || "" },
    };
  } catch {
    window.SITE_CONFIG = { authEnabled: true, supabase: { url: "", anon_key: "" } };
  }
}

/** One in-flight refresh at a time, per tab AND across tabs: concurrent
 *  refreshes race GoTrue's refresh-token rotation, and the loser of that
 *  race can invalidate the winner's brand-new cookie (last Set-Cookie
 *  wins). The Web Locks API serializes refreshes across tabs where
 *  available; older browsers fall back to two localStorage markers — a
 *  lock with a timestamp (crash-safe: it expires) and the last outcome. */
let _refreshing: Promise<boolean> | null = null;

const REFRESH_LOCK_KEY = "huggingarch.auth.refresh.lock";
const REFRESH_DONE_KEY = "huggingarch.auth.refresh.done";
const LOCK_STALE_MS = 15_000;
// Comfortably past the backend's 10s GoTrue timeout, so a slow-but-alive
// sibling refresh isn't mistaken for a crashed one.
const WAIT_TIMEOUT_MS = 12_000;

type LocksAPI = {
  request: (name: string, cb: () => Promise<boolean>) => Promise<boolean>;
};

function _locks(): LocksAPI | null {
  const nav = typeof navigator !== "undefined"
    ? (navigator as Navigator & { locks?: LocksAPI })
    : undefined;
  return typeof nav?.locks?.request === "function" ? nav.locks : null;
}

function _claimRefreshLock(): number | null {
  try {
    const now = Date.now();
    const held = Number(localStorage.getItem(REFRESH_LOCK_KEY) || 0);
    if (now - held < LOCK_STALE_MS) return null;
    localStorage.setItem(REFRESH_LOCK_KEY, String(now));
    const read = localStorage.getItem(REFRESH_LOCK_KEY);
    // Non-persistent storage (private mode, test stubs) can't coordinate
    // — proceed as sole refresher. If a sibling's fresh timestamp survived
    // our write, it holds the lock and we wait for its outcome.
    if (read === null || read === String(now)) return now;
    return now - Number(read || 0) < LOCK_STALE_MS ? null : now;
  } catch {
    return Date.now(); // private mode: fall back to per-tab single-flight
  }
}

function _publishRefreshOutcome(claim: number, ok: boolean): void {
  try {
    localStorage.setItem(REFRESH_DONE_KEY, JSON.stringify({ claim, ok }));
  } catch { /* private mode */ }
}

function _readRefreshOutcome(since: number): boolean | null {
  try {
    const raw = localStorage.getItem(REFRESH_DONE_KEY);
    if (!raw) return null;
    const done = JSON.parse(raw) as { claim?: number; ok?: boolean };
    return typeof done.claim === "number" && done.claim >= since ? !!done.ok : null;
  } catch {
    return null;
  }
}

async function _waitForSiblingRefresh(since: number): Promise<boolean> {
  const deadline = Date.now() + WAIT_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 300));
    const outcome = _readRefreshOutcome(since);
    if (outcome !== null) return outcome;
  }
  return false;
}

async function _refreshFetch(): Promise<boolean> {
  return fetch("/api/auth/refresh", { method: "POST", headers: SESSION_HEADER })
    .then((r) => r.ok)
    .catch(() => false);
}

async function _refreshCoordinated(): Promise<boolean> {
  const locks = _locks();
  if (locks) {
    // Serialized across tabs by the platform; a tab that queued behind a
    // sibling finds the just-published outcome and skips its own fetch
    // (the cookie jar is shared, so the sibling's result applies to us).
    return locks.request("huggingarch.auth.refresh", async () => {
      const recent = _readRefreshOutcome(Date.now() - LOCK_STALE_MS);
      if (recent !== null) return recent;
      const ok = await _refreshFetch();
      _publishRefreshOutcome(Date.now(), ok);
      return ok;
    });
  }
  const claim = _claimRefreshLock();
  if (claim === null) {
    // A sibling tab holds the lock — its Set-Cookie lands in the shared
    // cookie jar, so its outcome is our outcome.
    return _waitForSiblingRefresh(Date.now() - LOCK_STALE_MS);
  }
  const ok = await _refreshFetch();
  _publishRefreshOutcome(claim, ok);
  return ok;
}

export function refreshSession(): Promise<boolean> {
  if (!_refreshing) {
    _refreshing = _refreshCoordinated().finally(() => { _refreshing = null; });
  }
  return _refreshing;
}

/** One-time handover for sessions stored by pre-cookie builds: refresh the
 *  legacy localStorage token against GoTrue, exchange the fresh pair for
 *  cookies, and drop the local copy. Any failure just signs the user out. */
let _migrated = false;

async function migrateLegacySession(): Promise<void> {
  if (_migrated) return;
  _migrated = true;
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(LEGACY_STORAGE_KEY);
  } catch { /* private mode */ }
  if (!raw) return;
  try {
    const parsed = JSON.parse(raw) as {
      access_token?: string;
      refresh_token?: string;
      currentSession?: { access_token?: string; refresh_token?: string };
    };
    const legacy = parsed.access_token ? parsed : parsed.currentSession;
    const rt = legacy?.refresh_token;
    if (!rt) {
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      return;
    }
    await ensureSiteConfig();
    const { url, anon } = siteConfig();
    if (!url || !anon) return;
    const r = await fetch(`${url}/auth/v1/token?grant_type=refresh_token`, {
      method: "POST",
      headers: { apikey: anon, "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: rt }),
    });
    if (!r.ok) {
      // Definitive rejection is useless baggage; a provider 5xx is worth
      // retrying on the next load.
      if (r.status < 500) localStorage.removeItem(LEGACY_STORAGE_KEY);
      return;
    }
    const next = (await r.json()) as {
      access_token?: string; refresh_token?: string; expires_in?: number;
    };
    if (!next.access_token) return;
    const x = await fetch("/api/auth/exchange", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...SESSION_HEADER },
      body: JSON.stringify({
        access_token: next.access_token,
        refresh_token: next.refresh_token,
        expires_in: next.expires_in,
      }),
    });
    // Drop the legacy copy only once the session safely lives in cookies.
    if (x.ok) localStorage.removeItem(LEGACY_STORAGE_KEY);
  } catch { /* signed out */ }
}

/** GET /api/auth/me with one refresh-and-retry on an expired cookie.
 *  The server 401s only when a refresh cookie exists but the access cookie
 *  no longer authenticates, so 401 + failed refresh = a session that died
 *  (`expired`); a plain signed-out browser gets 200 with user=null.
 *  `me` null on a network/server failure (callers treat as signed out). */
async function fetchMe(): Promise<{ me: MeResponse | null; expired: boolean }> {
  try {
    let r = await fetch("/api/auth/me");
    if (r.status === 401) {
      if (!(await refreshSession())) return { me: null, expired: true };
      r = await fetch("/api/auth/me");
      if (r.status === 401) return { me: null, expired: true };
    }
    if (!r.ok) return { me: null, expired: false };
    return { me: (await r.json()) as MeResponse, expired: false };
  } catch {
    return { me: null, expired: false };
  }
}

async function readDetail(r: Response, fallback: string): Promise<Error> {
  const body = (await r.json().catch(() => ({}))) as {
    detail?: string; error_description?: string; msg?: string;
  };
  const msg = body.detail || body.error_description || body.msg;
  return new Error(typeof msg === "string" && msg ? msg : fallback);
}

export async function signInWithPassword(email: string, password: string): Promise<void> {
  await ensureSiteConfig();
  const { authEnabled, url, anon } = siteConfig();
  if (!authEnabled) throw new Error("Auth is not configured on this server.");
  if (!url || !anon) throw new Error("Supabase not ready.");
  const r = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...SESSION_HEADER },
    body: JSON.stringify({ email, password }),
  });
  if (!r.ok) throw await readDetail(r, `HTTP ${r.status}`);
}

export async function signUpWithPassword(email: string, password: string, username?: string, next = "/"): Promise<void> {
  await ensureSiteConfig();
  const { url, anon, authEnabled } = siteConfig();
  if (!authEnabled) throw new Error("Auth is not configured on this server.");
  if (!url || !anon) throw new Error("Supabase not ready.");
  const redirect = `${window.location.origin}/auth/verify.html?next=${encodeURIComponent(next)}`;
  const q = new URLSearchParams({ redirect_to: redirect });
  const r = await fetch(`${url}/auth/v1/signup?${q.toString()}`, {
    method: "POST",
    headers: { apikey: anon, "Content-Type": "application/json" },
    body: JSON.stringify({
      email,
      password,
      data: username ? { username } : {},
    }),
  });
  const body = await r.json() as { error_description?: string; msg?: string };
  if (!r.ok) throw new Error(body.error_description || body.msg || `HTTP ${r.status}`);
}

/** Implicit-flow hash from OAuth / magic-link / recovery. Hands the tokens
 *  to the backend for cookies, then scrubs the URL. */
export async function consumeAuthHash(): Promise<{ session: boolean; type: string | null }> {
  const raw = (window.location.hash || "").replace(/^#\/?/, "");
  const p = new URLSearchParams(raw);
  const type = p.get("type");
  const access_token = p.get("access_token");
  if (!access_token) return { session: false, type };
  const r = await fetch("/api/auth/exchange", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...SESSION_HEADER },
    body: JSON.stringify({
      access_token,
      refresh_token: p.get("refresh_token") || undefined,
      expires_in: Number(p.get("expires_in") || 0) || undefined,
    }),
  }).catch(() => null);
  // Scrub the fragment only once the tokens are safely in cookies — a
  // failed exchange would otherwise destroy their only copy.
  if (r?.ok) {
    history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  return { session: !!r?.ok, type };
}

export async function requestPasswordReset(email: string): Promise<void> {
  await ensureSiteConfig();
  const { url, anon, authEnabled } = siteConfig();
  if (!authEnabled || !url || !anon) throw new Error("Supabase not ready.");
  const redirect = `${window.location.origin}/auth/reset-password.html`;
  const q = new URLSearchParams({ redirect_to: redirect });
  const r = await fetch(`${url}/auth/v1/recover?${q.toString()}`, {
    method: "POST",
    headers: { apikey: anon, "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({})) as { error_description?: string; msg?: string };
    throw new Error(body.error_description || body.msg || `HTTP ${r.status}`);
  }
}

export async function updatePassword(password: string): Promise<void> {
  const send = () => fetch("/api/auth/password", {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...SESSION_HEADER },
    body: JSON.stringify({ password }),
  });
  let r = await send();
  if (r.status === 401 && await refreshSession()) r = await send();
  if (!r.ok) throw await readDetail(r, `HTTP ${r.status}`);
}

export async function oauthAuthorizeUrl(provider: "github" | "google", next: string): Promise<string> {
  await ensureSiteConfig();
  const { url, anon, authEnabled } = siteConfig();
  if (!authEnabled || !url || !anon) throw new Error("Supabase not ready.");
  const redirect = `${window.location.origin}/auth/verify.html?next=${encodeURIComponent(next)}`;
  const q = new URLSearchParams({
    provider,
    redirect_to: redirect,
  });
  return `${url}/auth/v1/authorize?${q.toString()}`;
}

/** Inspector routing: anonymous vs signed-in. `expired` means a session
 *  existed but the refresh failed — callers should stop and ask the user to
 *  sign in again rather than silently downgrade to the anonymous endpoints. */
export async function authContext(): Promise<{
  authEnabled: boolean;
  loggedIn: boolean;
  expired: boolean;
}> {
  await migrateLegacySession();
  const { me, expired } = await fetchMe();
  if (!me || !me.authEnabled) return { authEnabled: false, loggedIn: false, expired };
  return { authEnabled: true, loggedIn: !!me.user, expired: false };
}

export async function loadAuth(): Promise<AuthState> {
  await migrateLegacySession();
  const { me } = await fetchMe();
  if (!me || !me.authEnabled) {
    return { authEnabled: me?.authEnabled ?? true, user: null, profile: null, isAdmin: false };
  }
  return { authEnabled: true, user: me.user, profile: me.profile, isAdmin: me.isAdmin };
}

export async function signOut(): Promise<void> {
  try {
    await fetch("/api/auth/logout", { method: "POST", headers: SESSION_HEADER });
  } catch { /* still signed out locally once cookies expire */ }
}

/** Fire-and-forget. Outage must not block analyze / estimate. */
export function trackAnalyticsEvent(
  endpoint: string,
  model_id: string,
  duration_ms: number,
  metadata?: Record<string, unknown>,
): void {
  void fetch("/analytics/event", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ endpoint, model_id, duration_ms, metadata: metadata || {} }),
  }).catch(() => { /* swallow */ });
}

/** Send the browser to the sign-in page, returning here afterwards. Admin
 *  returns without the URL hash; the account pages keep it. */
export function loginRedirect(keepHash = true): void {
  const here = window.location.pathname + window.location.search + (keepHash ? window.location.hash : "");
  window.location.href = `/auth/login.html?next=${encodeURIComponent(here)}`;
}

/** JSON call to a signed-in endpoint. No session → sign-in page. A non-2xx
 *  response throws the FastAPI `detail` string (else the status text). */
export async function authJson<T>(
  method: string,
  path: string,
  body?: unknown,
  opts: { keepHash?: boolean } = {},
): Promise<T> {
  const send = () => fetch(path, {
    method,
    headers: {
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...(path.startsWith("/api/auth/") ? SESSION_HEADER : {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  let r = await send();
  if (r.status === 401 || r.status === 403) {
    if (await refreshSession()) r = await send();
  }
  if (r.status === 401) {
    loginRedirect(opts.keepHash);
    throw new Error("not signed in");
  }
  if (!r.ok) {
    const detail = ((await r.json().catch(() => ({}))) as { detail?: string }).detail || r.statusText;
    throw new Error(typeof detail === "string" ? detail : r.statusText);
  }
  return r.json() as Promise<T>;
}

/** Allowlisted admin RPC, proxied server-side with the caller's JWT. */
export async function adminRpc(name: string, args: Record<string, unknown>): Promise<void> {
  await authJson("POST", "/api/auth/admin-rpc", { name, args });
}

/** The caller's own rows as `/api/bench/calibration` reports them (read
 *  from Kernel Bench now): how many calibrate their estimates and how many the
 *  current validator refuses, and why; `measurement_scope` is the digest of
 *  the scope an estimate calibrating with them goes under. */
export type MyCalibration = {
  rows_used: number;
  rows_rejected: number;
  rejected_reasons: { reason: string; rows: number }[];
  first_rejected: { id: string; reason: string }[];
  snapshot: number;
  measurement_scope: string;
};

/** The caller's calibration corpus (`/api/bench/calibration`); `null` when
 *  unsigned / 401 after refresh. A non-2xx throws the server `detail`. */
export async function fetchMyCalibration(): Promise<MyCalibration | null> {
  const path = "/api/bench/calibration";
  let r = await fetch(path);
  if (r.status === 401 && await refreshSession()) {
    r = await fetch(path);
  }
  if (r.status === 401) return null;
  if (!r.ok) {
    const detail = ((await r.json().catch(() => ({}))) as { detail?: string }).detail || r.statusText;
    throw new Error(typeof detail === "string" ? detail : r.statusText);
  }
  return await r.json() as MyCalibration;
}
