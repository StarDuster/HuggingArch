import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { loadAuth, signOut, type AuthState } from "./auth";
import { AvatarFace } from "./Avatar";

const TABS = [
  { key: "inspector", href: "/", label: "Inspector",
    icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" },
  { key: "gallery", href: "/gallery.html", label: "Gallery",
    icon: "M4 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zM14 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" },
  { key: "inference", href: "/inference.html", label: "Inference",
    icon: "M13 10V3L4 14h7v7l9-11h-7z" },
  { key: "playground", href: "/playground.html", label: "Playground",
    icon: "M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" },
  { key: "fusion", href: "/fusion.html", label: "Fusion",
    icon: "M4 7h16M4 12h16M4 17h10" },
  { key: "corpus", href: "/kernel-corpus.html", label: "Kernel Bench",
    icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" },
] as const;

// Labels never wrap: a two-line tab, tagline or status reads as broken. The
// tagline waits for 2xl, where the row has room for it on one line.
const BASE = "top-tab h-full flex items-center px-3 text-base font-medium whitespace-nowrap border-b-2";
const ACTIVE = `${BASE} border-amber-500 text-amber-700`;
const IDLE = `${BASE} border-transparent text-gray-500 hover:text-gray-800`;

export function TopNav({ active }: { active: (typeof TABS)[number]["key"] | null }) {
  // Below lg the tab row folds into a menu button with a dropdown panel.
  const [menuOpen, setMenuOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => { setMenuOpen(false); }, [pathname]);
  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setMenuOpen(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [menuOpen]);
  return (
    <header id="top-nav" className="bg-white border-b border-gray-200 sticky top-0 z-30">
      <div className="page-shell">
        <div className="flex items-center h-14 gap-3 lg:gap-6">
          <Link to="/" className="flex items-center gap-2.5 shrink-0 group">
            <img src="/static/logo.png" alt="HuggingArch" className="w-8 h-8 rounded-lg shadow-sm" />
            <span className="text-lg font-bold text-gray-900 group-hover:text-amber-600 transition">HuggingArch</span>
          </Link>
          <nav className="hidden lg:flex items-center gap-1 h-full">
            {TABS.map((t) => (
              <Link key={t.key} to={t.href} className={t.key === active ? ACTIVE : IDLE} data-tab={t.key}>
                <TabIcon d={t.icon} />
                {t.label}
              </Link>
            ))}
          </nav>
          <div className="flex-1" />
          <span className="hidden 2xl:block whitespace-nowrap text-xs text-gray-400">Understand before you deploy.</span>
          <ApiStatus />
          <AuthSlot />
          <button
            id="nav-menu-btn"
            type="button"
            className="lg:hidden w-10 h-10 flex items-center justify-center rounded-lg text-gray-600 hover:bg-gray-100"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
            aria-controls="nav-menu"
            onClick={() => setMenuOpen((o) => !o)}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d={menuOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"} />
            </svg>
          </button>
        </div>
      </div>
      {menuOpen && (
        <nav id="nav-menu" className="lg:hidden absolute inset-x-0 top-full bg-white border-b border-gray-200 shadow-lg">
          <div className="page-shell py-2 grid grid-cols-1 sm:grid-cols-2 gap-1">
            {TABS.map((t) => (
              <Link
                key={t.key}
                to={t.href}
                data-menu-tab={t.key}
                aria-current={t.key === active ? "page" : undefined}
                className={`flex items-center px-3 py-3 rounded-lg text-base font-medium ${
                  t.key === active ? "bg-amber-50 text-amber-700" : "text-gray-700 hover:bg-gray-50"}`}
                onClick={() => setMenuOpen(false)}
              >
                <TabIcon d={t.icon} />
                {t.label}
              </Link>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}

/** How often the nav asks `/health`, and how long one ask may take. It asks
 *  again whenever the tab comes back into view. */
const HEALTH_EVERY_MS = 45_000;
const HEALTH_TIMEOUT_MS = 8_000;

/** Whether the backend answers `/health` — asked on load, every
 *  `HEALTH_EVERY_MS` while the tab is visible, and on returning to the tab. */
function ApiStatus() {
  const [up, setUp] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    const check = async () => {
      if (document.visibilityState === "hidden") return;
      let ok = false;
      try {
        const r = await fetch("/health", { cache: "no-store", signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) });
        ok = r.ok;
      } catch { /* unreachable: stays false */ }
      if (alive) setUp(ok);
    };
    void check();
    const timer = setInterval(() => void check(), HEALTH_EVERY_MS);
    const onShow = () => { if (document.visibilityState === "visible") void check(); };
    document.addEventListener("visibilitychange", onShow);
    window.addEventListener("focus", onShow);
    return () => {
      alive = false;
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onShow);
      window.removeEventListener("focus", onShow);
    };
  }, []);
  const [dot, text] = up === null ? ["bg-gray-300", "Checking API…"]
    : up ? ["bg-emerald-500 animate-pulse", "API Ready"]
    : ["bg-red-500", "API unavailable"];
  return (
    <div
      id="api-status"
      data-api-up={up === null ? undefined : String(up)}
      role="status"
      title={up === false ? "The analysis server is not answering; results may fail to load. Retrying automatically." : undefined}
      className={`hidden sm:flex lg:hidden xl:flex items-center gap-1.5 whitespace-nowrap text-xs ${up === false ? "text-red-600" : "text-gray-500"}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      {text}
    </div>
  );
}

function TabIcon({ d }: { d: string }) {
  return (
    <svg className="w-4 h-4 mr-1.5 opacity-60" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d={d} />
    </svg>
  );
}

const AUTH_MENU_ITEM = "flex items-center gap-2.5 w-full px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 hover:text-gray-900";
const PROFILE_ICON = "M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z";
const WORKSPACE_ICON = "M3 7a2 2 0 012-2h4l2 2h6a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z";
const ADMIN_ICON = "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z";
const SIGNOUT_ICON = "M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9";

function AuthMenuIcon({ d }: { d: string }) {
  return (
    <svg
      className="w-4 h-4 shrink-0 text-gray-400"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d={d} />
    </svg>
  );
}

/** Sign in when auth is on or still unknown — never drop the entry silently. */
function AuthSlot() {
  const [state, setState] = useState<AuthState | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    void loadAuth().then(setState);
  }, []);
  useEffect(() => {
    if (!menuOpen) return;
    const close = () => setMenuOpen(false);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [menuOpen]);
  if (state?.authEnabled === false) return null;
  const user = state?.user;
  const profile = state?.profile || {};
  const name = profile.username || user?.email || "";
  const avatarUrl = profile.avatar_url || user?.user_metadata?.avatar_url || "";
  return (
    <div id="auth-widget" className="flex items-center gap-2">
      <a
        id="auth-signin"
        href="/auth/login.html"
        className={`text-xs px-3 py-1.5 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 ${user ? "hidden" : ""}`}
      >
        Sign in
      </a>
      <div id="auth-userbox" className={`relative ${user ? "" : "hidden"}`}>
        <button
          id="auth-userbtn"
          type="button"
          className="flex items-center gap-2 text-xs text-gray-700 hover:text-amber-700 px-2 py-1.5 rounded-lg hover:bg-gray-50"
          onClick={(e) => {
            e.stopPropagation();
            setMenuOpen((o) => !o);
          }}
        >
          <AvatarFace id="auth-avatar" url={avatarUrl} text={name} size="w-6 h-6 text-xs" />
          <span id="auth-name" className="hidden md:inline lg:hidden xl:inline max-w-[7rem] whitespace-nowrap text-ellipsis overflow-clip [overflow-clip-margin:4px]" title={name}>{name}</span>
          <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        <div
          id="auth-menu"
          className={`${menuOpen ? "" : "hidden"} absolute right-0 top-full mt-1 w-48 bg-white border border-gray-200 rounded-lg shadow-lg z-40 py-1`}
        >
          <a href="/account/profile.html" className={AUTH_MENU_ITEM}>
            <AuthMenuIcon d={PROFILE_ICON} />
            Profile
          </a>
          <a href="/account/workspace.html" className={AUTH_MENU_ITEM}>
            <AuthMenuIcon d={WORKSPACE_ICON} />
            Workspace
          </a>
          <a
            id="auth-admin-link"
            href="/admin/"
            className={`${AUTH_MENU_ITEM} ${state?.isAdmin ? "" : "hidden"}`}
          >
            <AuthMenuIcon d={ADMIN_ICON} />
            Admin
          </a>
          <hr className="border-gray-100 my-1" />
          <button
            id="auth-signout"
            type="button"
            className={AUTH_MENU_ITEM}
            onClick={() => void signOut().then(() => window.location.reload())}
          >
            <AuthMenuIcon d={SIGNOUT_ICON} />
            Sign out
          </button>
        </div>
      </div>
    </div>
  );
}
