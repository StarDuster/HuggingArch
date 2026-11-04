// Shared sidebar nav for the account pages (profile / workspace / admin).
// Two groups: "You" (account-level pages; Admin sits last, admins only) and
// "Workspace" (spec generations, module variants, kernel benchmarks).
// Each panel lives on one page (PANEL_PAGE); panels in `localPanels` switch
// in place via hash (onPanelSelect), the rest link to their hosting page.

export const ACCOUNT_PANELS = ["profile", "credentials"] as const;

export const WORKSPACE_PANELS = ["generations", "modules", "benchmarks"] as const;

export type AccountPanel = (typeof ACCOUNT_PANELS)[number] | (typeof WORKSPACE_PANELS)[number];

export type AccountNavKey = AccountPanel | "admin";

const PANEL_PAGE: Record<AccountPanel, string> = {
  profile: "/account/profile.html",
  credentials: "/account/profile.html",
  generations: "/account/workspace.html",
  modules: "/account/workspace.html",
  benchmarks: "/account/workspace.html",
};

const ICONS: Record<AccountNavKey, string> = {
  profile: "M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z",
  credentials: "M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z",
  generations: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z",
  modules: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
  benchmarks: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z",
  admin: "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z",
};

const LABELS: Record<AccountNavKey, string> = {
  profile: "Profile",
  credentials: "Keys & tokens",
  generations: "Spec generations",
  modules: "My modules",
  benchmarks: "Kernel benchmarks",
  admin: "Admin",
};

const ITEM = "flex items-center gap-2 md:w-full text-left px-3 py-1.5 rounded-lg text-sm font-medium transition-colors";
const IDLE = "text-gray-600 hover:bg-gray-100";
const ACTIVE = "bg-amber-50 text-amber-700 font-semibold";
const GROUP = "hidden md:block px-3 pt-3 pb-1 text-10 uppercase tracking-wider text-gray-400";

function NavIcon({ d }: { d: string }) {
  return (
    <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d={d} />
    </svg>
  );
}

export function AccountNav({
  active,
  isAdmin,
  localPanels = [],
  onPanelSelect,
}: {
  active: AccountNavKey;
  isAdmin?: boolean;
  // Panels the current page hosts itself — they switch in place; every
  // other panel renders as a plain cross-page link.
  localPanels?: readonly string[];
  onPanelSelect?: (panel: AccountPanel) => void;
}) {
  // The entry for the page/panel you're already on renders as an inert
  // current marker — a self-link would trigger a full page reload and drop
  // in-memory state (selected batches, open tabs).
  const entry = (key: AccountNavKey, href: string, onClick?: () => void) =>
    active === key ? (
      <span key={key} aria-current="page" className={`${ITEM} ${ACTIVE}`}>
        <NavIcon d={ICONS[key]} />
        <span>{LABELS[key]}</span>
      </span>
    ) : (
      <a key={key} href={href} className={`${ITEM} ${IDLE}`} onClick={onClick}>
        <NavIcon d={ICONS[key]} />
        <span>{LABELS[key]}</span>
      </a>
    );
  const panelEntry = (key: AccountPanel) =>
    localPanels.includes(key) && onPanelSelect
      ? entry(key, `#${key}`, () => onPanelSelect(key))
      : entry(key, `${PANEL_PAGE[key]}#${key}`);
  return (
    <nav className="flex flex-wrap md:flex-col gap-0.5 text-sm" aria-label="Account sections">
      <div className={GROUP}>You</div>
      {panelEntry("profile")}
      {panelEntry("credentials")}
      {isAdmin && entry("admin", "/admin/")}
      <div className={GROUP}>Workspace</div>
      {panelEntry("generations")}
      {panelEntry("modules")}
      {panelEntry("benchmarks")}
    </nav>
  );
}
