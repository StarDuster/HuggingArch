/** Last Inspector destination in this browser tab.
 *
 * Top-level pages are client-side routes, but the Inspector owns additional
 * state in its hash.  A fixed `/` nav link discarded that state every time a
 * reader left the page.  Keep the destination session-scoped: a new browser
 * tab still starts at the ordinary landing page, while navigation in this tab
 * returns to the exact model/view it left.
 */
export const INSPECTOR_VISIT_EVENT = "ha:inspector-visit";
const INSPECTOR_VISIT_KEY = "ha:last-inspector-url";

function isInspectorHref(value: string): boolean {
  try {
    const url = new URL(value, window.location.origin);
    return url.origin === window.location.origin
      && (url.pathname === "/" || url.pathname === "/inspector.html");
  } catch {
    return false;
  }
}

export function currentHref(): string {
  return `${window.location.pathname}${window.location.search}${window.location.hash}`;
}

export function readLastInspectorHref(): string {
  try {
    const value = sessionStorage.getItem(INSPECTOR_VISIT_KEY) || "";
    return isInspectorHref(value) ? value : "/";
  } catch {
    return "/";
  }
}

/** Store and announce a destination. The event matters because Inspector
 * updates its hash with history.replaceState, which React Router does not
 * publish as a location change by itself. */
export function rememberInspectorHref(value = currentHref()): string {
  if (!isInspectorHref(value)) return readLastInspectorHref();
  try { sessionStorage.setItem(INSPECTOR_VISIT_KEY, value); } catch { /* unavailable storage */ }
  window.dispatchEvent(new CustomEvent<string>(INSPECTOR_VISIT_EVENT, { detail: value }));
  return value;
}
