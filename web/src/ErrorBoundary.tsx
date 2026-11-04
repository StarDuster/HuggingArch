import { Component, type ReactNode } from "react";

/** A dynamic import that failed because the page predates a rebuild: the
 *  chunk it names (content-hashed) is gone. Each engine words it its own way. */
export function isStaleChunk(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed/i.test(message);
}

// When the page last reloaded itself for a stale chunk. Another stale chunk
// within minutes of that reload is not staleness a reload fixes (the chunk is
// missing from the new build too) — show it instead of reloading in a loop.
const RELOADED_AT = "ha-stale-chunk-reload";

type Props = {
  children: ReactNode;
  /** A slot inside a page (a chart): a one-line note in its place, the page
   *  around it untouched. Otherwise the note fills the page area. */
  compact?: boolean;
  height?: number;
};

/** One failure under it must not blank the whole page. A stale chunk reloads
 *  the page once (the URL carries the page state, and the new build has the
 *  chunk); any other error, or a stale chunk again right after reloading,
 *  says what failed and offers a reload. */
export class ErrorBoundary extends Component<Props, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: unknown) {
    if (!isStaleChunk(error)) return;
    const last = Number(sessionStorage.getItem(RELOADED_AT) || 0);
    if (Date.now() - last < 5 * 60_000) return;
    sessionStorage.setItem(RELOADED_AT, String(Date.now()));
    window.location.reload();
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const stale = isStaleChunk(error);
    const what = stale
      ? "The site was updated since this page loaded."
      : `Something here failed to render: ${error.message}`;
    const reload = (
      <button type="button" onClick={() => window.location.reload()}
        className="ml-2 text-xs px-2.5 py-1 rounded-full border border-amber-500 text-amber-700 hover:bg-amber-50">
        Reload
      </button>
    );
    if (this.props.compact) {
      return (
        <div data-render-error className="text-xs text-gray-500 text-center py-8"
          style={{ height: this.props.height }}>
          {what}{reload}
        </div>
      );
    }
    return (
      <div data-render-error className="flex-1 px-[var(--page-gutter)] py-10">
        <div className="card p-6 text-sm text-gray-700">{what}{reload}</div>
      </div>
    );
  }
}
