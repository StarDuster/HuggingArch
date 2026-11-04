/** What a page shows while the server computes a view it has never served:
 *  a label, an amber bar — determinate when the server's 503s say how far
 *  the work got (`progress.done` of `progress.total` jobs), an indeterminate
 *  slide until they do — a little rocket riding the bar's tip, and an
 *  optional note under it. `unit` names what the count counts. `compact` is the inline mini bar for a status
 *  line (no rocket there: it would overlap the line above). */
export function ComputeProgress({ label, progress, unit, note, compact }: {
  label: string;
  progress?: { done: number; total: number };
  unit?: string;
  note?: string;
  compact?: boolean;
}) {
  const p = progress && progress.total > 0 ? progress : undefined;
  const pct = p ? Math.min(100, (100 * p.done) / p.total) : null;
  // The rocket floats above the bar, so it lives outside the track's
  // overflow-hidden: an outer span places it at the fill's tip (and carries
  // the slide when the count is unknown), an inner one bobs and points it
  // along the bar — one transform each, so neither clobbers the other.
  const rocket = compact ? null : (
    <span
      aria-hidden
      className={`progress-rocket${pct === null ? " progress-rocket-slide" : ""}`}
      style={pct === null ? undefined : { left: `${pct}%` }}
    >
      <span className="progress-rocket-body">🚀</span>
    </span>
  );
  const bar = (
    <div className="relative">
      <div
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        {...(p ? { "aria-valuemax": p.total, "aria-valuenow": p.done } : {})}
        className={`rounded-full bg-gray-200 overflow-hidden ${compact ? "h-1 w-20" : "h-1.5 w-full"}`}
      >
        {p ? (
          <div
            className="h-full bg-amber-500 rounded-full transition-[width]"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="h-full bg-amber-500 rounded-full progress-indeterminate" />
        )}
      </div>
      {rocket}
    </div>
  );
  if (compact) {
    return (
      <span className="inline-flex items-center gap-1.5">
        <span>{label}</span>
        {bar}
        {p ? <span className="tabular-nums">{p.done} / {p.total}{unit ? ` ${unit}` : ""}</span> : null}
      </span>
    );
  }
  return (
    <div className="inline-block w-full max-w-md text-left">
      <div className="flex items-baseline gap-2 text-xs text-gray-500">
        <span>{label}</span>
        {p ? <span className="tabular-nums">{p.done} / {p.total}{unit ? ` ${unit}` : ""}</span> : null}
      </div>
      <div className="mt-3.5">{bar}</div>
      {note ? <div className="mt-1 text-11 text-gray-400">{note}</div> : null}
    </div>
  );
}
