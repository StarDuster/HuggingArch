// A background agent run (spec review or fix) as the drawers show it: spinner,
// elapsed time since the server started it, the agent's latest step and its
// activity so far. Shared by the account page and the admin drawer so a run
// looks the same wherever it is watched.

export function Spinner() {
  return (
    <svg className="w-3.5 h-3.5 animate-spin shrink-0" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function fmtElapsed(ms: number): string {
  const sec = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(sec / 60);
  return m ? `${m}m ${String(sec % 60).padStart(2, "0")}s` : `${sec}s`;
}

export type AgentJobKind = "review" | "fix" | "pr" | "recheck";

const JOB_TITLE: Record<AgentJobKind, string> = {
  review: "Agent review in progress",
  fix: "Agent fix in progress",
  pr: "Pull request in progress",
  recheck: "Re-check in progress",
};

export function AgentJobCard({ kind, startedAt, steps, now, onStop }: {
  kind: AgentJobKind; startedAt: number; steps: string[]; now: number;
  onStop?: () => void;
}) {
  const last = steps[steps.length - 1];
  return (
    <div className="border border-amber-200 bg-amber-50 rounded-lg px-3 py-2.5 mb-2">
      <div className="flex items-center gap-2 text-xs font-medium text-amber-800">
        <Spinner />
        <span>{JOB_TITLE[kind]}</span>
        <span className="font-normal text-amber-700">· {fmtElapsed(now - startedAt)}</span>
      </div>
      <div className="mt-1.5 text-11 font-mono text-gray-700 truncate" title={last || ""}>
        {last || "Starting…"}
      </div>
      {steps.length > 1 && (
        <details className="mt-1.5">
          <summary className="text-11 text-gray-500 cursor-pointer select-none">Activity ({steps.length})</summary>
          <pre className="mt-1 text-11 font-mono text-gray-600 whitespace-pre-wrap max-h-48 overflow-auto m-0">{steps.join("\n")}</pre>
        </details>
      )}
      <div className="mt-1.5 flex items-center justify-between gap-2">
        <div className="text-11 text-gray-500">
          {kind === "fix" ? "Re-reviews automatically when done. " : ""}
          {kind === "pr"
            ? "Measuring the corpus rows against main, then writing the PR. A few minutes; it keeps running if you close this panel."
            : kind === "recheck"
              ? "Validating the saved spec against the current engine; no agent runs."
              : "Usually a few minutes. You can close this panel — it keeps running."}
        </div>
        {onStop && kind !== "pr" && kind !== "recheck" ? (
          <button
            type="button"
            className="shrink-0 text-11 px-2 py-0.5 rounded border border-red-200 text-red-600 hover:bg-red-50"
            onClick={onStop}
          >
            Stop
          </button>
        ) : null}
      </div>
    </div>
  );
}
