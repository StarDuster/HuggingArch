/** Badges for where a generation record stands and why a spec run failed.
 *
 *  The keys are the closed vocabularies of `backend/spec_worker/outcome.py`
 *  (`GenerationStatus`, `RunErrorKind`), served at
 *  `/api/spec/generation-vocabulary`; `tests/meta/test_generation_vocabulary.py`
 *  holds both tables to every value. A badge is chosen by the record's
 *  status or the run's `error_kind` — never by reading a message. */

export type Badge = {
  label: string;
  /** Tailwind classes of the pill. */
  cls: string;
  title: string;
};

export type StatusBadge = Badge & {
  /** A pending record: it waits on the engine or on a repair, beside the
   *  user's valid records, and is re-checked rather than reviewed. */
  pending?: boolean;
};

export const GENERATION_STATUS_BADGE: Record<string, StatusBadge> = {
  generated: { label: "generated", cls: "bg-gray-100 text-gray-600", title: "A valid spec: your working candidate." },
  contributed: { label: "contributed", cls: "bg-amber-100 text-amber-700", title: "A pull request was opened from this spec." },
  merged: { label: "merged", cls: "bg-green-100 text-green-700", title: "Its pull request merged." },
  rejected: { label: "rejected", cls: "bg-red-100 text-red-700", title: "An admin turned this spec down." },
  closed: { label: "closed", cls: "bg-slate-100 text-slate-600", title: "Its pull request closed unmerged; the spec is still valid." },
  blocked_by_engine: {
    label: "engine-blocked",
    cls: "bg-violet-100 text-violet-700",
    title: "The spec states the model correctly; the analysis engine does not yet support a capability it needs.",
    pending: true,
  },
  needs_repair: {
    label: "needs repair",
    cls: "bg-amber-100 text-amber-800",
    title: "A re-check found errors in the spec; Agent Fix can repair them.",
    pending: true,
  },
};

export const RUN_ERROR_KIND_BADGE: Record<string, Badge> = {
  spec_invalid: { label: "spec invalid", cls: "bg-red-100 text-red-700", title: "The spec failed validation or no longer loads." },
  spec_bundle_incomplete: { label: "bundle incomplete", cls: "bg-red-100 text-red-700", title: "The spec's bundle is missing or misnames a component." },
  spec_needs_declaration: { label: "needs declaration", cls: "bg-red-100 text-red-700", title: "The spec must declare a fact the current schema asks for." },
  bundle_invalid: { label: "bundle invalid", cls: "bg-red-100 text-red-700", title: "The saved bundle must be repaired before a PR." },
  blocked_by_engine: {
    label: "engine-blocked",
    cls: "bg-violet-100 text-violet-700",
    title: "Every remaining error is a capability the analysis engine does not yet support; the spec is correct.",
  },
  analysis_refusal_unattributed: {
    label: "unattributed refusal",
    cls: "bg-orange-100 text-orange-700",
    title: "The run ended on analysis refusals whose raise site names no owner.",
  },
  agent_unavailable: {
    label: "agent unavailable",
    cls: "bg-amber-100 text-amber-800",
    title: "No configured agent backend could serve the run (auth, quota, no model configured, 5xx, no connection). The spec was not judged.",
  },
  timed_out: {
    label: "timed out",
    cls: "bg-amber-100 text-amber-800",
    title: "The run spent its overall deadline. Its draft is kept and can be continued.",
  },
  killed: {
    label: "agent killed",
    cls: "bg-amber-100 text-amber-800",
    title: "A signal the run's timers did not send stopped the agent (a stop, the OOM killer, a restart).",
  },
  context_exhausted: {
    label: "context exhausted",
    cls: "bg-amber-100 text-amber-800",
    title: "The agent's session outgrew the model's context window. Its draft is kept and can be continued in a new session.",
  },
};

/** One engine refusal a blocked record met (`engine_gap_entries`). */
export type EngineGap = {
  capability: string;
  node?: string | null;
  site?: string | null;
  exception?: string | null;
  deployments: string[];
  stages: string[];
  detail?: string | null;
};

/** A status outside the vocabulary — or none at all — shows as what it is,
 *  never as another status. */
export function generationBadge(status: unknown): StatusBadge {
  const s = typeof status === "string" ? status : "";
  return GENERATION_STATUS_BADGE[s] || (s
    ? { label: s, cls: "bg-gray-100 text-gray-500", title: `Unknown status: ${s}` }
    : { label: "unknown", cls: "bg-gray-100 text-gray-500", title: "The record carries no status." });
}

export function isPendingStatus(status: unknown): boolean {
  return !!GENERATION_STATUS_BADGE[String(status || "")]?.pending;
}

/** `capability @ node (deployments; stages)`. */
export function gapLine(g: EngineGap): string {
  const where = g.node ? ` @ ${g.node}` : "";
  return `${g.capability}${where} (${(g.deployments || []).join(", ")}; ${(g.stages || []).join(", ")})`;
}

/** Every gap, one per line — a badge's hover text. */
export function gapsTitle(gaps: EngineGap[] | null | undefined): string {
  return (gaps || []).map(gapLine).join("\n");
}

/** The closed vocabularies as the backend serves them. */
export type GenerationVocabulary = {
  generation_status: string[];
  pending: string[];
  servable: string[];
  run_error_kind: string[];
};
