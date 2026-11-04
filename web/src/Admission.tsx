// A PASS review's corpus admission (ADMIT §4.2), as the account page and the
// admin drawer show it. After a review passes, the service measures the
// record into every corpus-wide table the tests pin (fingerprints, SOL
// cells, checkpoint reconciliation, README counts, ...) against main and
// stores the report on the review row. This module turns that report into
// one summary line, an expandable table grouped by what a person has to do,
// and the Open PR button's state.

import { useState } from "react";
import { Spinner } from "./AgentJob";

export type KeyState = {
  state?: string;
  eligible?: boolean;
  waits_on?: string[];
  blocked_by?: string[];
};

export type AdmissionItem = {
  id: string;
  title?: string;
  scope?: string;
  key_states?: Record<string, KeyState>;
  expected_failing_tests?: string[];
};

export type AdmissionCheck = { id: string; title?: string; state?: string; violations?: string[] };
export type AdmissionDecision = { id: string; state?: string; detail?: string[]; suggestion?: unknown };

export type AdmissionReport = {
  status?: string;
  blocked_kind?: string | null;
  mode?: string;
  preview?: boolean;
  base?: { sha?: string | null; kind?: string };
  gates?: { G0?: string[]; G1?: Record<string, string[]>; G2?: string[] };
  items?: AdmissionItem[];
  checks?: AdmissionCheck[];
  decisions?: AdmissionDecision[];
  base_notes?: { id: string; detail?: string[] }[];
  errors?: string[];
  generated_at?: string;
  pr?: { url?: string; head_sha?: string; draft?: boolean; reconciled?: string } | null;
};

export type AdmissionEnvelope = {
  job?: "running" | "done" | "failed";
  mode?: string;
  report?: AdmissionReport | null;
  error?: string;
  updated_at?: string;
};

// What a review row carries about its admission.
export type AdmissionRow = {
  verdict?: string;
  admission?: AdmissionEnvelope | null;
  admission_unavailable?: boolean;
  bundle_current?: boolean;
};

// The one state a row's admission is in, for the summary line and the
// Open PR button. "none": a PASS with no report yet (it predates
// admissions, or the job never started).
export type AdmissionView =
  | { kind: "unsupported" }
  | { kind: "none" }
  | { kind: "running" }
  | { kind: "failed"; error: string }
  | { kind: "stale"; report: AdmissionReport | null }
  | { kind: "report"; report: AdmissionReport };

export function admissionView(row: AdmissionRow): AdmissionView {
  if (row.admission_unavailable) return { kind: "unsupported" };
  const env = row.admission;
  if (!env) return { kind: "none" };
  if (env.job === "running") return { kind: "running" };
  if (env.job === "failed") return { kind: "failed", error: env.error || "the admission job failed" };
  const report = env.report || null;
  if (row.bundle_current === false) return { kind: "stale", report };
  if (!report) return { kind: "none" };
  return { kind: "report", report };
}

const OPEN = new Set(["pending", "inconsistent"]);

function keysIn(item: AdmissionItem, states: string[]): string[] {
  return Object.entries(item.key_states || {})
    .filter(([, v]) => states.includes(String(v.state)))
    .map(([k]) => k);
}

// The report grouped the way a person acts on it.
export function admissionGroups(report: AdmissionReport) {
  const items = report.items || [];
  return {
    failedChecks: (report.checks || []).filter((c) => c.state === "failed"),
    pending: (report.decisions || []).filter((d) => OPEN.has(String(d.state))),
    withheld: items
      .map((it) => ({ item: it, keys: keysIn(it, ["withheld"]) }))
      .filter((x) => x.keys.length),
    ready: items
      .map((it) => ({ item: it, keys: keysIn(it, ["ready", "written"]) }))
      .filter((x) => x.keys.length),
    held: items
      .map((it) => ({ item: it, keys: keysIn(it, ["withheld_by_global"]) }))
      .filter((x) => x.keys.length),
    baseNotes: report.base_notes || [],
    info: (report.decisions || []).filter((d) => d.state === "info"),
  };
}

const STATUS_TEXT: Record<string, string> = {
  ready: "ready to admit",
  needs_review: "needs a maintainer's decision",
  blocked: "blocked",
  already_admitted: "already in the library",
  unavailable: "unavailable",
};

const STATUS_STYLE: Record<string, string> = {
  ready: "bg-green-100 text-green-800",
  needs_review: "bg-amber-100 text-amber-800",
  blocked: "bg-red-100 text-red-700",
  already_admitted: "bg-gray-100 text-gray-600",
  unavailable: "bg-gray-100 text-gray-600",
};

function sha7(s?: string | null) {
  return s ? s.slice(0, 7) : "?";
}

export function statusLabel(report: AdmissionReport): string {
  const s = String(report.status || "unavailable");
  const kind = report.blocked_kind === "base" ? " (main's baseline)"
    : report.blocked_kind === "candidate_or_rule" ? " (corpus checks)" : "";
  return (STATUS_TEXT[s] || s) + kind;
}

// The Open PR button for the record, from its latest PASS row.
export type PrGate = {
  label: string;
  disabled: boolean;
  reason: string;
  draftAnyway: boolean;  // an admin may open a draft past failed checks
};

export function prGate(view: AdmissionView | null, isAdmin: boolean): PrGate {
  const open = { label: "Open PR", disabled: false, reason: "", draftAnyway: false };
  if (!view || view.kind === "none" || view.kind === "unsupported" || view.kind === "failed")
    return { ...open, reason: "Opening measures the corpus rows against main first (a few minutes)." };
  if (view.kind === "running")
    return { ...open, disabled: true, reason: "The admission is still measuring this spec against main." };
  if (view.kind === "stale")
    return { ...open, disabled: true, reason: "The spec changed since this PASS — review it again." };
  const r = view.report;
  switch (r.status) {
    case "ready":
      return { ...open, reason: "Every corpus row is ready; the PR carries them." };
    case "needs_review":
      return { ...open, label: "Open draft PR",
        reason: "Rows wait on a maintainer's decision; the PR opens as a draft with the ready rows." };
    case "blocked":
      if (r.blocked_kind === "base")
        return { ...open, disabled: true,
          reason: "main's corpus baseline needs a refresh by a maintainer; no one can open this PR until then." };
      return { ...open, disabled: true, draftAnyway: isAdmin,
        reason: "The corpus checks fail for this model: fix the spec (Agent Fix), or a maintainer fixes the check's rule on main." };
    case "already_admitted":
      return { ...open, disabled: true, reason: "main already holds this model in every corpus table." };
    default:
      return { ...open, reason: "The last admission could not run; opening runs it again." };
  }
}

function Json({ value }: { value: unknown }) {
  return (
    <pre className="mt-1 text-10 font-mono bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto whitespace-pre-wrap max-h-56">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function Group({ title, tone, children }: { title: string; tone: string; children: React.ReactNode }) {
  return (
    <div className="mt-2">
      <div className={`text-11 font-semibold ${tone}`}>{title}</div>
      <ul className="mt-0.5 space-y-1 text-11 text-gray-700 list-none p-0 m-0">{children}</ul>
    </div>
  );
}

function ReportTable({ report }: { report: AdmissionReport }) {
  const g = admissionGroups(report);
  const reason = (report.gates?.G0 || []).join("; ");
  return (
    <div data-admission-table className="px-3 pb-3">
      {report.status === "unavailable" && reason ? (
        <div className="mt-2 text-11 text-gray-600">Could not run: {reason}</div>
      ) : null}
      {g.failedChecks.length ? (
        <Group title="Corpus checks that fail" tone="text-red-700">
          {g.failedChecks.map((c) => (
            <li key={c.id}>
              <span className="font-mono">{c.id}</span> {c.title}
              {(c.violations || []).map((v, i) => <div key={i} className="font-mono text-gray-500 break-all">{v}</div>)}
            </li>
          ))}
        </Group>
      ) : null}
      {g.pending.length || g.withheld.length ? (
        <Group title="Needs a maintainer's decision" tone="text-amber-800">
          {g.pending.map((d) => (
            <li key={d.id}>
              <span className="font-mono">{d.id}</span> <span className="text-gray-500">({d.state})</span>{" "}
              {(d.detail || []).join("; ")}
              {d.suggestion && Object.keys(d.suggestion as object).length ? (
                <details>
                  <summary className="cursor-pointer text-gray-500">Suggested value</summary>
                  <Json value={d.suggestion} />
                </details>
              ) : null}
            </li>
          ))}
          {g.withheld.map(({ item, keys }) => (
            <li key={item.id}>
              <span className="font-mono">{item.id}</span> {item.title}: {keys.length} row(s) wait on{" "}
              {Array.from(new Set(keys.flatMap((k) => item.key_states?.[k]?.waits_on || []))).join(", ")}
              {item.expected_failing_tests?.length ? (
                <div className="text-gray-500">Failing until then: <span className="font-mono">{item.expected_failing_tests.join(", ")}</span></div>
              ) : null}
            </li>
          ))}
        </Group>
      ) : null}
      {g.ready.length || g.held.length ? (
        <Group title={report.mode === "pr" ? "Written into the PR" : "Ready to admit"} tone="text-green-800">
          {g.ready.map(({ item, keys }) => (
            <li key={item.id}><span className="font-mono">{item.id}</span> {item.title}: <span className="font-mono break-all">{keys.join(", ")}</span></li>
          ))}
          {g.held.map(({ item, keys }) => (
            <li key={`held-${item.id}`} className="text-gray-500">
              <span className="font-mono">{item.id}</span> {item.title}: {keys.length} row(s) ready once the blocking gate clears
            </li>
          ))}
        </Group>
      ) : null}
      {g.baseNotes.length ? (
        <Group title="main's baseline (not this spec's)" tone="text-gray-700">
          {g.baseNotes.map((n) => <li key={n.id}><span className="font-mono">{n.id}</span> {(n.detail || []).join("; ")}</li>)}
        </Group>
      ) : null}
      {g.info.length ? (
        <Group title="Information" tone="text-gray-500">
          {g.info.map((d) => <li key={d.id}><span className="font-mono">{d.id}</span> {(d.detail || []).join("; ")}</li>)}
        </Group>
      ) : null}
      {report.pr?.reconciled ? (
        <div className="mt-2 text-11 text-red-700">{report.pr.reconciled}</div>
      ) : null}
    </div>
  );
}

// One line under a PASS review, expandable to the grouped table.
export function AdmissionLine({ row, onRerun }: { row: AdmissionRow; onRerun?: () => void }) {
  const [open, setOpen] = useState(false);
  const view = admissionView(row);
  const rerun = onRerun ? (
    <button type="button" onClick={onRerun}
      className="ml-2 text-11 px-2 py-0.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50">
      Run again
    </button>
  ) : null;
  let body: React.ReactNode;
  let report: AdmissionReport | null = null;
  switch (view.kind) {
    case "unsupported":
      body = <span className="text-gray-500">Admission: unavailable — this server cannot store admission reports yet.</span>;
      break;
    case "none":
      body = <span className="text-gray-500">Admission: no report for this review.{rerun}</span>;
      break;
    case "running":
      body = <span className="inline-flex items-center gap-1.5 text-amber-800"><Spinner />Admission: measuring this spec against main…</span>;
      break;
    case "failed":
      body = <span className="text-red-700">Admission: failed ({view.error}).{rerun}</span>;
      break;
    case "stale":
      report = view.report;
      body = <span className="text-gray-500">Admission: stale — the spec changed since this review.</span>;
      break;
    case "report": {
      report = view.report;
      const g = admissionGroups(report);
      const s = String(report.status || "unavailable");
      const why = s === "unavailable" ? (report.gates?.G0 || [])[0] || (report.errors || [])[0] : "";
      body = (
        <span className="inline-flex items-center gap-1.5 flex-wrap">
          <span>Admission:</span>
          <span className={`px-1.5 py-0.5 rounded font-medium ${STATUS_STYLE[s] || STATUS_STYLE.unavailable}`} data-admission-status={s}>
            {statusLabel(report)}
          </span>
          <span className="text-gray-500">
            · {report.base?.kind === "deployment" ? "deployment" : "base"} <span className="font-mono">{sha7(report.base?.sha)}</span>
            {" "}· {g.pending.length + g.withheld.length} awaiting a decision · {g.failedChecks.length} check(s) failed
            {why ? ` · ${why}` : ""}
          </span>
          {s === "unavailable" ? rerun : null}
        </span>
      );
      break;
    }
  }
  return (
    <div className="border-t border-gray-200 bg-white text-11" data-admission-line>
      <div className="px-3 py-1.5 flex items-center gap-2">
        <div className="min-w-0 flex-1">{body}</div>
        {report ? (
          <button type="button" className="shrink-0 text-11 text-gray-500 hover:text-gray-800 underline underline-offset-2"
            onClick={() => setOpen((o) => !o)}>
            {open ? "Hide" : "Details"}
          </button>
        ) : null}
      </div>
      {open && report ? <ReportTable report={report} /> : null}
    </div>
  );
}
