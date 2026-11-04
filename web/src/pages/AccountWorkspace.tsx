import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError, fetchJson } from "../api";
import { agentLabel, useAgentBackends, type AgentBackends } from "../agents";
import { authJson, loadAuth, loginRedirect, refreshSession, signOut, type AuthState } from "../auth";
import { AgentJobCard, Spinner } from "../AgentJob";
import { ACCOUNT_PANELS, AccountNav, WORKSPACE_PANELS } from "../AccountNav";
import { SpecFiles } from "../SpecFiles";
import { MarkdownReport } from "../Markdown";
import { specStatusLabel } from "../specGen";
import { isPendingStatus, type EngineGap } from "../specStatus";
import { EngineBlockedBadge, EngineGapTable } from "../EngineGaps";
import { AdmissionLine, admissionView, prGate, type AdmissionRow } from "../Admission";
import { DataTable, sortRows, type ColumnDef, type SortState } from "../DataTable";
import { alertError, confirmChoice, confirmDialog, promptDialog } from "../Dialog";
import { BenchReviewModal, benchCorpusHref, benchReviewTarget, peakCheckColumns, type BenchReviewTarget, type PeakCounts } from "./BenchReview";

type WorkspacePanel = (typeof WORKSPACE_PANELS)[number];

type LogRef = {
  id: string;
  success?: boolean | null;
  created_at?: string;
  // A review log: the verdict recorded for its review, which can differ from
  // the one the agent wrote at the end of the transcript.
  verdict?: string | null;
};

type Generation = {
  id: string;
  model_id?: string;
  status?: string;
  pr_url?: string | null;
  agent_cli?: string;
  agent_model?: string;
  created_at?: string;
  release_date?: string | null;
  log_id?: string;
  log_success?: boolean | null;
  fix_logs?: LogRef[];
  review_logs?: LogRef[];
  latest_verdict?: string | null;
  // The library spec that supersedes this record (the Inspector serves that
  // spec, never this record); set by the backend, null otherwise.
  superseded_by?: string | null;
  // The library spec the Inspector serves for this record's model, null when
  // it serves the record; set by the backend. With `superseded_by` null it
  // means a merged record is now in the library.
  library_spec?: string | null;
  // A pending record (`blocked_by_engine` / `needs_repair`): the engine
  // refusals it met, or the errors a re-check found; `engine_changed` when
  // the engine moved on since it was blocked.
  engine_gaps?: EngineGap[] | null;
  repairable_errors?: string[] | null;
  engine_changed?: boolean;
};

// Full generation payload (owner-scoped GET /api/account/generations/{id}) —
// what the spec-detail drawer shows on top of the list summary.
type GenDetail = Generation & {
  spec_yaml?: string;
  snapshot_json?: unknown;
  new_component_yamls?: { path?: string; body?: string }[];
};

// A failed run leaves an orphan log (no spec record); it folds into the same
// list as a lightweight row carrying just the log.
type OrphanLog = {
  id: string;
  model_id?: string;
  success?: boolean | null;
  agent_cli?: string;
  agent_model?: string;
  created_at?: string;
  release_date?: string | null;
  kind?: string | null;
  run_key?: string | null;
  // The run's failure as a machine key (`RunErrorKind`) and, for a run the
  // engine blocked, its refusals.
  error_kind?: string | null;
  engine_gaps?: EngineGap[] | null;
};

type Row = (Generation & { kind: "gen" }) | (OrphanLog & { kind: "log" });

// A saved module variant ("my modules" in the Playground) — managed under the
// Workspace → My modules panel.
type Variant = {
  id: string;
  component: string;
  name: string;
  dims: Record<string, number>;
  base_model_id?: string | null;
};

type Review = AdmissionRow & {
  id?: string;
  reviewer_role?: string;
  created_at?: string;
  report?: string;
};

// A live job on one record. Review/fix are polled from
// /api/spec/{review,fix}-status; `pr` (Open PR / update-pr: the corpus
// admission against main, then GitHub) from the record's pr-status.
// Setting a PR job must not clobber a running review.
type AgentJob = {
  kind: "review" | "fix" | "pr" | "recheck";
  startedAt: number;
  steps: string[];
};

type JobStatus = {
  state?: "idle" | "running" | "done" | "interrupted";
  // `fixable` rides on a review's `done` event when the saved spec does not
  // load but Agent Fix can open it on those errors.
  events?: { type?: string; text?: string; fixable?: boolean; pr_url?: string;
    code?: string; draft?: boolean; record?: { outcome?: string; id?: string } }[];
  next_cursor?: number;
  success?: boolean | null;
  message?: string | null;
  started_at?: number | null;
};

type ActionMsg = { text: string; tone: "ok" | "err" };

type GenLog = {
  id: string;
  model_id?: string;
  success?: boolean | null;
  created_at?: string;
  log?: string;
  kind?: string | null;
  run_key?: string | null;
  agent_cli?: string;
  agent_model?: string;
};

const CARD = "bg-white border border-gray-200 rounded-2xl p-6 sm:p-7 shadow-sm";

const PANEL_TITLE: Record<WorkspacePanel, string> = {
  generations: "Spec generations",
  modules: "My modules",
  benchmarks: "Kernel benchmarks",
};

function panelFromHash(): WorkspacePanel {
  const h = (window.location.hash || "").replace(/^#/, "");
  return (WORKSPACE_PANELS as readonly string[]).includes(h) ? (h as WorkspacePanel) : "generations";
}

// Where a record job is polled: review / fix by the spec routes, Open PR /
// update-pr by the record's own pr-status.
function jobStatusPath(id: string, kind: AgentJob["kind"], cursor: number): string {
  const rid = encodeURIComponent(id);
  return kind === "pr"
    ? `/api/account/generations/${rid}/pr-status?cursor=${cursor}`
    : `/api/spec/${kind}-status?record_id=${rid}&cursor=${cursor}`;
}

async function accountJson<T>(method: string, path: string, body?: unknown): Promise<T> {
  const send = () => fetchJson<T>(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : {},
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  try {
    return await send();
  } catch (e) {
    if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
      if (await refreshSession()) return await send();
      loginRedirect();
      throw new Error("Not signed in");
    }
    if (e instanceof ApiError && (e.status === 502 || e.status === 503)) {
      await new Promise((r) => setTimeout(r, 600));
      return await send();
    }
    throw e;
  }
}

function AvatarFace({ url, text, size }: { url: string; text: string; size: string }) {
  const [failed, setFailed] = useState(false);
  const initial = (text || "?").trim().slice(0, 1).toUpperCase() || "?";
  if (!url || failed) {
    return (
      <span className={`${size} rounded-full bg-amber-100 text-amber-700 inline-flex items-center justify-center font-semibold`}>
        {initial}
      </span>
    );
  }
  return (
    <img
      src={url}
      alt=""
      referrerPolicy="no-referrer"
      className={`${size} rounded-full object-cover`}
      onError={() => setFailed(true)}
    />
  );
}

// Spec-record status as a dot + label: lifecycle state, visually distinct
// from the review verdict pill beside it.
const STATUS_TONE: Record<string, { dot: string; text: string }> = {
  generated: { dot: "bg-gray-400", text: "text-gray-500" },
  "no PR": { dot: "border-[1.5px] border-amber-500 bg-transparent", text: "text-amber-700" },
  "PR open": { dot: "bg-amber-500", text: "text-amber-700" },
  "needs repair": { dot: "bg-amber-600", text: "text-amber-800" },
  merged: { dot: "bg-green-500", text: "text-green-700" },
  rejected: { dot: "bg-red-500", text: "text-red-600" },
  closed: { dot: "bg-slate-400", text: "text-slate-500" },
};

function StatusDot({ status, prUrl }: { status?: string; prUrl?: string | null }) {
  const label = specStatusLabel(status, prUrl ?? null);
  const tone = STATUS_TONE[label] || { dot: "border-[1.5px] border-gray-400 bg-transparent", text: "text-gray-500" };
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium whitespace-nowrap ${tone.text}`} title="Spec status">
      <span className={`h-2 w-2 rounded-full shrink-0 ${tone.dot}`} />
      {label}
    </span>
  );
}

function reviewBadgeClass(v: string): string {
  if (v === "pass") return "bg-green-100 text-green-700";
  if (v === "no_pass") return "bg-red-100 text-red-700";
  return "bg-gray-100 text-gray-500";
}

// A record whose model the library serves: its reviews judged the record's
// own spec, not what the product uses, so its verdict renders greyed.
const LIBRARY_VERDICT_TITLE = "Review of this record's own spec; the library spec is what the product uses";

// Latest review verdict: an outlined pill with an icon, so it never reads as
// another status. `historical` greys it (see LIBRARY_VERDICT_TITLE).
function verdictBadge(v?: string | null, historical = false) {
  if (!v) return null;
  const tone = historical
    ? "border-gray-200 text-gray-400"
    : v === "pass"
      ? "border-green-200 text-green-700"
      : v === "no_pass"
        ? "border-red-200 text-red-600"
        : "border-gray-200 text-gray-500";
  const icon = v === "pass" ? "\u2713" : v === "no_pass" ? "\u2715" : "?";
  return (
    <span
      className={`shrink-0 inline-flex items-center gap-1 px-1.5 py-px rounded-md border bg-white text-10 font-semibold tracking-wide ${tone}`}
      title={historical ? LIBRARY_VERDICT_TITLE : "Latest review verdict"}
      data-verdict={v}
      data-verdict-historical={historical ? "" : undefined}
    >
      <span aria-hidden>{icon}</span>
      {v.toUpperCase()}
    </span>
  );
}

// A merged record whose model the library now serves (backend
// `library_spec` with no `superseded_by`).
function InLibraryBadge({ g }: { g: Pick<Generation, "library_spec" | "superseded_by"> }) {
  if (!g.library_spec || g.superseded_by) return null;
  return (
    <span
      className="shrink-0 inline-flex items-center px-1.5 py-px rounded-md border border-green-200 bg-green-50 text-10 font-semibold tracking-wide text-green-700"
      title={`In the library as ${g.library_spec}.yaml — the Inspector serves that spec for this model`}
      data-in-library={g.library_spec}
    >
      in library
    </span>
  );
}

// Compact timestamp for list rows ("5m ago", "3d ago", "Sep 6"); the full
function logKind(row: { kind?: string | null; run_key?: string | null }): "fix" | "review" | null {
  const k = (row.kind || "").toLowerCase();
  if (k === "fix" || k === "review") return k;
  const head = (row.run_key || "").split("|", 1)[0];
  return head === "fix" || head === "review" ? head : null;
}

// local time rides in the title tooltip.
function publishedDate(ymd?: string | null): string {
  if (!ymd) return "—";
  const parts = ymd.split("-").map(Number);
  if (parts.length < 3 || parts.some((n) => !Number.isFinite(n))) return ymd;
  const [y, m, d] = parts;
  return new Date(y, m - 1, d).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function shortDate(iso?: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  const mins = Math.floor((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  if (mins < 60 * 24) return `${Math.floor(mins / 60)}h ago`;
  if (mins < 60 * 24 * 7) return `${Math.floor(mins / (60 * 24))}d ago`;
  const sameYear = d.getFullYear() === new Date().getFullYear();
  return d.toLocaleDateString(undefined, sameYear ? { month: "short", day: "numeric" } : { month: "short", day: "numeric", year: "numeric" });
}

function runAgentLine(row: { agent_cli?: string; agent_model?: string },
                      agents: AgentBackends | null): string {
  const agent = agentLabel(agents, row.agent_cli);
  const model = (row.agent_model || "").trim();
  if (agent && model) return `${agent} · ${model}`;
  return model || agent || "run";
}

// "org/name" → muted org + medium-weight name. The row click opens the
// drawer; Hugging Face is a separate control (`HfButton`).
function ModelName({ id }: { id?: string }) {
  const full = id || "";
  const cut = full.indexOf("/");
  if (!full) return <span className="w-full min-w-0" />;
  return (
    <span
      data-model-name={full}
      className="min-w-0 flex-1 text-sm break-all line-clamp-2 xl:line-clamp-none xl:truncate"
      title={full}
    >
      {cut > 0 ? <span className="text-gray-400">{full.slice(0, cut + 1)}</span> : null}
      <span className="font-medium text-gray-900">{cut > 0 ? full.slice(cut + 1) : full}</span>
    </span>
  );
}

const ICON = "h-4 w-4 shrink-0";
const ChevronRight = ({ className = ICON }: { className?: string }) => (
  <svg className={className} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth={1.8} aria-hidden>
    <path d="M7.5 5l5 5-5 5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const ExternalIcon = () => (
  <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth={1.8} aria-hidden>
    <path d="M11 4h5v5M16 4l-7 7M14 11.5V15a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h3.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

function HfButton({ id }: { id?: string }) {
  const full = (id || "").trim();
  if (!full) return null;
  return (
    <a
      href={`https://huggingface.co/${full}`}
      target="_blank"
      rel="noopener noreferrer"
      onClick={(e) => e.stopPropagation()}
      className="shrink-0 px-1 py-px rounded text-10 font-semibold tracking-wide text-gray-400 hover:text-amber-800 hover:bg-amber-50"
      title={`Open ${full} on Hugging Face`}
      aria-label="Open on Hugging Face"
    >
      HF
    </a>
  );
}
const TrashIcon = () => (
  <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth={1.7} aria-hidden>
    <path d="M4 6h12M8 6V4.5A.5.5 0 018.5 4h3a.5.5 0 01.5.5V6M6 6l.7 9.1a1 1 0 001 .9h4.6a1 1 0 001-.9L14 6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

// Row grid shared by spec rows and run-log rows: [checkbox] name · status/
// verdict column · meta column · actions. Keep the stacked form through lg:
// the account sidebar leaves the card much narrower than the viewport, so a
// viewport-only sm:flex-row layout can take all of that narrow card's space
// away from the model id.
const ROW = "group flex items-center gap-3 px-3 xl:px-4 py-2.5";
const NAME_COL = "flex w-full min-w-0 flex-wrap items-center gap-2 xl:w-auto xl:flex-grow xl:basis-52";
// The id and its HF link own a full line in the stacked form, so the badges
// and the "N runs" toggle wrap below it instead of squeezing the id; from xl
// the line shrinks to share the row with them.
const ID_LINE = "flex w-full min-w-0 items-center gap-2 xl:w-auto xl:flex-1";
const META_GROUP = "flex flex-wrap items-center gap-x-3 gap-y-0.5 min-w-0 xl:flex-nowrap";
const STATUS_COL = "flex items-center gap-2 xl:w-44 shrink-0";
const RELEASE_COL = "xl:w-28 shrink-0 text-xs text-gray-400 tabular-nums whitespace-nowrap xl:text-right";
const META_COL = "min-w-0 xl:w-56 truncate text-xs text-gray-400 tabular-nums xl:text-right";

// A spec record is deletable unless it has an open or merged PR (mirrors
// spec_generations.delete_for_user); orphan logs are always deletable.
const genDeletable = (g: { status?: string | null; pr_url?: string | null }) =>
  ["generated", "rejected", "closed"].includes(g.status || "") || isPendingStatus(g.status)
  || (g.status === "contributed" && !g.pr_url);

// A record's status: the engine-blocked badge (with its gaps) or the dot.
function RecordStatus({ g }: { g: Generation }) {
  return g.status === "blocked_by_engine"
    ? <EngineBlockedBadge gaps={g.engine_gaps} engineChanged={g.engine_changed} />
    : <StatusDot status={g.status} prUrl={g.pr_url} />;
}

const isArchivedGen = (r: Row) =>
  r.kind === "gen" && (r.status === "merged" || r.status === "closed");

function mergeRows(gens: Generation[], logs: OrphanLog[]): Row[] {
  const merged: Row[] = [
    ...gens.map((g) => ({ ...g, kind: "gen" as const })),
    ...logs.map((l) => ({ ...l, kind: "log" as const })),
  ];
  return merged.sort((a, b) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime());
}

function groupRowsByModel(list: Row[]): { key: string; head: Row; rest: Row[] }[] {
  const order: string[] = [];
  const map = new Map<string, Row[]>();
  for (const r of list) {
    const key = r.model_id || `__${r.kind}:${r.id}`;
    if (!map.has(key)) {
      order.push(key);
      map.set(key, []);
    }
    map.get(key)!.push(r);
  }
  return order.map((key) => {
    const members = map.get(key)!;
    const head = members.find((m) => m.kind === "gen") ?? members[0];
    return { key, head, rest: members.filter((m) => m !== head) };
  });
}

/** One batch of a set: one upload, reviewed on its own. */
type BenchBatch = {
  upload_id: string;
  label?: string | null;
  created_at?: string;
  status: string;
  row_count?: number;
  anomalous?: number;
};

/** A set of the caller's: a run (its batches share a run_id) or an upload
 *  without one. `status` is what is left to do with it (`private` while any
 *  batch is still to nominate); each batch's own is in `batches`. */
type Upload = {
  kind?: string;
  id?: string;
  run_id?: string;
  upload_id?: string;
  label?: string;
  frameworks?: string[];
  gpus?: string[];
  row_count?: number;
  created_at?: string;
  status?: string;
  batches?: BenchBatch[];
} & PeakCounts;

const CSV_COLUMNS = [
  "model", "spec_hash", "phase", "parallel", "framework", "framework_version",
  "gpu", "op", "node_spec_id", "shape_json", "dtype_env_json", "kernel_impl",
  "latency_us", "p50_us", "p95_us", "samples", "count_per_step", "achieved_tflops",
  "achieved_bw_gbps", "peak_tflops_at_test", "peak_bw_at_test",
  "schema_version", "contributor", "date", "run_id", "box",
];

function mgmtPath(kind: string, id: string, action: "delete" | "contribute" | "rows"): string {
  const enc = encodeURIComponent(id);
  if (kind === "run") {
    if (action === "delete") return `/api/bench/run/${enc}`;
    if (action === "contribute") return `/api/bench/run/${enc}/contribute`;
    return `/api/bench/rows?run_id=${enc}&limit=500`;
  }
  if (action === "delete") return `/api/bench/${enc}`;
  if (action === "contribute") return `/api/bench/${enc}/contribute`;
  return `/api/bench/rows?upload_id=${enc}&limit=500`;
}

function rowId(u: Upload): string {
  return u.id || u.run_id || u.upload_id || "";
}

function csvCell(v: unknown): string {
  if (v == null) return "";
  const s = typeof v === "object" ? JSON.stringify(v) : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function badge(status: string) {
  const m: Record<string, string> = {
    shared: "bg-green-100 text-green-700",
    nominated: "bg-amber-100 text-amber-700",
    private: "bg-gray-100 text-gray-500",
    rejected: "bg-red-100 text-red-700",
  };
  return <span className={`text-11 px-2 py-0.5 rounded-full ${m[status] || m.private}`}>{status || "private"}</span>;
}

const STATUS_ORDER = ["private", "nominated", "shared", "rejected"];

/** A set's status; a run whose batches differ lists each status with its
 *  count of batches. */
function statusCell(u: Upload) {
  const counts = new Map<string, number>();
  for (const b of u.batches || []) counts.set(b.status, (counts.get(b.status) || 0) + 1);
  if (counts.size <= 1) return badge(u.status || "private");
  return (
    <span className="inline-flex flex-wrap gap-1">
      {STATUS_ORDER.filter((st) => counts.has(st)).map((st) => (
        <span key={st} title={`${counts.get(st)} batch${counts.get(st) === 1 ? "" : "es"} ${st}`}>
          {badge(st)}<span className="text-10 text-gray-400 ml-0.5">×{counts.get(st)}</span>
        </span>
      ))}
    </span>
  );
}

/** The batches "Contribute" nominates: those still private. */
function privateBatches(u: Upload): BenchBatch[] {
  return (u.batches || []).filter((b) => b.status === "private");
}

const BENCH_PAGE_SIZE = 50;

/** When a batch was uploaded, short: the date, and the time of day. The full
 *  timestamp is the cell's hover. */
function uploadedAt(iso: string): string {
  const d = new Date(iso);
  return `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })}, ${d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`;
}

function batchColumns(onReview: (u: Upload) => void): ColumnDef<Upload>[] {
  return [
    { key: "batch", header: "Batch", className: "font-mono text-11", render: (u) => u.run_id || (u.upload_id || "").slice(0, 8) || "—", sortValue: (u) => u.run_id || (u.upload_id || "").slice(0, 8) },
    {
      key: "actions",
      header: "Actions",
      className: "whitespace-nowrap",
      sortValue: null,
      // Beside the batch id, stacked: the actions stay on screen however
      // many columns follow in the workspace's card.
      render: (u) => (
        <span className="inline-flex flex-col items-stretch gap-1">
          <button type="button" className="text-xs px-2 py-0.5 border rounded hover:bg-gray-50" onClick={() => onReview(u)}>Brief Review</button>
          <a className="text-xs px-2 py-0.5 border border-blue-200 text-blue-700 rounded hover:bg-blue-50 text-center" href={benchCorpusHref(benchReviewTarget(u.kind || "upload", rowId(u)))}>Detail Review</a>
        </span>
      ),
    },
    { key: "label", wrap: true, header: "Label", render: (u) => u.label || "(unnamed)", sortValue: (u) => u.label },
    { key: "status", header: "Status", render: (u) => statusCell(u), sortValue: (u) => u.status || "private" },
    { key: "framework", header: "Framework", className: "text-gray-600", render: (u) => (u.frameworks || []).join(", ") || "–", sortValue: (u) => (u.frameworks || []).join(", ") },
    { key: "gpu", header: "GPU", className: "text-gray-600", render: (u) => (u.gpus || []).join(", ") || "–", sortValue: (u) => (u.gpus || []).join(", ") },
    ...peakCheckColumns<Upload>(),
    { key: "rows", header: "Rows", align: "right", className: "text-gray-600", render: (u) => u.row_count, sortValue: (u) => u.row_count },
    { key: "uploaded", header: "Uploaded", className: "text-gray-500", render: (u) => (u.created_at ? uploadedAt(u.created_at) : "–"), cellProps: (u) => ({ title: u.created_at ? new Date(u.created_at).toLocaleString() : undefined }), sortValue: (u) => (u.created_at ? Date.parse(u.created_at) : null) },
  ];
}

// One of the panel's two batch tables. It pages, so it owns its sort (a page
// of a sorted list needs the whole list sorted before the page is cut). Above
// it, "Select page" and "Select all" pick the two scopes a bulk action is
// asked over; the header box toggles the page.
const SEL_BTN = "px-2 py-1 border border-gray-300 rounded bg-white hover:bg-gray-50 disabled:opacity-40 disabled:cursor-default";
function BatchTable({ title, hint, rows, loading, empty, sel, toggle, setMany, onReview }: {
  title: string;
  hint: string;
  rows: Upload[];
  loading: boolean;
  empty: string;
  sel: Set<string>;
  toggle: (id: string, on: boolean) => void;
  setMany: (ids: string[], on: boolean) => void;
  onReview: (u: Upload) => void;
}) {
  const [sort, setSort] = useState<SortState>(null);
  const [page, setPage] = useState(0);
  const base = batchColumns(onReview);
  const sorted = sortRows(rows, base, sort);
  const pages = Math.max(1, Math.ceil(sorted.length / BENCH_PAGE_SIZE));
  const pageIx = Math.min(page, pages - 1);
  const pageRows = sorted.slice(pageIx * BENCH_PAGE_SIZE, (pageIx + 1) * BENCH_PAGE_SIZE);
  const pageIds = pageRows.map(rowId);
  const allIds = rows.map(rowId);
  const pageAll = pageIds.length > 0 && pageIds.every((id) => sel.has(id));
  const tableAll = allIds.length > 0 && allIds.every((id) => sel.has(id));
  const nSel = allIds.filter((id) => sel.has(id)).length;
  const cols: ColumnDef<Upload>[] = [
    {
      key: "sel",
      header: (
        <input type="checkbox" aria-label="Select this page" checked={pageAll}
          onChange={(e) => setMany(pageIds, e.target.checked)} />
      ),
      headerClassName: "w-8",
      render: (u) => (
        <input type="checkbox" checked={sel.has(rowId(u))} onChange={(e) => toggle(rowId(u), e.target.checked)} />
      ),
      sortValue: null,
    },
    ...base,
  ];
  return (
    <div className="mb-6">
      <div className="flex items-baseline gap-3 mb-2">
        <h4 className="text-sm font-semibold text-gray-900">{title}</h4>
        <span className="text-xs text-gray-400">{rows.length.toLocaleString()} batch{rows.length === 1 ? "" : "es"}</span>
        <span className="text-xs text-gray-500">{hint}</span>
      </div>
      {rows.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 mb-2 text-xs">
          <button type="button" className={SEL_BTN} disabled={pageAll}
            onClick={() => setMany(pageIds, true)}>
            Select page ({pageRows.length})
          </button>
          <button type="button" className={SEL_BTN} disabled={tableAll}
            onClick={() => setMany(allIds, true)}>
            Select all ({rows.length.toLocaleString()})
          </button>
          {nSel > 0 && (
            <>
              <span className="text-gray-600">{nSel.toLocaleString()} selected</span>
              <button type="button" className="text-blue-700 hover:underline" onClick={() => setMany(allIds, false)}>Clear</button>
            </>
          )}
        </div>
      )}
      <div className="rounded-xl border border-gray-200 bg-white shadow-sm">
        <DataTable
          columns={cols}
          rows={pageRows}
          rowKey={(u) => rowId(u)}
          loading={loading}
          empty={empty}
          framed={false}
          sort={sort}
          onSort={(s) => {
            setSort(s);
            setPage(0);
          }}
        />
        {sorted.length > BENCH_PAGE_SIZE && (
          <div className="flex items-center justify-end gap-3 px-3 py-2 text-xs text-gray-500 border-t border-gray-100">
            <span>{(pageIx * BENCH_PAGE_SIZE + 1).toLocaleString()}–{Math.min(sorted.length, (pageIx + 1) * BENCH_PAGE_SIZE).toLocaleString()} / {sorted.length.toLocaleString()}</span>
            <button type="button" className="px-2 py-1 border border-gray-200 rounded bg-white disabled:opacity-40"
              disabled={pageIx <= 0} onClick={() => setPage(pageIx - 1)}>Prev</button>
            <button type="button" className="px-2 py-1 border border-gray-200 rounded bg-white disabled:opacity-40"
              disabled={pageIx >= pages - 1} onClick={() => setPage(pageIx + 1)}>Next</button>
          </div>
        )}
      </div>
    </div>
  );
}

// Kernel benchmarks panel (was the standalone my-benchmarks page): auth and
// the shell are the page's job; everything below is panel-local state, so
// leaving and returning to the panel reloads the batch list.

function BenchmarksPanel() {
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [status, setStatus] = useState("");
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [review, setReview] = useState<BenchReviewTarget | null>(null);
  const closeReview = useCallback(() => setReview(null), []);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const data = await authJson<{ uploads?: Upload[] }>("GET", "/api/bench/list");
      setUploads(data.uploads || []);
      setErr("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const selected = useMemo(
    () => uploads.filter((u) => sel.has(rowId(u))),
    [uploads, sel],
  );
  const nPriv = selected.filter((u) => (u.status || "private") === "private").length;

  const toggle = (id: string, on: boolean) => {
    setSel((s) => {
      const n = new Set(s);
      if (on) n.add(id); else n.delete(id);
      return n;
    });
  };

  const bulk = async (action: "delete" | "contribute") => {
    const rows = action === "contribute"
      ? selected.filter((u) => (u.status || "private") === "private")
      : selected;
    if (!rows.length) return;
    const n = `${rows.length} ${action === "delete" ? "batch" : "set"}${rows.length === 1 ? "" : action === "delete" ? "es" : "s"}`;
    // Rows faster than a peak they recorded, in the batches about to be
    // nominated: the contributor chooses whether those go up for review too.
    const over = rows.reduce((a, u) => a + privateBatches(u).reduce((t, b) => t + (b.anomalous || 0), 0), 0);
    let withinPeak = false;
    if (action === "contribute" && over > 0) {
      const choice = await confirmChoice({
        title: `Nominate ${n} for the shared corpus?`,
        message: "An admin reviews each before it goes public.",
        option: {
          label: <>Only rows within peak — leave out the <b>{over.toLocaleString()}</b> row{over === 1 ? "" : "s"} that ran
            faster than a peak they recorded (MFU or MBU over 100%)</>,
          checked: true,
        },
        confirmLabel: "Nominate",
      });
      if (choice === null) return;
      withinPeak = choice;
    } else {
      const confirmed = await confirmDialog(action === "delete"
        ? { title: `Delete ${n}?`, message: "Every row in them goes too. This cannot be undone.",
            confirmLabel: "Delete", danger: true }
        : { title: `Nominate ${n} for the shared corpus?`,
            message: "An admin reviews each before it goes public.", confirmLabel: "Nominate" });
      if (!confirmed) return;
    }
    setStatus(action === "delete" ? `Deleting ${rows.length}…` : `Submitting ${rows.length}…`);
    let ok = 0, fail = 0;
    for (const u of rows) {
      const kind = u.kind || "upload";
      const id = rowId(u);
      try {
        await authJson(action === "delete" ? "DELETE" : "POST",
          mgmtPath(kind, id, action) + (withinPeak ? "?within_peak=true" : ""));
        ok++;
      } catch {
        fail++;
      }
    }
    setStatus(action === "delete"
      ? `Deleted ${ok}${fail ? `, ${fail} failed` : ""}.`
      : `Nominated ${ok}${fail ? `, ${fail} failed` : ""}${withinPeak ? ` (${over.toLocaleString()} over-peak rows left out)` : ""} — pending admin review.`);
    setSel(new Set());
    await load();
  };

  const download = async () => {
    if (!selected.length) return;
    setStatus(`Preparing ${selected.length} set(s)…`);
    try {
      const lines = [CSV_COLUMNS.join(",")];
      for (const u of selected) {
        const data = await authJson<{ rows?: Record<string, unknown>[] }>(
          "GET",
          mgmtPath(u.kind || "upload", rowId(u), "rows"),
        );
        for (const row of data.rows || []) {
          lines.push(CSV_COLUMNS.map((c) => csvCell(row[c])).join(","));
        }
      }
      const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `benchmarks_${selected.length}sets.csv`;
      a.click();
      URL.revokeObjectURL(url);
      setStatus("");
    } catch (e) {
      setStatus("Download failed: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  const onFile = async (file: File) => {
    setStatus(`Reading ${file.name}…`);
    try {
      const text = await file.text();
      setStatus("Uploading…");
      const data = await authJson<{ row_count?: number; duplicates?: number; anomalous_count?: number }>(
        "POST",
        "/api/bench/upload",
        { csv_text: text, label: file.name },
      );
      await load();
      const anom = data.anomalous_count ? ` (${data.anomalous_count} flagged anomalous)` : "";
      const dup = data.duplicates ? `; ${data.duplicates} already stored, skipped` : "";
      setStatus(`Uploaded ${data.row_count} row(s)${anom}${dup}.`);
    } catch (e) {
      setStatus("Upload failed: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  // Two tables, by whether the set still has a batch that is the
  // contributor's alone: private ones are still to nominate (or delete), and
  // the nominated / shared / rejected ones are past that step. One selection
  // spans both, so the actions bar acts on whatever is ticked in either.
  const privateUploads = uploads.filter((u) => (u.status || "private") === "private");
  const submittedUploads = uploads.filter((u) => (u.status || "private") !== "private");
  const openReview = (u: Upload) => setReview(benchReviewTarget(u.kind || "upload", rowId(u)));
  const setMany = (ids: string[], on: boolean) => {
    setSel((s) => {
      const n = new Set(s);
      for (const id of ids) {
        if (on) n.add(id); else n.delete(id);
      }
      return n;
    });
  };

  return (
    <>
      <section className={CARD}>
        <p className="text-10 font-semibold uppercase tracking-wider text-amber-700 mb-1">Benchmarks</p>
        <h2 className="text-lg font-semibold text-gray-900 mb-1">Kernel benchmarks</h2>
        <p className="text-sm text-gray-500 mb-4 max-w-xl leading-relaxed">
          Private measured timings — one <b>run_id</b> is one sweep. Select batches to inspect, delete, contribute, or download.
        </p>
        <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
          <h3 className="text-base font-semibold text-gray-900">Measurement batches</h3>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500">{status}</span>
            <button type="button" className="text-xs px-3 py-1.5 border border-amber-300 text-amber-800 rounded hover:bg-amber-50" onClick={() => fileRef.current?.click()}>
              Upload measured CSV
            </button>
            <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) void onFile(f);
            }} />
          </div>
        </div>
        {selected.length > 0 && (
          <div className="flex items-center gap-2 mb-3 p-2 bg-amber-50 border border-amber-200 rounded-lg">
            <span className="text-xs text-gray-700 font-medium">{selected.length} selected</span>
            <div className="flex-1" />
            <button type="button" disabled={nPriv === 0} className={`text-xs px-2 py-1 border border-amber-300 text-amber-700 rounded ${nPriv === 0 ? "opacity-40" : "hover:bg-amber-100"}`} onClick={() => void bulk("contribute")}>
              {nPriv && nPriv !== selected.length ? `Contribute ${nPriv} private` : "Contribute selected"}
            </button>
            <button type="button" className="text-xs px-2 py-1 border border-gray-300 rounded hover:bg-gray-100" onClick={() => void download()}>Download selected</button>
            <button type="button" className="text-xs px-2 py-1 border border-red-300 text-red-700 rounded hover:bg-red-100" onClick={() => void bulk("delete")}>Delete selected</button>
          </div>
        )}
        {err && <p className="text-sm text-red-600 mb-2">{err}</p>}
        <BatchTable
          title="Private"
          hint="Runs and uploads with a batch only you can see. Nominate it for the shared corpus, or delete it."
          rows={privateUploads}
          loading={loading}
          empty={err ? "" : "No private batches."}
          sel={sel}
          toggle={toggle}
          setMany={setMany}
          onReview={openReview}
        />
        <BatchTable
          title="Nominated & shared"
          hint="Submitted for review (nominated), in the shared corpus, or rejected by review."
          rows={submittedUploads}
          loading={loading}
          empty={err ? "" : "Nothing nominated yet."}
          sel={sel}
          toggle={toggle}
          setMany={setMany}
          onReview={openReview}
        />
      </section>
      {review && <BenchReviewModal target={review} onClose={closeReview} />}
    </>
  );
}

export function AccountWorkspace() {
  // Agent names only label runs: until the registry is read, a run shows its
  // agent's id.
  const { agents } = useAgentBackends();
  const nav = useNavigate();
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [panel, setPanel] = useState<WorkspacePanel>(panelFromHash);
  const [avatarUrl, setAvatarUrl] = useState("");

  const [gens, setGens] = useState<Generation[]>([]);
  // Records a library spec supersedes: the backend lists them apart
  // (`?superseded=true`) and they render in their own archived section.
  const [supersededGens, setSupersededGens] = useState<Generation[]>([]);
  const [gensReady, setGensReady] = useState(false);
  const [genErr, setGenErr] = useState("");
  const [genStatus, setGenStatus] = useState("");
  const [reviews, setReviews] = useState<Record<string, Review[]>>({});
  const [reviewsErr, setReviewsErr] = useState<Record<string, string>>({});
  const [reviewsLoaded, setReviewsLoaded] = useState<Record<string, boolean>>({});
  const [jobs, setJobs] = useState<Record<string, AgentJob>>({});
  const [actionMsg, setActionMsg] = useState<Record<string, ActionMsg>>({});
  // Records whose last review could not start because the saved spec does not
  // load, with errors Agent Fix can repair: Fix is offered on those, too.
  const [fixableLoad, setFixableLoad] = useState<Record<string, boolean>>({});
  const [now, setNow] = useState(() => Date.now());
  const [orphanLogs, setOrphanLogs] = useState<OrphanLog[]>([]);
  // Run logs of models the library now serves: archived with those records.
  const [supersededLogs, setSupersededLogs] = useState<OrphanLog[]>([]);
  const [logModal, setLogModal] = useState<{ title: string; body: string } | null>(null);
  const logPreRef = useRef<HTMLPreElement>(null);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  // Spec-detail drawer: the record whose full content (YAML / components /
  // reviews) is open, plus its fetched payload. The list endpoint omits the
  // heavy payload, so the drawer fetches it on open.
  const [detailId, setDetailId] = useState<string | null>(null);
  const [detail, setDetail] = useState<GenDetail | null>(null);
  const [detailErr, setDetailErr] = useState("");
  const detailIdRef = useRef<string | null>(null);
  const genLoaded = useRef(false);
  const pollers = useRef<Record<string, number>>({});
  const prInflight = useRef(new Set<string>());

  const [variants, setVariants] = useState<Variant[] | null>(null);
  const [variantErr, setVariantErr] = useState("");

  useEffect(() => {
    document.title = `${PANEL_TITLE[panel]} · HuggingArch`;
    return () => {
      document.title = "HuggingArch";
    };
  }, [panel]);

  useEffect(() => {
    void loadAuth().then((a) => {
      if (!a.authEnabled) {
        nav("/", { replace: true });
        return;
      }
      if (!a.user) {
        loginRedirect();
        return;
      }
      setAuth(a);
      setAvatarUrl(a.profile?.avatar_url || a.user.user_metadata?.avatar_url || "");
    });
  }, [nav]);

  useEffect(() => {
    const onHash = () => setPanel(panelFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const want = `#${panel}`;
    if (window.location.hash !== want) history.replaceState(null, "", want);
  }, [panel]);

  // Deep links to the profile page's panels land here when the hash predates
  // the split (the hash never reaches the server) — forward them over.
  useEffect(() => {
    const h = (window.location.hash || "").replace(/^#/, "");
    if ((ACCOUNT_PANELS as readonly string[]).includes(h)) {
      nav(`/account/profile.html#${h}`, { replace: true });
    }
  }, [nav]);

  const loadGens = useCallback(async () => {
    try {
      const [data, archived] = await Promise.all([
        accountJson<{ items?: Generation[]; orphan_logs?: OrphanLog[] }>("GET", "/api/account/generations"),
        accountJson<{ items?: Generation[]; orphan_logs?: OrphanLog[] }>("GET", "/api/account/generations?superseded=true"),
      ]);
      setGens(data.items || []);
      setSupersededGens(archived.items || []);
      setSupersededLogs(archived.orphan_logs || []);
      setOrphanLogs(data.orphan_logs || []);
      setGenErr("");
      genLoaded.current = true;
    } catch (e) {
      setGenErr(e instanceof Error ? e.message : String(e));
    } finally {
      setGensReady(true);
    }
  }, []);

  const variantsLoading = useRef(false);
  const loadVariants = useCallback(async () => {
    if (variantsLoading.current) return;
    variantsLoading.current = true;
    try {
      const data = await accountJson<{ success: boolean; variants?: Variant[]; message?: string }>("GET", "/api/account/modules");
      if (!data.success) throw new Error(data.message || "server rejected the request");
      setVariants(data.variants || []);
      setVariantErr("");
    } catch (e) {
      setVariants(null);
      setVariantErr(e instanceof Error ? e.message : String(e));
    } finally {
      variantsLoading.current = false;
    }
  }, []);

  const renameVariant = useCallback(async (v: Variant) => {
    const name = await promptDialog({ title: "Rename variant", defaultValue: v.name, confirmLabel: "Rename" });
    if (!name?.trim() || name.trim() === v.name) return;
    try {
      await accountJson("PATCH", `/api/account/modules/${encodeURIComponent(v.id)}`, { name: name.trim() });
      await loadVariants();
    } catch (e) {
      setVariantErr(`Rename failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [loadVariants]);

  const deleteVariant = useCallback(async (v: Variant) => {
    if (!(await confirmDialog({ title: `Delete “${v.name}”?`, message: `A saved parameterization of ${v.component}.`,
      confirmLabel: "Delete", danger: true }))) return;
    try {
      await accountJson("DELETE", `/api/account/modules/${encodeURIComponent(v.id)}`);
      await loadVariants();
    } catch (e) {
      setVariantErr(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [loadVariants]);

  useEffect(() => {
    if (!auth?.user) return;
    void loadGens();
  }, [auth, loadGens]);

  useEffect(() => {
    if (panel === "modules" && auth?.user && variants === null && !variantErr) void loadVariants();
  }, [panel, auth, variants, variantErr, loadVariants]);

  useEffect(() => {
    if (panel === "generations" && genLoaded.current) void loadGens();
  }, [panel, loadGens]);

  useEffect(() => () => {
    Object.values(pollers.current).forEach((id) => clearTimeout(id));
  }, []);

  // Jump the log viewer to the end (the run's outcome) when it opens/updates,
  // matching the old openGenLog's scrollTop = scrollHeight.
  useEffect(() => {
    if (logModal && logPreRef.current) logPreRef.current.scrollTop = logPreRef.current.scrollHeight;
  }, [logModal]);

  const email = auth?.user?.email || "";
  const name = auth?.profile?.username || auth?.profile?.full_name || email || "Account";

  // One list, newest first: spec generations + orphan run logs (failed runs
  // produce no spec record — their rows carry just the log).
  const rows = useMemo<Row[]>(() => mergeRows(gens, orphanLogs), [gens, orphanLogs]);

  const activeRows = useMemo(() => rows.filter((r) => !isArchivedGen(r)), [rows]);
  const allGens = useMemo(() => [...gens, ...supersededGens], [gens, supersededGens]);
  const supersededGroups = useMemo(
    () => groupRowsByModel(mergeRows(supersededGens, supersededLogs)),
    [supersededGens, supersededLogs],
  );
  const archivedRows = useMemo(() => rows.filter(isArchivedGen), [rows]);
  const activeGroups = useMemo(() => groupRowsByModel(activeRows), [activeRows]);
  const archivedGroups = useMemo(() => groupRowsByModel(archivedRows), [archivedRows]);

  // Bulk selection — only deletable rows carry a checkbox. Keyed by `kind:id`
  // (ids are UUIDs; the kind prefix keeps a gen and a log from colliding).
  // Only rows on screen: a collapsed group's older runs are not selected by
  // "select all", so bulk delete never removes a run the user cannot see.
  const selectableKeys = useMemo(
    () => activeGroups
      .flatMap((g) => (openGroups[g.key] ? [g.head, ...g.rest] : [g.head]))
      .filter((r) => r.kind === "log" || genDeletable(r))
      .map((r) => `${r.kind}:${r.id}`),
    [activeGroups, openGroups],
  );
  const selectedCount = selectableKeys.filter((k) => selected[k]).length;
  const allSelected = selectableKeys.length > 0 && selectedCount === selectableKeys.length;

  // Drop selections whose row is gone after a reload/delete.
  useEffect(() => {
    setSelected((prev) => {
      const live = new Set(selectableKeys);
      const next: Record<string, boolean> = {};
      for (const k of Object.keys(prev)) if (prev[k] && live.has(k)) next[k] = true;
      return Object.keys(next).length === Object.keys(prev).length ? prev : next;
    });
  }, [selectableKeys]);

  const toggleSel = (key: string, on: boolean) =>
    setSelected((p) => {
      const n = { ...p };
      if (on) n[key] = true;
      else delete n[key];
      return n;
    });

  const renderRow = (row: Row, opts?: { nested?: boolean; runCount?: number; expanded?: boolean; onToggleExpand?: () => void }) => {
    if (row.kind === "log") {
      // A run the engine blocked is not a failure of its spec: it carries the
      // engine-blocked badge. Every other error kind reads as failed.
      const engineBlocked = row.success === false && row.error_kind === "blocked_by_engine";
      const outcome = row.success === true ? "succeeded" : row.success === false ? "failed" : "unknown";
      const dot = row.success === true ? "bg-green-500" : row.success === false ? "bg-red-500" : "bg-gray-400";
      const text = row.success === true ? "text-green-700" : row.success === false ? "text-red-600" : "text-gray-500";
      const jobKind = logKind(row);
      const jobBadge = jobKind === "fix" ? "fix log" : jobKind === "review" ? "review log" : "log only";
      return (
        <div
          key={`log-${row.id}`}
          role="button"
          tabIndex={0}
          className={`${ROW} ${opts?.nested ? "pl-8 xl:pl-10 bg-gray-50/60" : ""} cursor-pointer transition-colors hover:bg-gray-50 focus-visible:outline-none focus-visible:bg-gray-50`}
          onClick={() => openLog(row.id)}
          onKeyDown={(e) => {
            if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) {
              e.preventDefault();
              openLog(row.id);
            }
          }}
          title="Run log only — the run produced no spec. Click to open the log."
        >
          <input
            type="checkbox"
            className="rounded border-gray-300 shrink-0"
            title="Select for bulk delete"
            checked={!!selected[`log:${row.id}`]}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => toggleSel(`log:${row.id}`, e.target.checked)}
          />
          <div className="flex-1 min-w-0 flex flex-col gap-1 xl:flex-row xl:items-center xl:gap-4">
            <div className={NAME_COL}>
              {opts?.nested ? (
                <span className="text-xs text-gray-600 truncate" title={runAgentLine(row, agents)}>
                  {runAgentLine(row, agents)}
                </span>
              ) : (
                <span className={ID_LINE}>
                  <ModelName id={row.model_id} />
                  <HfButton id={row.model_id} />
                </span>
              )}
              {opts?.runCount && opts.runCount > 1 ? (
                <button
                  type="button"
                  className="shrink-0 inline-flex items-center gap-0.5 px-1.5 py-px rounded-md border border-gray-200 bg-white text-11 font-medium text-gray-600 hover:bg-gray-50"
                  title={opts.expanded ? "Hide older runs" : `Show ${opts.runCount - 1} older run${opts.runCount === 2 ? "" : "s"}`}
                  aria-expanded={opts.expanded}
                  onClick={(e) => {
                    e.stopPropagation();
                    opts.onToggleExpand?.();
                  }}
                >
                  {opts.runCount} runs
                  <ChevronRight className={`${ICON} text-gray-400 ${opts.expanded ? "rotate-90" : ""}`} />
                </button>
              ) : null}
            </div>
            <div className={META_GROUP}>
              <div className={STATUS_COL}>
                {engineBlocked ? (
                  <EngineBlockedBadge gaps={row.engine_gaps} />
                ) : (
                  <span className={`inline-flex items-center gap-1.5 text-xs font-medium whitespace-nowrap ${text}`}>
                    <span className={`h-2 w-2 rounded-full shrink-0 ${dot}`} />
                    run {outcome}
                  </span>
                )}
                <span className="inline-flex items-center px-1.5 py-px rounded-md border border-dashed border-gray-300 text-10 font-medium text-gray-500 whitespace-nowrap">
                  {jobBadge}
                </span>
              </div>
              <span className={RELEASE_COL} title={row.release_date ? `Published on Hugging Face ${row.release_date}` : "No Hugging Face createdAt on record"}>
                {publishedDate(row.release_date)}
              </span>
              <span className={META_COL} title={row.created_at ? new Date(row.created_at).toLocaleString() : ""}>{shortDate(row.created_at)}</span>
            </div>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              className="p-1 rounded-md text-gray-400 hover:text-red-600 hover:bg-red-50"
              title="Delete this run log. This cannot be undone."
              aria-label="Delete run log"
              onClick={async (e) => {
                e.stopPropagation();
                if (!(await confirmDialog({ title: "Delete this run log?", message: "This cannot be undone.",
                  confirmLabel: "Delete", danger: true }))) return;
                void accountJson("DELETE", `/api/account/gen-logs/${encodeURIComponent(row.id)}`).then(() => {
                  setGenStatus("Deleted.");
                  void loadGens();
                }).catch(alertError);
              }}
            >
              <TrashIcon />
            </button>
            <ChevronRight className={`${ICON} text-gray-300 transition-transform group-hover:text-gray-500 group-hover:translate-x-0.5`} />
          </div>
        </div>
      );
    }
    const g = row;
    const deletable = genDeletable(g);
    const job = jobs[g.id];
    return (
      <div
        key={g.id}
        role="button"
        tabIndex={0}
        className={`${ROW} ${opts?.nested ? "pl-8 xl:pl-10 bg-gray-50/60" : ""} cursor-pointer transition-colors hover:bg-amber-50/50 focus-visible:outline-none focus-visible:bg-amber-50/60`}
        onClick={() => openDetail(g.id)}
        onKeyDown={(e) => {
          if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            openDetail(g.id);
          }
        }}
        title="View this spec's YAML, reviews, and PR actions"
      >
        {g.status !== "merged" && (
          <input
            type="checkbox"
            className="rounded border-gray-300 shrink-0 disabled:opacity-30 disabled:cursor-not-allowed"
            disabled={!deletable}
            title={deletable ? "Select for bulk delete" : "A PR is open for this spec — it can't be deleted. Close the PR first."}
            checked={deletable && !!selected[`gen:${g.id}`]}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => toggleSel(`gen:${g.id}`, e.target.checked)}
          />
        )}
        <div className="flex-1 min-w-0 flex flex-col gap-1 xl:flex-row xl:items-center xl:gap-4">
          <div className={NAME_COL}>
            {opts?.nested ? (
              <span className="text-xs text-gray-600 truncate" title={runAgentLine(g, agents)}>
                {g.agent_model || g.agent_cli ? runAgentLine(g, agents) : "earlier spec"}
              </span>
            ) : (
              <span className={ID_LINE}>
                <ModelName id={g.model_id} />
                <HfButton id={g.model_id} />
              </span>
            )}
            <InLibraryBadge g={g} />
            {opts?.runCount && opts.runCount > 1 ? (
              <button
                type="button"
                className="shrink-0 inline-flex items-center gap-0.5 px-1.5 py-px rounded-md border border-gray-200 bg-white text-11 font-medium text-gray-600 hover:bg-gray-50"
                title={opts.expanded ? "Hide older runs" : `Show ${opts.runCount - 1} older run${opts.runCount === 2 ? "" : "s"}`}
                aria-expanded={opts.expanded}
                onClick={(e) => {
                  e.stopPropagation();
                  opts.onToggleExpand?.();
                }}
              >
                {opts.runCount} runs
                <ChevronRight className={`${ICON} text-gray-400 ${opts.expanded ? "rotate-90" : ""}`} />
              </button>
            ) : null}
            {g.superseded_by && !opts?.nested ? (
              <span className="basis-full text-11 text-slate-500 truncate" title="The Inspector serves the library spec for this model, not this record.">
                Superseded by library spec <span className="font-mono">{g.superseded_by}</span>
              </span>
            ) : null}
            {job ? (
              <span className="shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-11 font-medium bg-amber-50 text-amber-700 border border-amber-200">
                <Spinner />{job.kind === "review" ? "Reviewing…" : job.kind === "fix" ? "Fixing…" : job.kind === "recheck" ? "Re-checking…" : g.pr_url ? "Updating PR…" : "Opening PR…"}
              </span>
            ) : null}
          </div>
          <div className={META_GROUP}>
            <div className={STATUS_COL}>
              <span className="shrink-0 min-w-[4.75rem]"><RecordStatus g={g} /></span>
              {verdictBadge(g.latest_verdict, !!g.library_spec)}
            </div>
            <span className={RELEASE_COL} title={g.release_date ? `Published on Hugging Face ${g.release_date}` : "No Hugging Face createdAt on record"}>
              {publishedDate(g.release_date)}
            </span>
            <span className={META_COL} title={g.created_at ? new Date(g.created_at).toLocaleString() : ""}>
              {agentLabel(agents, g.agent_cli)}{g.agent_cli ? " · " : ""}{g.agent_model ? `${g.agent_model} · ` : ""}{shortDate(g.created_at)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {g.pr_url ? (
            <a
              href={g.pr_url}
              target="_blank"
              rel="noopener"
              className="p-1 rounded-md text-gray-400 hover:text-amber-700 hover:bg-amber-100/60"
              title="Open the pull request on GitHub"
              aria-label="Open pull request"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalIcon />
            </a>
          ) : (
            <span className="w-[1.375rem]" aria-hidden />
          )}
          <ChevronRight className={`${ICON} text-gray-300 transition-transform group-hover:text-amber-600 group-hover:translate-x-0.5`} />
        </div>
      </div>
    );
  };

  const renderGroup = (g: { key: string; head: Row; rest: Row[] }) => {
    const n = 1 + g.rest.length;
    if (n === 1) return renderRow(g.head);
    const expanded = !!openGroups[g.key];
    const toggle = () => setOpenGroups((p) => ({ ...p, [g.key]: !p[g.key] }));
    return (
      <div key={g.key} className="divide-y divide-gray-100">
        {renderRow(g.head, {
          runCount: n,
          expanded,
          onToggleExpand: toggle,
        })}
        {expanded ? g.rest.map((m) => renderRow(m, { nested: true })) : null}
      </div>
    );
  };

  const bulkDelete = () => {
    void (async () => {
      const sel = selectableKeys.filter((k) => selected[k]);
      if (!sel.length) return;
      if (!(await confirmDialog({ title: `Delete ${sel.length} item${sel.length === 1 ? "" : "s"}?`,
        message: "Spec records' reviews go with them. This cannot be undone.",
        confirmLabel: "Delete", danger: true }))) return;
      let ok = 0;
      let fail = 0;
      for (const k of sel) {
        const kind = k.slice(0, k.indexOf(":"));
        const id = k.slice(k.indexOf(":") + 1);
        const path = kind === "log"
          ? `/api/account/gen-logs/${encodeURIComponent(id)}`
          : `/api/account/generations/${encodeURIComponent(id)}`;
        try {
          await accountJson("DELETE", path);
          ok++;
        } catch {
          fail++;
        }
      }
      setGenStatus(`Deleted ${ok}${fail ? `, ${fail} failed` : ""}.`);
      setSelected({});
      void loadGens();
    })();
  };

  const startPoll = (key: string, tick: () => Promise<boolean>, ms = 3000) => {
    const run = async () => {
      const done = await tick();
      if (!done) pollers.current[key] = window.setTimeout(() => void run(), ms);
    };
    if (pollers.current[key]) clearTimeout(pollers.current[key]);
    // First look right away (a re-attached run shows its real elapsed time and
    // steps at once), then every `ms`.
    pollers.current[key] = window.setTimeout(() => void run(), 0);
  };

  // Never throws: a failed read is kept per record and shown in its
  // Reviews section, the reviews already shown stay.
  const loadReviews = async (id: string) => {
    try {
      const data = await accountJson<{ items?: Review[] }>("GET", `/api/account/generations/${encodeURIComponent(id)}/reviews`);
      setReviews((prev) => ({ ...prev, [id]: data.items || [] }));
      setReviewsErr((prev) => {
        if (!(id in prev)) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });
    } catch (e) {
      setReviewsErr((prev) => ({ ...prev, [id]: e instanceof Error ? e.message : String(e) }));
    } finally {
      setReviewsLoaded((prev) => ({ ...prev, [id]: true }));
    }
  };

  // Re-fetch a record's full payload without clearing the drawer (used after a
  // fix rewrites the spec, or a PR action changes status). No-op if the drawer
  // has since moved to another record.
  const reloadDetail = (id: string) => {
    void accountJson<GenDetail>("GET", `/api/account/generations/${encodeURIComponent(id)}`)
      .then((d) => setDetail((cur) => (detailIdRef.current === id ? d : cur)))
      .catch(() => { /* transient — keep showing what we have */ });
  };

  const openDetail = (id: string) => {
    detailIdRef.current = id;
    setDetailId(id);
    setDetail(null);
    setDetailErr("");
    setReviewsLoaded((prev) => ({ ...prev, [id]: false }));
    void loadReviews(id);
    attachRunningJobs(id);
    void accountJson<GenDetail>("GET", `/api/account/generations/${encodeURIComponent(id)}`)
      .then((d) => { if (detailIdRef.current === id) setDetail(d); })
      .catch((e) => { if (detailIdRef.current === id) setDetailErr(e instanceof Error ? e.message : String(e)); });
  };

  const closeDetail = () => {
    detailIdRef.current = null;
    setDetailId(null);
    setDetail(null);
    setDetailErr("");
  };

  // Shared by spec rows (via their paired log_id) and orphan log rows.
  const openLog = (logId: string) => {
    void (async () => {
      setLogModal({ title: "Loading…", body: "" });
      try {
        const row = await accountJson<GenLog>("GET", `/api/account/gen-logs/${encodeURIComponent(logId)}`);
        const verdict = row.success === true ? "success" : row.success === false ? "failed" : "unknown";
        const kind = logKind(row);
        setLogModal({
          title: `${row.model_id} · ${kind || "run"} · ${verdict} · ${row.created_at ? new Date(row.created_at).toLocaleString() : ""}`,
          body: row.log || "(empty log)",
        });
      } catch (e) {
        setLogModal({ title: "Error", body: "Could not load log: " + (e instanceof Error ? e.message : String(e)) });
      }
    })();
  };

  const flash = (id: string, text: string, tone: ActionMsg["tone"] = "ok") =>
    setActionMsg((p) => ({ ...p, [id]: { text, tone } }));

  const dropJob = (id: string) =>
    setJobs((p) => {
      if (!p[id]) return p;
      const n = { ...p };
      delete n[id];
      return n;
    });

  // Follow one background agent run to completion, streaming its log lines
  // into the job's `steps` so the drawer can show what it is doing right now.
  const trackJob = (id: string, kind: AgentJob["kind"]) => {
    setJobs((p) => (p[id]?.kind === kind ? p : { ...p, [id]: { kind, startedAt: Date.now(), steps: [] } }));
    let cursor = 0;
    startPoll(`${kind}-${id}`, async () => {
      try {
        const s = await accountJson<JobStatus>("GET", jobStatusPath(id, kind, cursor));
        if (s.state === "idle" || s.state === "interrupted") {
          dropJob(id);
          flash(id, s.state === "interrupted"
            ? (s.message || "Interrupted by a server restart. You can retry.")
            : "The job is gone (the server may have restarted). You can retry.", "err");
          void loadReviews(id);
          return true;
        }
        cursor = s.next_cursor ?? cursor;
        const lines = (s.events || []).filter((e) => e.type === "log" && e.text).map((e) => e.text as string);
        setJobs((p) => {
          const j = p[id];
          if (!j) return p;
          return { ...p, [id]: {
            ...j,
            startedAt: s.started_at ? s.started_at * 1000 : j.startedAt,
            steps: lines.length ? [...j.steps, ...lines].slice(-60) : j.steps,
          } };
        });
        if (s.state !== "done") return false;
        dropJob(id);
        if (kind === "pr") {
          const done = (s.events || []).find((e) => e.type === "done");
          await loadReviews(id);
          void loadGens();
          reloadDetail(id);
          if (s.success) flash(id, done?.draft ? `Draft PR ready: ${s.message || ""}` : (s.message || "PR ready."));
          else flash(id, s.message || "The PR could not be opened.", "err");
          return true;
        }
        if (kind === "review") {
          const fixable = (s.events || []).some((e) => e.type === "done" && e.fixable === true);
          setFixableLoad((p) => ({ ...p, [id]: fixable }));
          await loadReviews(id);
          void loadGens();  // refresh the row's latest_verdict badge
          if (s.success === false) flash(id, s.message || "The review did not complete — try again.", "err");
          return true;
        }
        if (kind === "recheck") {
          void loadGens();
          reloadDetail(id);
          flash(id, s.message || "Re-check finished.", s.success === false ? "err" : "ok");
          return true;
        }
        // fix finished
        if (s.success === false) {
          flash(id, s.message || "The fix did not complete — see the review report.", "err");
          return true;
        }
        // A pending record's fix lands on the working candidate, which may be
        // another record: review the one that holds the fix now.
        const landed = (s.events || []).find((e) => e.type === "done")?.record?.id || id;
        void loadGens();
        if (landed !== id) openDetail(landed);
        else reloadDetail(id);  // the fix rewrote the spec bundle in place
        flash(landed, "Fix applied — re-reviewing the updated spec.");
        await startReview(landed);
        return true;
      } catch (e) {
        if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
          dropJob(id);
          flash(id, "Sign-in expired. Sign in again to reconnect.", "err");
          return true;
        }
        /* transient — keep polling */
      }
      return false;
    }, 2000);
  };

  // On opening a record, attach to any review/fix already running for it
  // (started earlier, or before a page reload).
  const attachRunningJobs = (id: string) => {
    for (const kind of ["fix", "review", "pr", "recheck"] as const) {
      void accountJson<JobStatus>("GET", jobStatusPath(id, kind, 0))
        .then((s) => { if (s.state === "running") trackJob(id, kind); })
        .catch((e) => flash(id, `Could not check for a running ${kind}: ${e instanceof Error ? e.message : String(e)}`, "err"));
    }
  };

  async function startReview(id: string) {
    try {
      const r = await accountJson<{ status?: string; message?: string }>(
        "POST", `/api/account/generations/${encodeURIComponent(id)}/review`);
      if (r.status === "started" || r.status === "already_running") trackJob(id, "review");
      else flash(id, r.message || "Could not start the review.", "err");
    } catch (e) {
      flash(id, e instanceof Error ? e.message : String(e), "err");
    }
  }

  const stopJob = async (id: string, kind: "review" | "fix") => {
    try {
      const r = await accountJson<{ success?: boolean; message?: string }>(
        "POST", "/api/spec/stop", { kind, record_id: id, model_id: "" });
      if (!r.success) flash(id, r.message || "No running task found.", "err");
    } catch (e) {
      flash(id, e instanceof Error ? e.message : String(e), "err");
    }
  };

  const startRecheck = async (id: string) => {
    try {
      const r = await accountJson<{ status?: string; message?: string }>(
        "POST", `/api/account/generations/${encodeURIComponent(id)}/recheck`);
      if (r.status === "started" || r.status === "already_running") trackJob(id, "recheck");
      else flash(id, r.message || "Could not start the re-check.", "err");
    } catch (e) {
      flash(id, e instanceof Error ? e.message : String(e), "err");
    }
  };

  const startFix = async (id: string) => {
    try {
      const r = await accountJson<{ status?: string; message?: string }>(
        "POST", `/api/account/generations/${encodeURIComponent(id)}/fix`);
      if (r.status === "started" || r.status === "already_running") trackJob(id, "fix");
      else flash(id, r.message || "Could not start the fix.", "err");
    } catch (e) {
      flash(id, e instanceof Error ? e.message : String(e), "err");
    }
  };

  const prAction = async (id: string, which: "create-pr" | "update-pr" | "rebase-pr",
                          draftAnyway = false) => {
    // Don't clobber a running review/fix — the button is disabled then.
    // The set also absorbs a double-click before the next render.
    if (jobs[id]?.kind === "review" || jobs[id]?.kind === "fix") return;
    if (prInflight.current.has(id)) return;
    prInflight.current.add(id);
    setJobs((p) => ({ ...p, [id]: { kind: "pr", startedAt: Date.now(), steps: ["Starting…"] } }));
    let started = false;
    try {
      const r = await accountJson<{ status?: string; message?: string; pr_url?: string }>(
        "POST", `/api/account/generations/${encodeURIComponent(id)}/${which}`,
        which === "create-pr" ? { draft_anyway: draftAnyway } : undefined);
      if (r.status === "started" || r.status === "already_running") {
        started = true;
        trackJob(id, "pr");
        return;
      }
      if (r.pr_url) {
        flash(id, "This record already has a PR.");
        reloadDetail(id);
        return;
      }
      flash(id, r.message || "The latest review must be PASS first.", "err");
    } catch (e) {
      flash(id, e instanceof Error ? e.message : String(e), "err");
    } finally {
      prInflight.current.delete(id);
      if (!started) dropJob(id);
    }
  };

  const rerunAdmission = async (id: string) => {
    try {
      const r = await accountJson<{ status?: string; message?: string }>(
        "POST", `/api/account/generations/${encodeURIComponent(id)}/admission`);
      if (r.status === "started" || r.status === "already_running") {
        flash(id, "Admission started — measuring this spec against main.");
        void loadReviews(id);
      } else flash(id, r.message || "Could not start the admission.", "err");
    } catch (e) {
      flash(id, e instanceof Error ? e.message : String(e), "err");
    }
  };

  // While the open record's admission is measuring, re-read its reviews so
  // the line under the PASS turns into the report when it lands.
  const admissionRunning = !!detailId && (reviews[detailId] || []).some((r) => r.admission?.job === "running");
  useEffect(() => {
    if (!admissionRunning || !detailId) return;
    const id = detailId;
    const t = window.setInterval(() => void loadReviews(id), 5000);
    return () => clearInterval(t);
  }, [admissionRunning, detailId]);

  // 1s clock for the elapsed-time readout, only while something is running.
  const anyJob = Object.keys(jobs).length > 0;
  useEffect(() => {
    if (!anyJob) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [anyJob]);

  if (!auth?.user) {
    return <p className="p-8 text-sm text-gray-500">Loading account…</p>;
  }

  return (
    <div className="bg-gray-50 min-h-screen" data-account-page="workspace">
      <header className="sticky top-0 z-30 bg-white/90 backdrop-blur border-b border-gray-200">
        <div className="page-shell h-14 flex items-center justify-between">
          <Link to="/" className="inline-flex items-center gap-2 font-semibold text-gray-900 hover:text-amber-600">
            <img src="/static/logo.png" alt="HuggingArch" className="w-7 h-7 rounded-lg" />
            HuggingArch
          </Link>
          <button
            type="button"
            className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
            onClick={() => void signOut().then(() => { window.location.href = "/"; })}
          >
            Sign out
          </button>
        </div>
      </header>

      <div className="page-shell py-8 flex flex-col md:flex-row gap-6">
        <aside className="md:w-60 shrink-0">
          <div className="flex items-center gap-2.5 px-2 pb-3 mb-2 border-b border-gray-100">
            <AvatarFace url={avatarUrl} text={name} size="w-9 h-9 text-sm" />
            <div className="min-w-0">
              <div className="text-sm font-semibold text-gray-900 truncate">{name}</div>
              <div className="text-xs text-gray-500 truncate mt-0.5">{email}</div>
            </div>
          </div>
          <AccountNav
            active={panel}
            isAdmin={auth.isAdmin}
            localPanels={WORKSPACE_PANELS}
            onPanelSelect={(p) => {
              if ((WORKSPACE_PANELS as readonly string[]).includes(p)) setPanel(p as WorkspacePanel);
            }}
          />
        </aside>

        <main className="flex-1 min-w-0">
          {panel === "generations" && (
            <section className={CARD}>
              <p className="text-10 font-semibold uppercase tracking-wider text-amber-700 mb-1">Specs</p>
              <h2 className="text-lg font-semibold text-gray-900 mb-1">My spec generations</h2>
              <p className="text-sm text-gray-500 mb-4 max-w-xl leading-relaxed">
                Specs you generated for unsupported models. Open one to see its YAML,
                run a <strong>Review</strong>, and — once it passes — <strong>open the PR</strong>{" "}
                that contributes it to the library, all yourself.
              </p>
              {genStatus ? <div className="mb-3 text-xs text-amber-700">{genStatus}</div> : null}
              {genErr ? <div className="text-sm text-red-600">Could not load: {genErr}</div> : null}
              {!genErr && !gensReady ? (
                <div className="text-sm text-gray-400 border border-dashed border-gray-200 rounded-xl px-4 py-6 text-center flex items-center justify-center gap-2">
                  <Spinner />Loading your specs…
                </div>
              ) : null}
              {!genErr && gensReady && rows.length === 0 && supersededGroups.length === 0 ? (
                <div className="text-sm text-gray-400 border border-dashed border-gray-200 rounded-xl px-4 py-6 text-center">
                  No generations yet. Generate a spec for an unsupported model from the Inspector.
                </div>
              ) : null}
              {activeRows.length > 0 && (
                <div className="border border-gray-200 rounded-xl overflow-hidden">
                  <div className="flex items-center gap-3 px-3 sm:px-4 h-10 bg-gray-50 border-b border-gray-200 text-xs">
                    <input
                      type="checkbox"
                      className="rounded border-gray-300 shrink-0"
                      checked={allSelected}
                      disabled={selectableKeys.length === 0}
                      aria-label="Select all"
                      title={selectableKeys.length ? "Select every deletable item" : "Nothing here can be deleted — every spec has an open PR."}
                      onChange={(e) => {
                        const on = e.target.checked;
                        setSelected(() => {
                          if (!on) return {};
                          const n: Record<string, boolean> = {};
                          for (const k of selectableKeys) n[k] = true;
                          return n;
                        });
                      }}
                    />
                    <span className={`tabular-nums ${selectedCount ? "font-medium text-gray-800" : "text-gray-500"}`}>
                      {selectedCount
                        ? `${selectedCount} selected`
                        : activeGroups.length === activeRows.length
                          ? `${activeRows.length} item${activeRows.length === 1 ? "" : "s"}`
                          : `${activeGroups.length} models · ${activeRows.length} runs`}
                    </span>
                    <button
                      type="button"
                      disabled={selectedCount === 0}
                      className={`ml-auto inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md font-medium transition-colors ${
                        selectedCount
                          ? "text-red-600 bg-red-50 border border-red-200 hover:bg-red-100"
                          : "text-gray-300 border border-transparent cursor-default"
                      }`}
                      onClick={bulkDelete}
                    >
                      <TrashIcon />
                      Delete selected
                    </button>
                  </div>
                  {/* Column captions — the two right-hand dates are
                      different facts (HF publish date vs this run's time)
                      and the rows alone don't say which is which. Mirrors
                      the row layout: checkbox gutter, name, status,
                      release, meta, actions gutter. Hidden below xl,
                      where the rows stack. */}
                  <div className="hidden xl:flex items-center gap-3 px-4 pt-2 pb-1 text-11 font-medium uppercase tracking-wide text-gray-400">
                    <span className="w-4 shrink-0" aria-hidden />
                    <span className="flex-1 min-w-0 xl:basis-52">Model</span>
                    <span className={STATUS_COL}>Status</span>
                    <span className={RELEASE_COL} title="The model's createdAt on Hugging Face — when it was published, not when the spec ran">Model released</span>
                    <span className={META_COL} title="When this spec was generated and by which agent">Last run</span>
                    <span className="w-16 shrink-0" aria-hidden />
                  </div>
                  <div className="divide-y divide-gray-100">{activeGroups.map(renderGroup)}</div>
                </div>
              )}
              {archivedRows.length > 0 && (
                <details className="mt-4 group/arch border border-gray-200 rounded-xl overflow-hidden">
                  <summary className="flex items-center gap-2 px-3 sm:px-4 h-10 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden bg-gray-50 hover:bg-gray-100/70 text-xs">
                    <ChevronRight className={`${ICON} text-gray-400 transition-transform group-open/arch:rotate-90`} />
                    <span className="font-medium text-gray-700">Archived (merged or closed)</span>
                    <span className="inline-flex items-center gap-1 px-1.5 py-px rounded-full bg-green-50 text-green-700 font-medium tabular-nums">
                      {archivedRows.filter((r) => r.kind === "gen" && r.status === "merged").length} merged
                    </span>
                    {archivedRows.some((r) => r.kind === "gen" && r.status === "closed") ? (
                      <span className="inline-flex items-center gap-1 px-1.5 py-px rounded-full bg-slate-100 text-slate-600 font-medium tabular-nums">
                        {archivedRows.filter((r) => r.kind === "gen" && r.status === "closed").length} closed
                      </span>
                    ) : null}
                    <span className="ml-auto text-gray-400 hidden sm:inline">Finished PRs · read-only</span>
                  </summary>
                  <div className="divide-y divide-gray-100 border-t border-gray-200">{archivedGroups.map(renderGroup)}</div>
                </details>
              )}
              {supersededGroups.length > 0 && (
                <details className="mt-4 group/sup border border-gray-200 rounded-xl overflow-hidden">
                  <summary className="flex items-center gap-2 px-3 sm:px-4 h-10 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden bg-gray-50 hover:bg-gray-100/70 text-xs">
                    <ChevronRight className={`${ICON} text-gray-400 transition-transform group-open/sup:rotate-90`} />
                    <span className="font-medium text-gray-700">Archived (superseded by library spec)</span>
                    <span className="inline-flex items-center gap-1 px-1.5 py-px rounded-full bg-slate-100 text-slate-600 font-medium tabular-nums">
                      {supersededGroups.length}
                    </span>
                    <span className="ml-auto text-gray-400 hidden sm:inline">The library now has these models · view or delete</span>
                  </summary>
                  <div className="divide-y divide-gray-100 border-t border-gray-200">{supersededGroups.map(renderGroup)}</div>
                </details>
              )}
            </section>
          )}

          {panel === "modules" && (
            <section className={CARD}>
              <p className="text-10 font-semibold uppercase tracking-wider text-amber-700 mb-1">Workspace</p>
              <h2 className="text-lg font-semibold text-gray-900 mb-1">My modules</h2>
              <p className="text-sm text-gray-500 mb-4 max-w-xl leading-relaxed">
                Custom parameterizations saved from the Playground (edit a bound column's
                dims → ☆ save). Usable there under Bound models → mine.
              </p>
              {variantErr ? (
                <div className="mb-3 text-xs text-red-600 flex items-center gap-2">
                  <span>{variantErr}</span>
                  <button
                    type="button"
                    className="px-2 py-0.5 rounded border border-red-200 text-red-600 hover:bg-red-50"
                    onClick={() => { setVariantErr(""); void loadVariants(); }}
                  >
                    Retry
                  </button>
                </div>
              ) : null}
              {variants === null ? (
                <div className="text-xs text-gray-400">{variantErr ? "Could not load your modules." : "Loading…"}</div>
              ) : variants.length === 0 ? (
                <div className="text-xs text-gray-400">
                  No saved modules yet — customize a column in the Playground and hit ☆ save.
                </div>
              ) : (
                <ul className="divide-y divide-gray-100">
                  {variants.map((v) => (
                    <li key={v.id} className="py-2 flex items-start gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="text-sm">
                          <span className="font-mono text-gray-900">{v.component}</span>
                          {" "}<span className="text-sky-700">· {v.name}</span>
                          {v.base_model_id && (
                            <span className="text-xs text-gray-400"> — tuned under {v.base_model_id}</span>
                          )}
                        </div>
                        <div className="text-11 font-mono text-gray-500 truncate">
                          {Object.entries(v.dims).map(([k, x]) => `${k}=${x}`).join(", ")}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => void renameVariant(v)}
                        className="text-xs px-2 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-50"
                      >
                        rename
                      </button>
                      <button
                        type="button"
                        onClick={() => void deleteVariant(v)}
                        className="text-xs px-2 py-1 rounded border border-red-200 text-red-500 hover:bg-red-50"
                      >
                        delete
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}

          {panel === "benchmarks" && <BenchmarksPanel />}
        </main>
      </div>

      {detailId && (
        <div className="fixed inset-0 z-40" onClick={closeDetail}>
          <div className="absolute inset-0 bg-black/30" />
          <aside className="absolute right-0 top-0 h-full w-full md:w-[720px] bg-white shadow-xl overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="text-sm font-mono font-semibold text-gray-900 truncate">{detail?.model_id || allGens.find((x) => x.id === detailId)?.model_id || "…"}</div>
                <div className="text-11 text-gray-500 mt-1 flex items-center gap-2 flex-wrap">
                  {detail ? (
                    <>
                      <RecordStatus g={detail} />
                      <InLibraryBadge g={detail} />
                      {verdictBadge((reviews[detailId] || [])[0]?.verdict || allGens.find((x) => x.id === detailId)?.latest_verdict, !!detail.library_spec)}
                      <span>{agentLabel(agents, detail.agent_cli)}{detail.agent_cli ? " · " : ""}{detail.agent_model ? `${detail.agent_model} · ` : ""}{detail.created_at ? new Date(detail.created_at).toLocaleString() : ""}</span>
                    </>
                  ) : detailErr ? <span className="text-red-600">Failed: {detailErr}</span> : <span>Loading…</span>}
                </div>
              </div>
              <button type="button" className="text-2xl text-gray-400 hover:text-gray-700 leading-none shrink-0" onClick={closeDetail}>×</button>
            </div>
            {detail && (() => {
              const d = detail;
              const id = detailId;
              // The detail payload omits logs; they are paired onto the list
              // row by _attach_run_logs, so read them from there.
              const listRow = allGens.find((x) => x.id === id);
              const rowLogId = listRow?.log_id;
              const fixLogs = listRow?.fix_logs || [];
              const reviewLogs = listRow?.review_logs || [];
              const latest = (reviews[id] || [])[0];
              const job = jobs[id];
              const busy = !!job;
              const passed = latest?.verdict === "pass";
              const loadFix = !!fixableLoad[id];
              // A pending record is re-checked, not reviewed; an engine-blocked
              // one has nothing to fix, a needs_repair one opens Fix on the
              // errors its re-check found.
              const pending = isPendingStatus(d.status);
              const blocked = d.status === "blocked_by_engine";
              const showFix = !blocked && (latest?.verdict === "no_pass" || loadFix || d.status === "needs_repair");
              const deletable = genDeletable(d);
              // Archived: the library spec replaced it, so the backend
              // refuses Review / Fix / PR on it — none are offered.
              const superseded = !!d.superseded_by;
              const msg = actionMsg[id];
              // The Open PR button follows the latest PASS's admission.
              const gate = passed ? prGate(admissionView(latest), !!auth.isAdmin) : null;
              const prBlocked = pending ? "Re-check this spec first: only a valid spec opens a PR."
                : job?.kind === "pr" ? "The pull request is being prepared."
                : busy ? "Wait for the running review to finish."
                : !passed ? "Needs a PASS review first — run Review."
                : !d.pr_url && gate?.disabled ? gate.reason : "";
              const BTN_2 = "text-xs px-3 py-1.5 border rounded-lg disabled:opacity-40 disabled:cursor-not-allowed";
              return (
                <div className="px-6 py-5">
                  {superseded ? (
                    <div className="mb-3 px-3 py-2 rounded-lg border border-slate-200 bg-slate-50 text-xs text-slate-600">
                      Archived — superseded by library spec <span className="font-mono">{d.superseded_by}</span>. The Inspector serves that spec for this model; this record can be viewed or deleted.
                    </div>
                  ) : null}
                  <div className="flex items-center gap-2 flex-wrap">
                    {superseded ? null : <a href={`/inspector.html#model=${encodeURIComponent(d.model_id || "")}`} target="_blank" rel="noopener" className="text-xs px-3 py-1.5 rounded-lg bg-amber-500 text-white font-medium hover:bg-amber-600" title="Analyze in the Inspector using your generated spec">Open in Inspector ↗</a>}
                    {d.pr_url ? <a href={d.pr_url} target="_blank" rel="noopener" className="text-xs px-3 py-1.5 border border-amber-300 text-amber-700 rounded-lg hover:bg-amber-50 font-medium">View PR ↗</a> : null}
                    {superseded ? null : pending ? (
                      <button
                        type="button"
                        disabled={busy || (blocked && !d.engine_changed)}
                        className={`${BTN_2} border-violet-300 text-violet-700 hover:bg-violet-50 font-medium`}
                        data-action="recheck"
                        title={busy ? "A job is already running on this record."
                          : blocked && !d.engine_changed
                            ? "The engine has not changed since this spec was blocked; a re-check would find the same gaps."
                            : "Validate the saved spec again against the current engine. No agent runs."}
                        onClick={() => { setActionMsg((p) => { const n = { ...p }; delete n[id]; return n; }); void startRecheck(id); }}
                      >
                        {job?.kind === "recheck" ? "Re-checking…" : "Re-check"}
                      </button>
                    ) : job?.kind === "review" ? (
                      <button
                        type="button"
                        className={`${BTN_2} border-red-200 text-red-600 hover:bg-red-50`}
                        title="Stop the running review."
                        onClick={() => void stopJob(id, "review")}
                      >
                        Stop review
                      </button>
                    ) : (
                      <button
                        type="button"
                        disabled={busy}
                        className={`${BTN_2} border-gray-300 hover:bg-gray-50`}
                        title={busy ? "A review or fix is already running." : "Run an agent review of this spec against the model's real architecture (PASS / NO_PASS)."}
                        onClick={() => { setActionMsg((p) => { const n = { ...p }; delete n[id]; return n; }); void startReview(id); }}
                      >
                        Review
                      </button>
                    )}
                    {superseded ? null : job?.kind === "fix" ? (
                      <button
                        type="button"
                        className={`${BTN_2} border-red-200 text-red-600 hover:bg-red-50 font-medium`}
                        title="Stop the running fix."
                        onClick={() => void stopJob(id, "fix")}
                      >
                        Stop fix
                      </button>
                    ) : showFix ? (
                      <button
                        type="button"
                        disabled={busy}
                        className={`${BTN_2} border-indigo-300 text-indigo-700 hover:bg-indigo-50 font-medium`}
                        title={loadFix
                          ? "Have an agent repair the errors that stop this spec from loading, then re-review."
                          : "Have an agent fix the defects the latest NO_PASS review named, then re-review."}
                        onClick={() => { setActionMsg((p) => { const n = { ...p }; delete n[id]; return n; }); void startFix(id); }}
                      >
                        Agent Fix
                      </button>
                    ) : null}
                    {superseded || pending ? null : d.status === "rejected" ? (
                      <span className="text-xs px-2 py-1 text-gray-400">Not accepted</span>
                    ) : d.pr_url ? (
                      <>
                      <button
                        type="button"
                        disabled={!!prBlocked}
                        className={`${BTN_2} border-amber-300 text-amber-700 hover:bg-amber-50 font-medium`}
                        title={prBlocked || "Rewrite the PR branch from current main as one commit with this spec."}
                        onClick={() => void prAction(id, "rebase-pr")}
                      >
                        {job?.kind === "pr" ? "Rebasing…" : "Rebase onto main"}
                      </button>
                      <button
                        type="button"
                        disabled={!!prBlocked}
                        className={`${BTN_2} border-amber-300 text-amber-700 hover:bg-amber-50 font-medium`}
                        title={prBlocked || "Push this spec's current content to its open PR (also onto latest main)."}
                        onClick={() => void prAction(id, "update-pr")}
                      >
                        {job?.kind === "pr" ? "Updating PR…" : "Update PR"}
                      </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        disabled={!!prBlocked}
                        className="text-xs px-3 py-1.5 rounded-lg bg-amber-500 text-white font-medium hover:bg-amber-600 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed"
                        title={prBlocked || gate?.reason || "Open the PR contributing this spec to the library."}
                        onClick={() => void prAction(id, "create-pr")}
                      >
                        {job?.kind === "pr" ? "Opening PR…" : gate?.label || "Open PR"}
                      </button>
                    )}
                    {!superseded && !pending && !d.pr_url && gate?.draftAnyway && !busy ? (
                      <button
                        type="button"
                        className={`${BTN_2} border-red-300 text-red-700 hover:bg-red-50 font-medium`}
                        title="Admin: open a draft PR despite failed corpus checks, to put the rule fix in the same PR."
                        onClick={async () => {
                          if (!(await confirmDialog({ title: "Open a draft PR anyway?",
                            message: "The corpus checks fail for this spec.", confirmLabel: "Open draft PR" }))) return;
                          void prAction(id, "create-pr", true);
                        }}
                      >
                        Open draft PR anyway
                      </button>
                    ) : null}
                    {deletable && (
                      <button
                        type="button"
                        disabled={busy}
                        className={`ml-auto ${BTN_2} border-red-200 text-red-600 hover:bg-red-50`}
                        title={busy ? "Wait for the running job to finish." : "Delete this generation record. This cannot be undone."}
                        onClick={async () => {
                          if (!(await confirmDialog({ title: "Delete this generation record?",
                            message: "Its reviews go with it. This cannot be undone.",
                            confirmLabel: "Delete", danger: true }))) return;
                          void accountJson("DELETE", `/api/account/generations/${encodeURIComponent(id)}`).then(() => {
                            setGenStatus("Deleted.");
                            closeDetail();
                            void loadGens();
                          }).catch(alertError);
                        }}
                      >
                        Delete
                      </button>
                    )}
                  </div>
                  {(() => {
                    // Transcripts sit apart from the actions: one quiet line,
                    // newest first, each entry a plain link.
                    const logs = [
                      ...(rowLogId ? [{ id: rowLogId, kind: "Run", at: d.created_at, title: "Run log — the run that produced this spec", verdict: null }] : []),
                      ...fixLogs.map((l) => ({ id: l.id, kind: "Fix", at: l.created_at, title: "Fix log — the agent-fix transcript", verdict: null })),
                      ...reviewLogs.map((l) => ({ id: l.id, kind: "Review", at: l.created_at, verdict: l.verdict || null,
                        title: "Review log — the review transcript. The badge is the verdict recorded, after the validator and the evidence check; the transcript ends on the agent's own." })),
                    ].sort((a, b) => String(b.at || "").localeCompare(String(a.at || "")));
                    return logs.length ? (
                      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-500">
                        <span className="text-gray-400">Logs</span>
                        {logs.map((l) => (
                          <button key={l.id} type="button" title={l.title} onClick={() => openLog(l.id)}
                            className="hover:text-gray-900 hover:underline underline-offset-2">
                            {l.kind}
                            {l.verdict ? (
                              <span data-log-verdict={l.verdict}
                                className={`ml-1 px-1 rounded text-10 font-semibold ${reviewBadgeClass(l.verdict)}`}>
                                {l.verdict.toUpperCase()}
                              </span>
                            ) : null}
                            {l.at ? <span className="text-gray-400"> · {shortDate(l.at)}</span> : null}
                          </button>
                        ))}
                      </div>
                    ) : null;
                  })()}
                  {msg ? (
                    <div className={`mt-2 text-xs ${msg.tone === "err" ? "text-red-600" : "text-green-700"}`}>{msg.text}</div>
                  ) : !superseded && !pending && !busy && !passed && d.status !== "rejected" ? (
                    <div className="mt-2 text-xs text-gray-400">
                      {d.pr_url ? "Update PR" : "Open PR"} unlocks once the latest review is PASS.
                    </div>
                  ) : !superseded && !busy && !d.pr_url && gate?.disabled ? (
                    <div className="mt-2 text-xs text-gray-500" data-pr-gate>{gate.reason}</div>
                  ) : null}

                  {blocked ? (
                    <>
                      <h3 className="text-sm font-semibold text-violet-800 mt-6 mb-1">Waiting on the analysis engine</h3>
                      <p className="text-xs text-gray-600 mb-2">
                        The spec states the model correctly. The engine does not yet support what it
                        needs on these deployments; the Inspector serves the spec on every other one.
                        {d.engine_changed ? " The engine has changed since — re-check it." : ""}
                      </p>
                      <EngineGapTable gaps={d.engine_gaps} />
                    </>
                  ) : d.status === "needs_repair" ? (
                    <>
                      <h3 className="text-sm font-semibold text-amber-800 mt-6 mb-1">Errors a re-check found</h3>
                      <p className="text-xs text-gray-600 mb-2">Agent Fix repairs these; the fixed spec then becomes your working candidate.</p>
                      <ul className="text-xs text-gray-700 list-disc pl-5 space-y-1">
                        {(d.repairable_errors || []).map((e, i) => <li key={i}>{e}</li>)}
                      </ul>
                    </>
                  ) : null}

                  <h3 className="text-sm font-semibold text-gray-700 mt-6 mb-2">Reviews</h3>
                  {job && <AgentJobCard kind={job.kind} startedAt={job.startedAt} steps={job.steps} now={now}
                    onStop={(() => {
                      const kind = job.kind;
                      return kind === "pr" || kind === "recheck" ? undefined : () => void stopJob(id, kind);
                    })()} />}
                  {reviewsErr[id] ? (
                    <div className="text-xs text-red-600 mb-2">Could not load the reviews: {reviewsErr[id]}</div>
                  ) : null}
                  {(reviews[id] || []).length === 0 ? (
                    job || reviewsErr[id] ? null : !reviewsLoaded[id] ? <div className="text-xs text-gray-400">Loading…</div> : <div className="text-xs text-gray-400">No reviews yet — run one above.</div>
                  ) : (
                    <div className="space-y-2">
                      {(reviews[id] || []).map((r, i) => (
                        <div key={i} className="border border-gray-200 rounded-lg overflow-hidden">
                          <div className="bg-gray-50 px-3 py-1.5 text-11 text-gray-600 border-b border-gray-200">
                            <span
                              className={`px-2 py-0.5 rounded-full text-11 font-medium ${d.library_spec ? "bg-gray-100 text-gray-400" : reviewBadgeClass(r.verdict || "error")}`}
                              title={d.library_spec ? LIBRARY_VERDICT_TITLE : undefined}
                            >{(r.verdict || "error").toUpperCase()}</span>
                            <span className="ml-2">{r.reviewer_role || ""} · {r.created_at ? new Date(r.created_at).toLocaleString() : ""}</span>
                          </div>
                          {r.verdict === "pass" ? (
                            <AdmissionLine row={r} onRerun={i === 0 && r.bundle_current !== false && !d.pr_url
                              ? () => void rerunAdmission(id) : undefined} />
                          ) : null}
                          <MarkdownReport text={r.report || ""} />
                        </div>
                      ))}
                    </div>
                  )}

                  <h3 className="text-sm font-semibold text-gray-700 mt-6 mb-2">Files</h3>
                  <SpecFiles modelId={d.model_id} specYaml={d.spec_yaml || ""} components={d.new_component_yamls} />
                </div>
              );
            })()}
          </aside>
        </div>
      )}

      {logModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={() => setLogModal(null)}>
          <div className="bg-white rounded-xl shadow-xl max-w-3xl w-full max-h-[85vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
              <div className="text-xs font-medium text-gray-700 truncate mr-3">{logModal.title}</div>
              <button type="button" className="text-gray-400 hover:text-gray-700 text-2xl leading-none shrink-0" onClick={() => setLogModal(null)}>&times;</button>
            </div>
            <pre ref={logPreRef} className="flex-1 overflow-auto text-11 leading-snug bg-gray-900 text-green-300 p-4 rounded-b-xl whitespace-pre-wrap m-0">{logModal.body}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
