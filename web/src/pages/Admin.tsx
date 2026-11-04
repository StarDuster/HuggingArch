import { useCallback, useEffect, useRef, useState } from "react";
import { AgentJobCard } from "../AgentJob";
import type { EChartsOption } from "echarts";
import { Link, useNavigate } from "react-router-dom";
import { AXIS, HaChart } from "../charts/HaChart";
import { adminRpc, authJson, loadAuth, loginRedirect, signOut, type AuthState } from "../auth";
import { AccountNav } from "../AccountNav";
import { AvatarFace } from "../Avatar";
import { SpecFiles, type SpecFile } from "../SpecFiles";
import { MarkdownReport } from "../Markdown";
import { AdmissionLine, type AdmissionRow } from "../Admission";
import { specStatusLabel } from "../specGen";
import {
  generationBadge, isPendingStatus, RUN_ERROR_KIND_BADGE, gapsTitle,
  type EngineGap, type GenerationVocabulary,
} from "../specStatus";
import { EngineBlockedBadge, EngineGapTable } from "../EngineGaps";
import { DataTable, type ColumnDef } from "../DataTable";
import { BenchReviewModal, benchCorpusHref, benchReviewTarget, peakCheckColumns, type BenchReviewTarget, type PeakCounts } from "./BenchReview";
import { alertError, confirmDialog } from "../Dialog";

type Tab =
  | "users" | "logs" | "active"
  | "trend" | "latency" | "models" | "queries"
  | "cache" | "gens" | "runs" | "benchnoms" | "system";

type UserRow = {
  id: string;
  user_email?: string;
  username?: string | null;
  created_at?: string;
  is_admin?: boolean;
  is_super?: boolean;
  avatar_url?: string | null;
};

// One nominated batch: an upload of its owner's (`kind` upload, `id` its
// upload id — the set reference promote / reject take), and the run it
// belongs to, if any. `row_count` counts the rows promote would share.
type Nom = {
  owner: string;
  kind: "run" | "upload";
  id: string;
  run_id?: string | null;
  submitter?: { email?: string | null; username?: string | null };
  label?: string;
  nominated_at?: string;
  row_count?: number;
  models?: string[];
  frameworks?: string[];
  gpus?: string[];
} & PeakCounts;

function submitterOf(n: Nom): string {
  return n.submitter?.email || n.submitter?.username || n.owner.slice(0, 8) || "—";
}

const nomKey = (n: Nom) => `${n.owner}:${n.kind}:${n.id}`;

const TABS: { group: string; items: { id: Tab; label: string }[] }[] = [
  { group: "People", items: [{ id: "users", label: "Users" }, { id: "logs", label: "Logins" }, { id: "active", label: "Active" }] },
  { group: "Traffic", items: [{ id: "trend", label: "Call trend" }, { id: "latency", label: "Latency" }, { id: "models", label: "Top models" }, { id: "queries", label: "Queries" }] },
  { group: "Content", items: [{ id: "cache", label: "Shared specs" }, { id: "gens", label: "Generations" }, { id: "runs", label: "Runs" }, { id: "benchnoms", label: "Bench nominations" }] },
  { group: "Ops", items: [{ id: "system", label: "System" }] },
];

function ActiveLines({ rows }: { rows: Record<string, unknown>[] }) {
  const series: { key: string; color: string; fill?: string; label: string }[] = [
    { key: "active_users", color: "#F59E0B", fill: "rgba(245,158,11,0.15)", label: "Active users" },
    { key: "new_users", color: "#10B981", label: "New users" },
  ];
  const ordered = [...rows].reverse();
  if (!ordered.length) return null;
  const option: EChartsOption = {
    grid: { left: 36, right: 8, top: 8, bottom: 20 },
    tooltip: { trigger: "axis" },
    legend: { bottom: 0, textStyle: { fontSize: 10 } },
    xAxis: { type: "category", data: ordered.map((_, i) => String(i + 1)), show: false },
    yAxis: { type: "value", ...AXIS, splitLine: { lineStyle: { color: "#f3f4f6" } } },
    series: series.map((s) => ({
      type: "line" as const,
      name: s.label,
      data: ordered.map((r) => Number(r[s.key]) || 0),
      showSymbol: false,
      lineStyle: { color: s.color, width: 1.6 },
      areaStyle: s.fill ? { color: s.fill } : undefined,
    })),
  };
  return <HaChart option={option} height={112} />;
}

function TrendBars({ rows }: { rows: Record<string, unknown>[] }) {
  const series = [
    { key: "analyze_calls", color: "#3B82F6", label: "analyze" },
    { key: "inference_calls", color: "#A855F7", label: "inference_estimate" },
  ];
  if (!rows.length) return null;
  const option: EChartsOption = {
    grid: { left: 36, right: 8, top: 8, bottom: 20 },
    tooltip: { trigger: "axis" },
    legend: { bottom: 0, textStyle: { fontSize: 10 } },
    xAxis: { type: "category", data: rows.map((_, i) => String(i + 1)), show: false },
    yAxis: { type: "value", ...AXIS },
    series: series.map((s) => ({
      type: "bar" as const,
      name: s.label,
      stack: "calls",
      data: rows.map((r) => Number(r[s.key]) || 0),
      itemStyle: { color: s.color },
    })),
  };
  return <HaChart option={option} height={112} />;
}

function unwrapSummary(raw: unknown): Record<string, unknown> {
  if (Array.isArray(raw)) return (raw[0] as Record<string, unknown>) || {};
  if (raw && typeof raw === "object") return raw as Record<string, unknown>;
  return {};
}

function fmtTime(t: unknown): string {
  return t ? new Date(String(t)).toLocaleString() : "—";
}

function UdCard({ label, val, hint }: { label: string; val: string; hint?: string }) {
  return (
    <div className="bg-gray-50 border border-gray-200 rounded p-3">
      <div className="text-10 uppercase tracking-wide text-gray-500">{label}</div>
      <div className="mt-1 text-base font-semibold text-gray-900">{val}</div>
      {hint ? <div className="text-10 text-gray-500 mt-0.5">{hint}</div> : null}
    </div>
  );
}

function isoDaysAgo(days: number): { start: string; end: string } {
  const end = new Date();
  const start = new Date(end.getTime() - days * 86400_000);
  return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
}

function adminApi<T>(method: string, path: string, body?: unknown): Promise<T> {
  return authJson<T>(method, path, body, { keepHash: false });
}

/** Allowlisted PostgREST RPC via the server-side proxy; null on failure. */
async function adminRpcData<T>(name: string, args: Record<string, unknown> = {}): Promise<T | null> {
  try {
    return await authJson<T>("POST", "/api/auth/admin-rpc", { name, args }, { keepHash: false });
  } catch {
    return null;
  }
}

function bytesShort(n: number | null | undefined): string {
  if (n == null) return "–";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

// Sort keys for loosely typed rows: a number / an epoch, or null (sorts last).
function sortNum(v: unknown): number | null {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function sortTime(v: unknown): number | null {
  if (!v) return null;
  const t = Date.parse(String(v));
  return Number.isNaN(t) ? null : t;
}

function GenStatusBadge({ row, prUrl }: { row: Record<string, unknown>; prUrl?: unknown }) {
  const s = row.status;
  if (s === "blocked_by_engine") {
    return <EngineBlockedBadge gaps={row.engine_gaps as EngineGap[] | null} engineChanged={!!row.engine_changed} />;
  }
  const badge = generationBadge(s);
  const label = s === "contributed" ? specStatusLabel(s, prUrl) : badge.label;
  return <span className={`px-2 py-0.5 rounded text-11 font-medium ${badge.cls}`} title={badge.title}>{label}</span>;
}

function RunErrorBadge({ row }: { row: Record<string, unknown> }) {
  const kind = row.error_kind ? String(row.error_kind) : "";
  if (!kind) return <span className="text-xs text-gray-400">—</span>;
  const badge = RUN_ERROR_KIND_BADGE[kind];
  const gaps = row.engine_gaps as EngineGap[] | null;
  const title = [badge?.title, gapsTitle(gaps)].filter(Boolean).join("\n");
  return <span className={`px-2 py-0.5 rounded text-11 font-medium ${badge?.cls || "bg-gray-100 text-gray-600"}`} title={title}>{badge?.label || kind}</span>;
}

// Site-wide default model per (agent, agent task) — the policy an agent run
// falls back to when the user's credential pins no per-task model
// (migration 0039; resolution: per-task pin → this table → credential model
// → endpoint default). Edited cell-by-cell; a blank cell clears the row.
function TaskDefaultsCard() {
  const [agents, setAgents] = useState<string[]>([]);
  const [tasks, setTasks] = useState<{ key: string; label: string }[]>([]);
  const [cells, setCells] = useState<Record<string, string>>({});
  const [status, setStatus] = useState("");
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    void (async () => {
      try {
        const d = await adminApi<{
          defaults: Record<string, Record<string, string>>;
          agents: string[];
          tasks: { key: string; label: string }[];
        }>("GET", "/api/admin/agent-task-defaults");
        setAgents(d.agents || []);
        setTasks(d.tasks || []);
        const c: Record<string, string> = {};
        for (const [cli, m] of Object.entries(d.defaults || {})) {
          for (const [t, mo] of Object.entries(m)) c[`${cli}|${t}`] = mo;
        }
        setCells(c);
      } catch (e) {
        setStatus("Could not load: " + (e instanceof Error ? e.message : String(e)));
      } finally {
        setLoaded(true);
      }
    })();
  }, []);

  const save = async (cli: string, task: string) => {
    setStatus("Saving…");
    try {
      await adminApi("PUT", "/api/admin/agent-task-defaults", {
        agent_cli: cli, task, model: cells[`${cli}|${task}`] || "",
      });
      setStatus("Saved.");
    } catch (e) {
      setStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  const columns: ColumnDef<{ key: string; label: string }>[] = [
    {
      key: "task",
      header: "Task",
      className: "text-xs font-medium text-gray-600",
      render: (t) => t.label,
      sortValue: (t) => t.label,
    },
    ...agents.map((a) => ({
      key: a,
      header: a,
      sortValue: null,
      render: (t: { key: string; label: string }) => {
        const k = `${a}|${t.key}`;
        return (
          <input
            className="w-44 px-2 py-1 text-xs border border-gray-200 rounded bg-white focus:outline-none focus:border-amber-500 font-mono"
            placeholder="—"
            value={cells[k] || ""}
            onChange={(e) => setCells({ ...cells, [k]: e.target.value })}
            onBlur={() => void save(a, t.key)}
          />
        );
      },
    })),
  ];

  return (
    <section className="bg-white border rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-1">Agent task defaults</h3>
      <p className="text-sm text-gray-500 mb-3 max-w-xl">
        The model an agent run (spec generate / refine / review / fix) uses when the
        user's credential pins nothing for that task. A run's model resolves as:
        per-task pin → this table → the credential's model → the endpoint default.
        Blank = no site default. A cell saves on blur.
      </p>
      <DataTable columns={columns} rows={tasks} rowKey={(t) => t.key} framed={false} tableClassName="text-sm" loading={!loaded} />
      {status ? <p className="text-xs text-gray-400 mt-2">{status}</p> : null}
    </section>
  );
}

const REVIEW_STYLE: Record<string, string> = {
  pass: "bg-green-100 text-green-700",
  no_pass: "bg-red-100 text-red-700",
  error: "bg-gray-100 text-gray-500",
};

// `historical`: the record's model is served by a library spec, so the
// review judged a spec the product does not use — greyed, kept.
function ReviewBadge({ verdict, historical = false }: { verdict: unknown; historical?: boolean }) {
  const s = String(verdict || "error");
  const cls = historical ? "bg-gray-100 text-gray-400" : REVIEW_STYLE[s] || REVIEW_STYLE.error;
  return (
    <span
      className={`px-2 py-0.5 rounded-full text-11 font-medium ${cls}`}
      title={historical ? "Review of this record's own spec; the library spec is what the product uses" : undefined}
    >{s.toUpperCase()}</span>
  );
}

// A record against the library (backend `library_standing`): a merged
// record whose model the library serves is "in library"; any other record
// with a library spec is superseded by it.
function LibraryBadge({ row }: { row: Record<string, unknown> }) {
  const lib = row.library_spec ? String(row.library_spec) : "";
  if (!lib) return null;
  const superseded = !!row.superseded_by;
  return (
    <span
      className={`ml-1 px-2 py-0.5 rounded text-11 font-medium ${superseded ? "bg-slate-100 text-slate-500" : "bg-green-50 text-green-700"}`}
      title={superseded
        ? `Superseded by library spec ${lib}.yaml — the Inspector serves that spec, not this record`
        : `In the library as ${lib}.yaml — the Inspector serves that spec for this model`}
    >{superseded ? "superseded" : "in library"}</span>
  );
}

// `admission` / `admission_unavailable` (migration 0038): a PASS's corpus
// admission, shown read-only — the contributor opens the PR.
type GenReview = AdmissionRow & {
  verdict?: string;
  reviewer_role?: string;
  reviewer_email?: string;
  created_at?: string;
  report?: string;
};

export function Admin() {
  const nav = useNavigate();
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [denied, setDenied] = useState(false);
  const [tab, setTab] = useState<Tab>("users");
  const [stats, setStats] = useState({ users: "–", active: "–", logins: "–", analyze: "–" });
  const [range, setRange] = useState(() => isoDaysAgo(30));
  const [preset, setPreset] = useState("30");
  const [status, setStatus] = useState("");
  const [tabLoading, setTabLoading] = useState(true);
  const [users, setUsers] = useState<UserRow[]>([]);
  const [inviteOpen, setInviteOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteUser, setInviteUser] = useState("");
  const [inviteStatus, setInviteStatus] = useState("");
  const [logs, setLogs] = useState<Record<string, unknown>[]>([]);
  const [active, setActive] = useState<Record<string, unknown>[]>([]);
  const [trend, setTrend] = useState<Record<string, unknown>[]>([]);
  const [latency, setLatency] = useState<Record<string, unknown>[]>([]);
  const [models, setModels] = useState<Record<string, unknown>[]>([]);
  const [queries, setQueries] = useState<Record<string, unknown>[]>([]);
  const [cache, setCache] = useState<Record<string, unknown>[]>([]);
  const [cacheSort, setCacheSort] = useState("hits");
  const [cacheTotal, setCacheTotal] = useState(0);
  const [cd, setCd] = useState<string | null>(null);
  const [cdRow, setCdRow] = useState<Record<string, unknown> | null>(null);
  const [cdErr, setCdErr] = useState("");
  const [cdDeleteStatus, setCdDeleteStatus] = useState("");
  const [gens, setGens] = useState<Record<string, unknown>[]>([]);
  const [gensSort, setGensSort] = useState("recent");
  const [gensStatus, setGensStatus] = useState("");
  const [gensTotal, setGensTotal] = useState(0);
  const [vocab, setVocab] = useState<GenerationVocabulary | null>(null);
  const [runs, setRuns] = useState<Record<string, unknown>[]>([]);
  const [runsKind, setRunsKind] = useState("");
  const [runsTotal, setRunsTotal] = useState(0);
  const [gdRecheck, setGdRecheck] = useState(false);
  const [gd, setGd] = useState<string | null>(null);
  const [gdRow, setGdRow] = useState<Record<string, unknown> | null>(null);
  const [gdErr, setGdErr] = useState("");
  const [gdReviews, setGdReviews] = useState<GenReview[]>([]);
  const [gdActionStatus, setGdActionStatus] = useState("");
  const [gdActionKind, setGdActionKind] = useState<"ok" | "err" | "busy" | "">("");
  // The review this admin has running on the open record, followed live.
  const [gdJob, setGdJob] = useState<{ startedAt: number; steps: string[] } | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const pollers = useRef<{ review?: number }>({});
  const [noms, setNoms] = useState<Nom[]>([]);
  const [nomSel, setNomSel] = useState<Set<string>>(new Set());
  const [nomReview, setNomReview] = useState<BenchReviewTarget | null>(null);
  const [nomBusy, setNomBusy] = useState("");
  const closeNomReview = useCallback(() => setNomReview(null), []);
  const [stackStatus, setStackStatus] = useState("");
  const [stackUrl, setStackUrl] = useState("");
  const [ud, setUd] = useState<{ id: string; email: string; avatar?: string | null } | null>(null);
  const [udSummary, setUdSummary] = useState<Record<string, unknown> | null>(null);
  const [udEvents, setUdEvents] = useState<Record<string, unknown>[]>([]);
  const [udErr, setUdErr] = useState("");
  const [udDeleteStatus, setUdDeleteStatus] = useState("");

  const loadStats = useCallback(async () => {
    const [usersC, activeC, logins, analyze] = await Promise.all([
      adminRpcData<number>("get_users_count"),
      adminRpcData<number>("get_active_users_today"),
      adminRpcData<number>("get_total_login_logs_count"),
      adminRpcData<number>("get_total_analyze_logs_count"),
    ]);
    setStats({
      users: usersC == null ? "–" : String(usersC),
      active: activeC == null ? "–" : String(activeC),
      logins: logins == null ? "–" : String(logins),
      analyze: analyze == null ? "–" : String(analyze),
    });
  }, []);

  const loadTab = useCallback(async (name: Tab) => {
    const dateArgs = { p_start_date: range.start, p_end_date: range.end };
    setTabLoading(true);
    try {
      if (name === "users") {
        setUsers((await adminRpcData<UserRow[]>("get_users_with_admin_status", { p_limit: 200, p_offset: 0 })) || []);
      } else if (name === "logs") {
        setLogs((await adminRpcData<Record<string, unknown>[]>("get_login_logs_with_email", { p_limit: 200, p_offset: 0 })) || []);
      } else if (name === "active") {
        setActive((await adminRpcData<Record<string, unknown>[]>("get_active_users_stats", dateArgs)) || []);
      } else if (name === "trend") {
        setTrend((await adminRpcData<Record<string, unknown>[]>("get_daily_call_trend", dateArgs)) || []);
      } else if (name === "latency") {
        setLatency((await adminRpcData<Record<string, unknown>[]>("get_latency_summary", dateArgs)) || []);
      } else if (name === "models") {
        setModels((await adminRpcData<Record<string, unknown>[]>("get_top_models", { ...dateArgs, p_limit: 25 })) || []);
      } else if (name === "queries") {
        setQueries((await adminRpcData<Record<string, unknown>[]>("get_inference_query_dist", dateArgs)) || []);
      } else if (name === "cache") {
        const data = await adminApi<{ items?: Record<string, unknown>[]; total?: number }>("GET", `/api/admin/cached-specs?limit=100&offset=0&sort=${cacheSort}`);
        setCache(data.items || []);
        setCacheTotal(Number(data.total) || 0);
      } else if (name === "gens") {
        const status = gensStatus ? `&status=${encodeURIComponent(gensStatus)}` : "";
        const data = await adminApi<{ items?: Record<string, unknown>[]; total?: number }>("GET", `/api/admin/spec-generations?limit=100&offset=0&sort=${gensSort}${status}`);
        setGens(data.items || []);
        setGensTotal(Number(data.total) || 0);
      } else if (name === "runs") {
        const kind = runsKind ? `&error_kind=${encodeURIComponent(runsKind)}` : "";
        const data = await adminApi<{ items?: Record<string, unknown>[]; total?: number }>("GET", `/api/admin/spec-generation-runs?limit=100&offset=0${kind}`);
        setRuns(data.items || []);
        setRunsTotal(Number(data.total) || 0);
      } else if (name === "benchnoms") {
        const data = await adminApi<{ nominations?: Nom[] }>("GET", "/api/bench/nominations");
        const list = data.nominations || [];
        setNoms(list);
        setNomSel((prev) => new Set(list.map(nomKey).filter((k) => prev.has(k))));
      }
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
    } finally {
      setTabLoading(false);
    }
  }, [range.start, range.end, cacheSort, gensSort, gensStatus, runsKind]);

  const clearGenPolls = () => {
    if (pollers.current.review) clearTimeout(pollers.current.review);
    pollers.current = {};
  };

  const closeGenDetail = () => {
    clearGenPolls();
    setGdJob(null);
    setGd(null);
    setGdRow(null);
    setGdErr("");
    setGdReviews([]);
    setGdActionStatus("");
    setGdActionKind("");
  };

  const loadAdminReviews = async (recordId: string) => {
    try {
      const data = await adminApi<{ items?: GenReview[] }>("GET", `/api/admin/spec-generations/${encodeURIComponent(recordId)}/reviews`);
      setGdReviews(data.items || []);
    } catch { /* best-effort */ }
  };

  // Follow this admin's review of `recordId`, streaming the agent's log lines
  // (cursor-paged) into the progress card until the run is done.
  const pollAdminReview = (recordId: string) => {
    let cursor = 0;
    setGdJob((j) => j ?? { startedAt: Date.now(), steps: [] });
    const tick = async () => {
      try {
        const s = await adminApi<{
          state?: string; events?: { type?: string; text?: string }[]; next_cursor?: number;
          success?: boolean | null; message?: string | null; started_at?: number | null;
        }>("GET", `/api/spec/review-status?record_id=${encodeURIComponent(recordId)}&cursor=${cursor}`);
        if (s.state === "idle") {  // server lost the job (restart) — stop spinning
          setGdJob(null);
          await loadAdminReviews(recordId);
          return;
        }
        cursor = s.next_cursor ?? cursor;
        const lines = (s.events || []).filter((e) => e.type === "log" && e.text).map((e) => e.text as string);
        setGdJob((j) => (j ? {
          startedAt: s.started_at ? s.started_at * 1000 : j.startedAt,
          steps: lines.length ? [...j.steps, ...lines].slice(-60) : j.steps,
        } : j));
        if (s.state === "done") {
          setGdJob(null);
          await loadAdminReviews(recordId);
          if (s.success === false) {
            setGdActionStatus(s.message || "The review did not complete — try again.");
            setGdActionKind("err");
          }
          return;
        }
      } catch { /* transient */ }
      pollers.current.review = window.setTimeout(() => void tick(), 2000);
    };
    if (pollers.current.review) clearTimeout(pollers.current.review);
    pollers.current.review = window.setTimeout(() => void tick(), 0);  // first look right away
  };

  // Re-check a pending record and follow it to its verdict.
  const runAdminRecheck = async (recordId: string) => {
    setGdRecheck(true);
    setGdActionStatus("");
    setGdActionKind("");
    try {
      const r = await adminApi<{ status?: string; message?: string }>("POST", `/api/admin/spec-generations/${encodeURIComponent(recordId)}/recheck`);
      if (r.status !== "started" && r.status !== "already_running") {
        setGdActionStatus(r.message || "Could not start the re-check.");
        setGdActionKind("err");
        return;
      }
      let verdict: { text: string; ok: boolean } = { text: "The re-check is still running; reopen the record later.", ok: false };
      for (let i = 0; i < 600; i++) {
        const s = await adminApi<{ state?: string; success?: boolean | null; message?: string | null }>(
          "GET", `/api/spec/recheck-status?record_id=${encodeURIComponent(recordId)}&cursor=0`);
        if (s.state === "done" || s.state === "idle" || s.state === "interrupted") {
          verdict = {
            text: s.message || (s.state === "done" ? "Re-check finished." : "The re-check was lost; try again."),
            ok: !!s.success,
          };
          break;
        }
        await new Promise((res) => setTimeout(res, 2000));
      }
      await openGenDetail(recordId);
      setGdActionStatus(verdict.text);
      setGdActionKind(verdict.ok ? "ok" : "err");
      void loadTab("gens");
    } catch (e) {
      setGdActionStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
      setGdActionKind("err");
    } finally {
      setGdRecheck(false);
    }
  };

  // 1s clock for the elapsed-time readout, only while a review is running.
  useEffect(() => {
    if (!gdJob) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [gdJob]);

  const openGenDetail = async (recordId: string) => {
    clearGenPolls();
    setGdJob(null);
    setGd(recordId);
    setGdRow(null);
    setGdErr("");
    setGdReviews([]);
    setGdActionStatus("");
    setGdActionKind("");
    try {
      const row = await adminApi<Record<string, unknown>>("GET", `/api/admin/spec-generations/${encodeURIComponent(recordId)}`);
      setGdRow(row);
      void loadAdminReviews(recordId);
      // Re-attach to a review already running on this record.
      void adminApi<{ state?: string }>("GET", `/api/spec/review-status?record_id=${encodeURIComponent(recordId)}&cursor=0`)
        .then((s) => { if (s.state === "running") pollAdminReview(recordId); })
        .catch((e) => {
          setGdActionStatus(`Could not check for a running review: ${e instanceof Error ? e.message : String(e)}`);
          setGdActionKind("err");
        });
    } catch (e) {
      setGdErr(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    document.title = "Admin · HuggingArch";
    return () => {
      document.title = "HuggingArch";
      if (pollers.current.review) clearTimeout(pollers.current.review);
    };
  }, []);

  useEffect(() => {
    void loadAuth().then(async (a) => {
      if (!a.authEnabled) {
        nav("/", { replace: true });
        return;
      }
      if (!a.user) {
        loginRedirect(false);
        return;
      }
      if (!a.isAdmin) {
        setDenied(true);
        setAuth(a);
        return;
      }
      setAuth(a);
      await loadStats();
      await loadTab("users");
    });
    // gate once; tab loads below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nav]);

  useEffect(() => {
    if (auth?.isAdmin) void loadTab(tab);
  }, [auth?.isAdmin, tab, loadTab]);

  // The filters' options are the backend's own vocabularies.
  useEffect(() => {
    if (!auth?.isAdmin) return;
    void adminApi<GenerationVocabulary>("GET", "/api/spec/generation-vocabulary")
      .then(setVocab)
      .catch((e) => setStatus(`Could not load the filter options: ${e instanceof Error ? e.message : String(e)}`));
  }, [auth?.isAdmin]);

  const myId = auth?.user?.id;
  const selfSuper = users.some((r) => r.id === myId && r.is_super);

  const applyDates = () => {
    void loadStats();
    void loadTab(tab);
  };

  if (denied) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-20 text-center">
        <h1 className="text-xl font-semibold text-gray-900">Access denied</h1>
        <p className="mt-2 text-sm text-gray-600">Your account isn&apos;t an admin. If this is wrong, ask an existing admin to grant access.</p>
        <Link to="/" className="inline-block mt-4 px-4 py-2 bg-amber-500 text-white text-sm rounded-lg">Back to app</Link>
      </div>
    );
  }

  if (!auth?.user) {
    return <p className="p-8 text-sm text-gray-500">Loading…</p>;
  }

  return (
    <div className="bg-gray-50 min-h-screen" data-account-page="admin">
      <header className="sticky top-0 z-30 bg-white/90 backdrop-blur border-b border-gray-200">
        <div className="page-shell h-14 flex items-center justify-between">
          <Link to="/" className="inline-flex items-center gap-2 font-semibold text-gray-900 hover:text-amber-600">
            <img src="/static/logo.png" alt="HuggingArch" className="w-7 h-7 rounded-lg" />
            HuggingArch
            <span className={`text-11 px-2 py-0.5 rounded-full font-semibold ${selfSuper ? "bg-indigo-100 text-indigo-700" : "bg-amber-50 text-amber-700"}`}>
              {selfSuper ? "Super admin" : "Admin"}
            </span>
          </Link>
          <button type="button" className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg" onClick={() => void signOut().then(() => { window.location.href = "/"; })}>
            Sign out
          </button>
        </div>
      </header>

      <main className="page-shell py-6 flex flex-col md:flex-row gap-8">
        <aside className="md:w-60 shrink-0 text-sm">
          <AccountNav active="admin" isAdmin={auth.isAdmin} />
        </aside>
        <div className="flex-1 min-w-0">
          <h1 className="text-xl font-semibold text-gray-900">Admin</h1>
          <p className="text-sm text-gray-500 mb-5">Users, traffic, specs, and nominations.</p>
          {status && <p className="text-xs text-red-600 mb-2">{status}</p>}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
            {[["Users", stats.users], ["Active today", stats.active], ["Login records", stats.logins], ["Analyze events", stats.analyze]].map(([k, v]) => (
              <div key={k} className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-10 uppercase text-gray-400">{k}</div>
                <div className="text-lg font-semibold font-mono">{v}</div>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-3 mb-4 text-xs text-gray-600 bg-white border rounded-lg px-4 py-2.5">
            <span className="font-medium text-gray-500">Window</span>
            <select className="border rounded px-2 py-1" value={preset} onChange={(e) => {
              const v = e.target.value;
              setPreset(v);
              if (v !== "custom") setRange(isoDaysAgo(Number(v)));
            }}>
              <option value="7">Last 7 days</option>
              <option value="30">Last 30 days</option>
              <option value="90">Last 90 days</option>
              <option value="365">Last 365 days</option>
              <option value="custom">Custom…</option>
            </select>
            {preset === "custom" && (
              <span className="flex items-center gap-2">
                <input type="date" className="border rounded px-2 py-1" value={range.start} onChange={(e) => setRange((r) => ({ ...r, start: e.target.value }))} />
                <span>→</span>
                <input type="date" className="border rounded px-2 py-1" value={range.end} onChange={(e) => setRange((r) => ({ ...r, end: e.target.value }))} />
              </span>
            )}
            <button type="button" className="px-3 py-1 rounded bg-amber-500 text-white font-medium" onClick={applyDates}>Apply</button>
            <span className="text-gray-400 ml-auto">{range.start} → {range.end}</span>
          </div>

          <div className="flex flex-wrap items-start gap-x-8 gap-y-4 mb-5">
            {TABS.map((g) => (
              <div key={g.group}>
                <div className="text-10 font-semibold uppercase tracking-wider text-gray-400 mb-1.5">{g.group}</div>
                <div className="inline-flex flex-wrap gap-0.5 bg-gray-100 rounded-lg p-1">
                  {g.items.map((t) => (
                    <button key={t.id} type="button" className={`px-2.5 py-1 text-xs rounded-md ${tab === t.id ? "bg-white shadow text-amber-700 font-semibold" : "text-gray-600"}`} onClick={() => { setStatus(""); setTab(t.id); }}>
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {tab === "users" && (
            <section>
              <div className="flex items-center justify-between mb-3">
                <div className="text-xs text-gray-500">Click an email to drill down. Last column promotes / revokes admin.</div>
                <button type="button" className="px-3 py-1.5 text-xs font-medium rounded bg-amber-500 text-white" onClick={() => setInviteOpen((o) => !o)}>Invite user</button>
              </div>
              {inviteOpen && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-3">
                  <div className="text-xs font-semibold text-amber-800 mb-2">Send invite</div>
                  <div className="flex flex-wrap gap-2 items-center text-sm">
                    <input className="border rounded px-3 py-1.5 flex-1 min-w-[260px]" placeholder="email@example.com" value={inviteEmail} onChange={(e) => setInviteEmail(e.target.value)} />
                    <input className="border rounded px-3 py-1.5 w-44" placeholder="Username (optional)" value={inviteUser} onChange={(e) => setInviteUser(e.target.value)} />
                    <button type="button" className="px-3 py-1.5 text-xs rounded bg-amber-500 text-white" onClick={() => {
                      void (async () => {
                        if (!inviteEmail.includes("@")) { setInviteStatus("Enter a valid email address."); return; }
                        setInviteStatus("Sending invite…");
                        try {
                          await adminApi("POST", "/api/admin/users/invite", { email: inviteEmail.trim(), username: inviteUser.trim() || null });
                          setInviteStatus(`Invited ${inviteEmail}. They'll receive an email link to set a password.`);
                          await loadTab("users");
                          await loadStats();
                        } catch (e) {
                          setInviteStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
                        }
                      })();
                    }}>Send</button>
                    <button type="button" className="px-3 py-1.5 text-xs border rounded" onClick={() => setInviteOpen(false)}>Cancel</button>
                  </div>
                  {inviteStatus && <div className="mt-2 text-xs">{inviteStatus}</div>}
                </div>
              )}
              <DataTable
                columns={[
                  {
                    key: "email",
                    header: "Email",
                    className: "font-mono text-xs",
                    render: (r) => (
                      <button type="button" className="text-amber-700 hover:underline inline-flex items-center gap-2" onClick={() => {
                        const uid = r.id;
                        setUd({ id: uid, email: r.user_email || "", avatar: r.avatar_url });
                        setUdSummary(null);
                        setUdEvents([]);
                        setUdErr("");
                        setUdDeleteStatus("");
                        void (async () => {
                          try {
                            const [sum, ev] = await Promise.all([
                              adminApi<unknown>("GET", `/api/admin/users/${uid}/summary`),
                              adminApi<unknown>("GET", `/api/admin/users/${uid}/events?limit=50`),
                            ]);
                            setUdSummary(unwrapSummary(sum));
                            setUdEvents(Array.isArray(ev) ? ev as Record<string, unknown>[] : []);
                          } catch (e) {
                            setUdErr(e instanceof Error ? e.message : String(e));
                          }
                        })();
                      }}>
                        <AvatarFace url={r.avatar_url} text={r.username || r.user_email || "?"} size="w-5 h-5 text-10" />
                        {r.user_email}
                      </button>
                    ),
                    sortValue: (r) => r.user_email || null,
                  },
                  { key: "username", header: "Username", render: (r) => r.username || "—", sortValue: (r) => r.username || null },
                  {
                    key: "joined",
                    header: "Joined",
                    className: "text-gray-500",
                    render: (r) => r.created_at ? new Date(r.created_at).toLocaleString() : "—",
                    sortValue: (r) => sortTime(r.created_at),
                  },
                  {
                    key: "admin",
                    header: "Admin",
                    render: (r) => (
                      r.is_super ? <span className="text-11 px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700 font-semibold">super admin</span>
                        : r.is_admin ? <span className="text-amber-700 font-medium">admin</span>
                        : <span className="text-gray-400">user</span>
                    ),
                    sortValue: (r) => r.is_super ? 2 : r.is_admin ? 1 : 0,
                  },
                  {
                    key: "actions",
                    header: "Actions",
                    align: "right",
                    className: "whitespace-nowrap",
                    render: (r) => (
                      r.is_super && !selfSuper ? (
                        <span className="text-11 text-gray-400">super — protected</span>
                      ) : (
                        <>
                          {selfSuper && r.is_admin && (
                            <button type="button" className="text-xs px-2.5 py-1 rounded border mr-1" onClick={() => void adminRpc("set_super_admin", { p_user_id: r.id, p_is_super: !r.is_super }).then(() => loadTab("users")).catch((e) => setStatus(e instanceof Error ? e.message : String(e)))}>
                              {r.is_super ? "Revoke super" : "★ Make super"}
                            </button>
                          )}
                          <button type="button" className={`text-xs px-2.5 py-1 rounded border ${r.is_admin ? "border-red-200 text-red-700" : "border-amber-300 text-amber-700"}`} onClick={() => void adminRpc(r.is_admin ? "remove_admin" : "add_admin", { p_user_id: r.id }).then(() => loadTab("users")).catch((e) => setStatus(e instanceof Error ? e.message : String(e)))}>
                            {r.is_admin ? "Revoke admin" : "+ Grant admin"}
                          </button>
                        </>
                      )
                    ),
                    sortValue: null,
                  },
                ]}
                rows={users}
                rowKey={(r) => r.id}
                loading={tabLoading}
                empty={tabLoading ? "Loading…" : ""}
              />
            </section>
          )}

          {tab === "logs" && (
            <DataTable
              columns={[
                { key: "time", header: "Time", render: (r) => r.login_at ? new Date(String(r.login_at)).toLocaleString() : "—", sortValue: (r) => sortTime(r.login_at) },
                { key: "email", header: "Email", className: "font-mono text-xs", render: (r) => String(r.user_email || ""), sortValue: (r) => r.user_email ? String(r.user_email) : null },
                { key: "ip", header: "IP", className: "font-mono text-xs text-gray-500", render: (r) => String(r.ip_address || "—"), sortValue: (r) => r.ip_address ? String(r.ip_address) : null },
                { key: "ua", header: "User-Agent", className: "text-xs text-gray-500 truncate max-w-md", render: (r) => String(r.user_agent || ""), sortValue: (r) => r.user_agent ? String(r.user_agent) : null },
              ]}
              rows={logs}
              rowKey={(_r, i) => i}
              loading={tabLoading}
              empty={tabLoading ? "Loading…" : ""}
            />
          )}

          {tab === "active" && (
            <section className="bg-white border rounded-lg p-4">
              {tabLoading ? <div className="text-sm text-gray-400 py-8 text-center">Loading…</div> : <ActiveLines rows={active} />}
              <DataTable
                className="mt-6"
                framed={false}
                columns={[
                  { key: "date", header: "Date", render: (r) => String(r.date), sortValue: (r) => sortTime(r.date) },
                  { key: "new", header: "New", align: "right", className: "text-amber-700", render: (r) => String(r.new_users), sortValue: (r) => sortNum(r.new_users) },
                  { key: "active", header: "Active", align: "right", render: (r) => String(r.active_users), sortValue: (r) => sortNum(r.active_users) },
                  { key: "logins", header: "Logins", align: "right", className: "text-gray-500", render: (r) => String(r.login_count), sortValue: (r) => sortNum(r.login_count) },
                  { key: "avg", header: "Avg/user", align: "right", className: "text-gray-500", render: (r) => String(r.avg_login_per_user), sortValue: (r) => sortNum(r.avg_login_per_user) },
                ]}
                rows={active}
                rowKey={(_r, i) => i}
                loading={tabLoading}
                empty={tabLoading ? "Loading…" : ""}
              />
            </section>
          )}

          {tab === "trend" && (
            <section className="bg-white border rounded-lg p-4">
              {tabLoading ? <div className="text-sm text-gray-400 py-8 text-center">Loading…</div> : <TrendBars rows={trend} />}
              <div className="text-xs text-gray-400 mt-2">Stacked daily calls split by endpoint. <code>analyze</code> = Inspector tab. <code>inference_estimate</code> = Inference planner.</div>
              <DataTable
                className="mt-6"
                framed={false}
                columns={[
                  { key: "day", header: "Day", render: (r) => String(r.day), sortValue: (r) => sortTime(r.day) },
                  { key: "total", header: "Total", align: "right", className: "font-medium", render: (r) => String(r.total_calls), sortValue: (r) => sortNum(r.total_calls) },
                  { key: "analyze", header: "Analyze", align: "right", className: "text-blue-700", render: (r) => String(r.analyze_calls), sortValue: (r) => sortNum(r.analyze_calls) },
                  { key: "inference", header: "Inference", align: "right", className: "text-purple-700", render: (r) => String(r.inference_calls), sortValue: (r) => sortNum(r.inference_calls) },
                  { key: "users", header: "Users", align: "right", className: "text-gray-500", render: (r) => String(r.unique_users), sortValue: (r) => sortNum(r.unique_users) },
                  { key: "anon", header: "Anon", align: "right", className: "text-gray-400", render: (r) => String(r.anon_calls), sortValue: (r) => sortNum(r.anon_calls) },
                ]}
                rows={[...trend].reverse()}
                rowKey={(_r, i) => i}
                loading={tabLoading}
                empty={tabLoading ? "Loading…" : ""}
              />
            </section>
          )}

          {tab === "latency" && (
            <DataTable
              columns={[
                { key: "endpoint", header: "Endpoint", render: (r) => String(r.endpoint), sortValue: (r) => r.endpoint == null ? null : String(r.endpoint) },
                { key: "calls", header: "Calls", align: "right", render: (r) => String(r.call_count), sortValue: (r) => sortNum(r.call_count) },
                { key: "avg", header: "avg", align: "right", render: (r) => String(r.avg_ms ?? "–"), sortValue: (r) => sortNum(r.avg_ms) },
                { key: "p50", header: "p50", align: "right", render: (r) => r.p50_ms != null ? Number(r.p50_ms).toFixed(0) : "–", sortValue: (r) => sortNum(r.p50_ms) },
                { key: "p95", header: "p95", align: "right", className: "text-amber-700", render: (r) => r.p95_ms != null ? Number(r.p95_ms).toFixed(0) : "–", sortValue: (r) => sortNum(r.p95_ms) },
                { key: "max", header: "max", align: "right", className: "text-red-600", render: (r) => String(r.max_ms ?? "–"), sortValue: (r) => sortNum(r.max_ms) },
              ]}
              rows={latency}
              rowKey={(_r, i) => i}
              loading={tabLoading}
              empty={tabLoading ? "Loading…" : ""}
            />
          )}

          {tab === "models" && (
            <DataTable
              columns={[
                { key: "model", header: "Model", className: "font-mono text-xs", render: (r) => String(r.model_id), sortValue: (r) => r.model_id == null ? null : String(r.model_id) },
                { key: "total", header: "Total", align: "right", className: "font-medium", render: (r) => String(r.total_calls), sortValue: (r) => sortNum(r.total_calls) },
                { key: "analyze", header: "Analyze", align: "right", className: "text-gray-500", render: (r) => String(r.analyze_calls), sortValue: (r) => sortNum(r.analyze_calls) },
                { key: "inference", header: "Inference", align: "right", className: "text-gray-500", render: (r) => String(r.inference_calls), sortValue: (r) => sortNum(r.inference_calls) },
                { key: "users", header: "Users", align: "right", className: "text-gray-500", render: (r) => String(r.unique_users), sortValue: (r) => sortNum(r.unique_users) },
              ]}
              rows={models}
              rowKey={(_r, i) => i}
              loading={tabLoading}
              empty={tabLoading ? "Loading…" : "No analytics yet."}
            />
          )}

          {tab === "queries" && (
            <DataTable
              columns={[
                { key: "gpu", header: "GPU SKU", className: "font-mono text-xs", render: (r) => String(r.gpu_sku), sortValue: (r) => r.gpu_sku == null ? null : String(r.gpu_sku) },
                { key: "queries", header: "Queries", align: "right", className: "font-medium", render: (r) => String(r.query_count), sortValue: (r) => sortNum(r.query_count) },
                { key: "cluster", header: "Avg cluster size", align: "right", className: "text-gray-500", render: (r) => String(r.avg_cluster_size ?? "–"), sortValue: (r) => sortNum(r.avg_cluster_size) },
                { key: "users", header: "Users", align: "right", className: "text-gray-500", render: (r) => String(r.unique_users), sortValue: (r) => sortNum(r.unique_users) },
              ]}
              rows={queries}
              rowKey={(_r, i) => i}
              loading={tabLoading}
              empty={tabLoading ? "Loading…" : "No inference queries yet."}
            />
          )}

          {tab === "cache" && (
            <section>
              <div className="flex items-center justify-between mb-3 text-sm text-gray-500">
                <div>Global <strong>shared</strong> spec cache — the cross-VM serving tier for anonymous / self-host runs, populated after successful agent runs. Distinct from the per-user <strong>Generations</strong>. Click a row for the YAML preview; delete only drops the cache row.</div>
                <select className="border rounded px-2 py-1 text-xs" value={cacheSort} onChange={(e) => setCacheSort(e.target.value)}>
                  <option value="hits">Most popular</option>
                  <option value="recent">Recently used</option>
                  <option value="oldest">Oldest first</option>
                </select>
              </div>
              <DataTable
                columns={[
                  {
                    key: "model",
                    header: "Model",
                    className: "font-mono text-xs",
                    render: (r) => {
                      const mid = String(r.model_id);
                      return (
                        <button type="button" className="text-amber-700 hover:underline" onClick={() => {
                          setCd(mid);
                          setCdRow(null);
                          setCdErr("");
                          setCdDeleteStatus("");
                          void (async () => {
                            try {
                              setCdRow(await adminApi<Record<string, unknown>>("GET", `/api/admin/cached-specs/${encodeURIComponent(mid)}`));
                            } catch (e) {
                              setCdErr(e instanceof Error ? e.message : String(e));
                            }
                          })();
                        }}>{mid}</button>
                      );
                    },
                    sortValue: (r) => String(r.model_id),
                  },
                  { key: "agent", header: "Agent", className: "text-xs", render: (r) => String(r.agent_cli || "—"), sortValue: (r) => r.agent_cli ? String(r.agent_cli) : null },
                  { key: "hits", header: "Hits", align: "right", className: "font-medium", render: (r) => String(r.hit_count ?? r.hits ?? "–"), sortValue: (r) => sortNum(r.hit_count ?? r.hits) },
                  { key: "yaml", header: "YAML", align: "right", className: "text-xs text-gray-500", render: (r) => bytesShort(Number(r.spec_yaml_bytes)), sortValue: (r) => sortNum(r.spec_yaml_bytes) },
                  { key: "comp", header: "+Comp", align: "right", className: "text-gray-500", render: (r) => String(r.new_component_count || 0), sortValue: (r) => Number(r.new_component_count) || 0 },
                  { key: "by", header: "Generated by", className: "text-xs text-gray-500", render: (r) => String(r.generated_by_email || "anon"), sortValue: (r) => String(r.generated_by_email || "anon") },
                  { key: "used", header: "Last used", className: "text-xs text-gray-500", render: (r) => r.last_used_at ? new Date(String(r.last_used_at)).toLocaleString() : "—", sortValue: (r) => sortTime(r.last_used_at) },
                  {
                    key: "actions",
                    header: "Actions",
                    align: "right",
                    render: (r) => {
                      const mid = String(r.model_id);
                      return (
                        <button type="button" className="text-xs px-2 py-1 border border-red-200 text-red-700 rounded" onClick={async () => {
                          if (!(await confirmDialog({ title: `Delete the cache entry for ${mid}?`,
                            confirmLabel: "Delete", danger: true }))) return;
                          void adminApi("DELETE", `/api/admin/cached-specs/${encodeURIComponent(mid)}`).then(() => loadTab("cache")).catch(alertError);
                        }}>Delete</button>
                      );
                    },
                    sortValue: null,
                  },
                ]}
                rows={cache}
                rowKey={(_r, i) => i}
                loading={tabLoading}
                empty={tabLoading ? "Loading…" : "No shared specs yet."}
                rowClassName={() => "hover:bg-amber-50/40"}
              />
              <div className="text-xs text-gray-500 mt-2">{!tabLoading && (cache.length === 0 ? "No shared specs yet." : `Showing ${cache.length} of ${cacheTotal} shared specs.`)}</div>
            </section>
          )}

          {tab === "gens" && (
            <section>
              <div className="flex items-center justify-between mb-3 text-sm text-gray-500">
                <div>One record per user per generation run. Open a row for YAML and the review report (Generate review). Inspector is a separate tab.</div>
                <div className="flex items-center gap-2">
                  <select
                    className="border rounded px-2 py-1 text-xs"
                    value={gensStatus}
                    onChange={(e) => setGensStatus(e.target.value)}
                    data-filter="status"
                    title="Only records in this status"
                  >
                    <option value="">All statuses</option>
                    {(vocab?.generation_status || []).map((v) => (
                      <option key={v} value={v}>{generationBadge(v).label}</option>
                    ))}
                  </select>
                  <select className="border rounded px-2 py-1 text-xs" value={gensSort} onChange={(e) => setGensSort(e.target.value)}>
                    <option value="recent">Newest first</option>
                    <option value="oldest">Oldest first</option>
                    <option value="model">By model</option>
                  </select>
                </div>
              </div>
              <DataTable
                columns={[
                  { key: "user", header: "User", className: "text-xs text-gray-500", render: (r) => String(r.generated_by_email || "anon"), sortValue: (r) => String(r.generated_by_email || "anon") },
                  {
                    key: "model",
                    header: "Model",
                    className: "font-mono text-xs",
                    render: (r) => (
                      <button type="button" className="text-amber-700 hover:underline" onClick={() => void openGenDetail(String(r.id || ""))}>{String(r.model_id)}</button>
                    ),
                    sortValue: (r) => String(r.model_id),
                  },
                  { key: "agent", header: "Agent", className: "text-xs", render: (r) => String(r.agent_cli || "—"), sortValue: (r) => r.agent_cli ? String(r.agent_cli) : null },
                  {
                    key: "status",
                    header: "Status",
                    className: "whitespace-nowrap",
                    render: (r) => <><GenStatusBadge row={r} /><LibraryBadge row={r} /></>,
                    sortValue: (r) => generationBadge(r.status).label,
                  },
                  { key: "yaml", header: "YAML", align: "right", className: "text-gray-500 text-xs", render: (r) => bytesShort(r.spec_yaml_bytes == null ? null : Number(r.spec_yaml_bytes)), sortValue: (r) => sortNum(r.spec_yaml_bytes) },
                  { key: "comp", header: "+Comp", align: "right", className: "text-gray-500", render: (r) => Number(r.new_component_count) || 0, sortValue: (r) => Number(r.new_component_count) || 0 },
                  { key: "created", header: "Created", className: "text-xs text-gray-500", render: (r) => r.created_at ? new Date(String(r.created_at)).toLocaleString() : "", sortValue: (r) => sortTime(r.created_at) },
                  {
                    key: "report",
                    header: "Report",
                    align: "right",
                    render: (r) => (
                      <button type="button" className="text-xs px-2 py-1 border border-gray-300 text-gray-700 hover:bg-gray-50 rounded" onClick={() => void openGenDetail(String(r.id || ""))}>Report</button>
                    ),
                    sortValue: null,
                  },
                  {
                    key: "inspector",
                    header: "Inspector",
                    align: "right",
                    render: (r) => (
                      <a href={`/inspector.html#model=${encodeURIComponent(String(r.model_id))}`} target="_blank" rel="noopener" className="text-xs px-2 py-1 border border-amber-200 text-amber-700 hover:bg-amber-50 rounded" onClick={(e) => e.stopPropagation()}>Open ↗</a>
                    ),
                    sortValue: null,
                  },
                ]}
                rows={gens}
                rowKey={(r) => String(r.id || "")}
                loading={tabLoading}
                empty={tabLoading ? "Loading…" : "No spec generations yet."}
                onRowClick={(r) => void openGenDetail(String(r.id || ""))}
              />
              <div className="text-xs text-gray-500 mt-2">{!tabLoading && (gens.length === 0 ? "No spec generations yet." : `Showing ${gens.length} of ${gensTotal} generation records.`)}</div>
            </section>
          )}

          {tab === "runs" && (
            <section>
              <div className="flex items-center justify-between mb-3 text-sm text-gray-500">
                <div>Every spec run, from its log row — including runs that left no record: anonymous runs, records that failed to save, and runs left with only unattributed refusals.</div>
                <select
                  className="border rounded px-2 py-1 text-xs"
                  value={runsKind}
                  onChange={(e) => setRunsKind(e.target.value)}
                  data-filter="error_kind"
                  title="Only runs that ended with this error kind"
                >
                  <option value="">All runs</option>
                  {(vocab?.run_error_kind || []).map((v) => (
                    <option key={v} value={v}>{RUN_ERROR_KIND_BADGE[v]?.label || v}</option>
                  ))}
                </select>
              </div>
              <DataTable
                columns={[
                  { key: "user", header: "User", className: "text-xs text-gray-500", render: (r) => String(r.user_email || "anon"), sortValue: (r) => String(r.user_email || "anon") },
                  { key: "model", header: "Model", className: "font-mono text-xs", render: (r) => String(r.model_id || ""), sortValue: (r) => r.model_id ? String(r.model_id) : null },
                  { key: "kind", header: "Kind", className: "text-xs", render: (r) => String(r.kind || "generate"), sortValue: (r) => String(r.kind || "generate") },
                  {
                    key: "result",
                    header: "Result",
                    className: "text-xs",
                    render: (r) => r.state === "running" ? "running" : r.success === true ? "succeeded" : r.state === "interrupted" ? "interrupted" : "failed",
                    sortValue: (r) => r.state === "running" ? "running" : r.success === true ? "succeeded" : r.state === "interrupted" ? "interrupted" : "failed",
                  },
                  { key: "error", header: "Error kind", className: "whitespace-nowrap", render: (r) => <RunErrorBadge row={r} />, sortValue: (r) => r.error_kind ? RUN_ERROR_KIND_BADGE[String(r.error_kind)]?.label || String(r.error_kind) : null },
                  {
                    key: "message",
                    header: "Message",
                    className: "text-xs text-gray-500 max-w-md truncate",
                    render: (r) => String(r.message || ""),
                    cellProps: (r) => ({ title: String(r.message || "") }),
                    sortValue: (r) => r.message ? String(r.message) : null,
                  },
                  { key: "started", header: "Started", className: "text-xs text-gray-500", render: (r) => r.created_at ? new Date(String(r.created_at)).toLocaleString() : "", sortValue: (r) => sortTime(r.created_at) },
                ]}
                rows={runs}
                rowKey={(r) => String(r.id)}
                loading={tabLoading}
                empty={tabLoading ? "Loading…" : "No runs."}
              />
              <div className="text-xs text-gray-500 mt-2">{runs.length === 0 ? "" : `Showing ${runs.length} of ${runsTotal} runs.`}</div>
            </section>
          )}

          {tab === "benchnoms" && (() => {
            const ids = noms.map(nomKey);
            const allOn = ids.length > 0 && ids.every((id) => nomSel.has(id));
            const picked = noms.filter((n) => nomSel.has(nomKey(n)));
            const pickedRows = picked.reduce((t, n) => t + (n.row_count || 0), 0);
            const setOn = (id: string, on: boolean) => setNomSel((prev) => {
              const next = new Set(prev);
              if (on) next.add(id); else next.delete(id);
              return next;
            });
            // Promote / reject a set of nominations in one request.
            const decide = async (action: "promote" | "reject", sets: Nom[]) => {
              if (!sets.length) return;
              const rows = sets.reduce((t, n) => t + (n.row_count || 0), 0);
              const what = sets.length === 1 ? `“${sets[0].label || "(unnamed)"}” by ${submitterOf(sets[0])}`
                : `${sets.length} sets`;
              const ok = await confirmDialog(action === "promote"
                ? { title: `Promote ${what} to the shared corpus?`,
                    message: `${rows.toLocaleString()} nominated rows become public.`, confirmLabel: "Promote" }
                : { title: `Reject ${what}?`,
                    message: "The batches are marked rejected; they stay private to their owners.", confirmLabel: "Reject", danger: true });
              if (!ok) return;
              setNomBusy(action === "promote" ? "Promoting…" : "Rejecting…");
              try {
                await adminApi("POST", `/api/bench/${action}`, { sets: sets.map(({ owner, kind, id }) => ({ owner, kind, id })) });
              } catch (e) {
                await alertError(e);
              } finally {
                setNomBusy("");
                void loadTab("benchnoms");
              }
            };
            return (
              <section className="bg-white border rounded-lg overflow-x-auto">
                <div className="px-4 py-3 border-b text-sm text-gray-500">Batches nominated for the shared corpus, one upload each. <b>Review</b> opens the batch (its rows, roofline checks, and overlap with shared). Tick the batches to decide, then <b>Promote selected</b> publishes them or <b>Reject selected</b> turns them down (they stay their owners&apos;).</div>
                {noms.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2 px-4 py-2 border-b text-xs">
                    <button type="button" className="px-2 py-1 border border-gray-300 rounded bg-white hover:bg-gray-50 disabled:opacity-40"
                      disabled={allOn} onClick={() => setNomSel(new Set(ids))}>Select all ({noms.length})</button>
                    {picked.length > 0 && (
                      <>
                        <span className="text-gray-600">{picked.length} selected · {pickedRows.toLocaleString()} rows</span>
                        <button type="button" className="text-blue-700 hover:underline" onClick={() => setNomSel(new Set())}>Clear</button>
                        <button type="button" className="px-2 py-1 bg-green-600 text-white rounded disabled:opacity-40" disabled={!!nomBusy}
                          onClick={() => void decide("promote", picked)}>Promote selected</button>
                        <button type="button" className="px-2 py-1 border rounded disabled:opacity-40" disabled={!!nomBusy}
                          onClick={() => void decide("reject", picked)}>Reject selected</button>
                      </>
                    )}
                    {nomBusy && <span className="text-gray-500">{nomBusy}</span>}
                  </div>
                )}
                <DataTable
                  framed={false}
                  columns={[
                    {
                      key: "sel",
                      header: <input type="checkbox" aria-label="Select all" checked={allOn} onChange={(e) => setNomSel(e.target.checked ? new Set(ids) : new Set())} />,
                      headerClassName: "w-8",
                      render: (n) => <input type="checkbox" checked={nomSel.has(nomKey(n))} onChange={(e) => setOn(nomKey(n), e.target.checked)} />,
                      sortValue: null,
                    },
                    { key: "label", wrap: true, header: "Set", render: (n) => <>{n.label || "(unnamed)"}<div className="font-mono text-10 text-gray-400">{n.kind} {n.kind === "run" ? n.id : n.id.slice(0, 8)}{n.run_id ? ` · run ${n.run_id}` : ""}</div></>, sortValue: (n) => n.label || "(unnamed)" },
                    { key: "submitter", wrap: true, header: "Submitted by", className: "text-xs", render: (n) => <span title={n.owner}>{submitterOf(n)}</span>, sortValue: (n) => submitterOf(n) },
                    { key: "models", wrap: true, header: "Models", className: "text-xs", render: (n) => (n.models || []).map((m) => m.split("/").pop()).join(", "), sortValue: (n) => (n.models || []).map((m) => m.split("/").pop()).join(", ") },
                    { key: "frameworks", wrap: true, header: "Frameworks", render: (n) => (n.frameworks || []).join(", "), sortValue: (n) => (n.frameworks || []).join(", ") },
                    { key: "gpus", wrap: true, header: "GPUs", render: (n) => (n.gpus || []).join(", "), sortValue: (n) => (n.gpus || []).join(", ") },
                    ...peakCheckColumns<Nom>(),
                    { key: "rows", header: "Rows", align: "right", render: (n) => (n.row_count || 0).toLocaleString(), sortValue: (n) => n.row_count },
                    { key: "nominated", header: "Nominated", className: "text-xs text-gray-500", render: (n) => n.nominated_at ? new Date(n.nominated_at).toLocaleString() : "–", sortValue: (n) => sortTime(n.nominated_at) },
                    {
                      key: "review",
                      header: "Review",
                      align: "right",
                      className: "whitespace-nowrap",
                      render: (n) => (
                        <>
                          <button type="button" className="text-xs px-2 py-1 border rounded hover:bg-gray-50" onClick={() => setNomReview(benchReviewTarget(n.kind, n.id, n.owner))}>Review</button>
                          <a className="text-xs px-2 py-1 border border-blue-200 text-blue-700 rounded hover:bg-blue-50 ml-1" href={benchCorpusHref(benchReviewTarget(n.kind, n.id, n.owner))} target="_blank" rel="noreferrer">Rows ↗</a>
                        </>
                      ),
                      sortValue: null,
                    },
                  ]}
                  rows={noms}
                  rowKey={nomKey}
                  loading={tabLoading}
                  empty={tabLoading ? "Loading…" : "No pending nominations."}
                />
                {nomReview && <BenchReviewModal target={nomReview} onClose={closeNomReview} />}
              </section>
            );
          })()}

          {tab === "system" && (
            <div className="space-y-4">
              <TaskDefaultsCard />
              <section className="bg-white border rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-1">Model-loading stack</h3>
              <p className="text-sm text-gray-500 mb-3 max-w-xl">The deployment ships whatever <code>uv.lock</code> pins. This button fires the weekly re-lock job on demand.</p>
              <div className="flex items-center gap-3 flex-wrap">
                <button type="button" className="px-3 py-1.5 text-xs rounded bg-amber-500 text-white" onClick={() => {
                  void (async () => {
                    setStackStatus("Dispatching…");
                    setStackUrl("");
                    try {
                      const r = await adminApi<{ workflow_url?: string }>("POST", "/api/admin/update-model-stack");
                      setStackStatus("Workflow dispatched — it re-locks, runs the gates, and opens a PR if green.");
                      setStackUrl(r.workflow_url || "");
                    } catch (e) {
                      setStackStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
                    }
                  })();
                }}>Run stack refresh now</button>
                <span className="text-xs">{stackStatus}</span>
                {stackUrl && <a href={stackUrl} target="_blank" rel="noopener" className="text-xs text-amber-700 hover:underline">View workflow ↗</a>}
              </div>
              </section>
            </div>
          )}
        </div>
      </main>

      {ud && (
        <div className="fixed inset-0 z-40" onClick={() => setUd(null)}>
          <div className="absolute inset-0 bg-black/30" />
          <aside className="absolute right-0 top-0 h-full w-full md:w-[640px] bg-white shadow-xl overflow-y-auto p-6" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <AvatarFace url={ud.avatar} text={ud.email || "?"} size="w-10 h-10 text-base" />
                <div>
                  <div className="text-sm font-mono font-semibold">{ud.email || "—"}</div>
                  <div className="text-11 font-mono text-gray-400 mt-0.5">{ud.id}</div>
                </div>
              </div>
              <button type="button" className="text-2xl text-gray-400" onClick={() => setUd(null)}>×</button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs mb-4">
              {udErr ? (
                <div className="col-span-full text-red-600">{udErr}</div>
              ) : !udSummary ? (
                <div className="col-span-full text-gray-400">Loading…</div>
              ) : (
                <>
                  <UdCard label="Total events" val={String(udSummary.total_calls ?? 0)} hint={`${udSummary.analyze_calls ?? 0} analyze · ${udSummary.inference_calls ?? 0} inference`} />
                  <UdCard label="Unique models" val={String(udSummary.unique_models ?? 0)} />
                  <UdCard label="Logins" val={String(udSummary.login_count ?? 0)} hint={`last: ${fmtTime(udSummary.last_login_at)}`} />
                  <UdCard label="First / Last event" val={fmtTime(udSummary.last_event_at)} hint={`first: ${fmtTime(udSummary.first_event_at)}`} />
                </>
              )}
            </div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Recent events</h3>
            <DataTable
              className="mb-5"
              columns={[
                { key: "when", header: "When", className: "text-gray-600", render: (e) => e.created_at ? new Date(String(e.created_at)).toLocaleString() : "", sortValue: (e) => sortTime(e.created_at) },
                {
                  key: "endpoint",
                  header: "Endpoint",
                  render: (e) => (
                    <span className={`px-1.5 py-0.5 rounded text-10 ${e.endpoint === "analyze" ? "bg-blue-100 text-blue-700" : "bg-purple-100 text-purple-700"}`}>{String(e.endpoint || "")}</span>
                  ),
                  sortValue: (e) => e.endpoint ? String(e.endpoint) : null,
                },
                { key: "model", header: "Model", className: "font-mono text-10", render: (e) => String(e.model_id || ""), sortValue: (e) => e.model_id ? String(e.model_id) : null },
                { key: "duration", header: "Duration", align: "right", className: "text-gray-500", render: (e) => e.duration_ms != null ? `${e.duration_ms} ms` : "–", sortValue: (e) => sortNum(e.duration_ms) },
              ]}
              rows={udEvents}
              rowKey={(_e, i) => i}
              empty={udSummary || udErr ? "No events." : "Loading…"}
            />
            <div className="border border-red-200 bg-red-50 rounded-lg p-4">
              <div className="text-sm font-semibold text-red-700 mb-1">Danger zone</div>
              <p className="text-xs text-red-700/80 mb-3">Permanently delete this user from public tables (analyze_logs, login_logs, profile, user_models) and from <code>auth.users</code>. Cannot be undone. Last admin and self-deletion are blocked server-side.</p>
              <button type="button" className="text-xs px-3 py-1.5 bg-red-600 text-white rounded" onClick={async () => {
                if (!(await confirmDialog({ title: "Permanently delete this user?", message: "This cannot be undone.",
                  confirmLabel: "Delete user", danger: true }))) return;
                setUdDeleteStatus("Deleting…");
                void adminApi("POST", `/api/admin/users/${ud.id}/delete`).then(() => {
                  setUdDeleteStatus("Deleted.");
                  setTimeout(() => { setUd(null); void loadTab("users"); void loadStats(); }, 800);
                }).catch((e) => {
                  setUdDeleteStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
                });
              }}>Delete user</button>
              {udDeleteStatus && <span className={`ml-3 text-xs ${udDeleteStatus.startsWith("Failed") ? "text-red-600" : udDeleteStatus === "Deleted." ? "text-green-600" : "text-gray-500"}`}>{udDeleteStatus}</span>}
            </div>
          </aside>
        </div>
      )}

      {cd && (
        <div className="fixed inset-0 z-40" onClick={() => setCd(null)}>
          <div className="absolute inset-0 bg-black/30" />
          <aside className="absolute right-0 top-0 h-full w-full md:w-[760px] bg-white shadow-xl overflow-y-auto p-6" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="text-sm font-mono font-semibold text-gray-900">{cd}</div>
                <div className="text-11 text-gray-500 mt-0.5">
                  {cdErr ? `Failed: ${cdErr}` : !cdRow ? "Loading…" : (
                    `${String(cdRow.agent_cli || "unknown agent")} · ${String(cdRow.hit_count ?? 0)} hits · generated by ${String(cdRow.generated_by_email || "anon")} · created ${cdRow.created_at ? new Date(String(cdRow.created_at)).toLocaleString() : "—"}`
                  )}
                </div>
              </div>
              <button type="button" className="text-2xl text-gray-400" onClick={() => setCd(null)}>×</button>
            </div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2 mt-2">Files</h3>
            <div className="mb-5">
              {cdRow ? (
                <SpecFiles modelId={cd} specYaml={String(cdRow.spec_yaml || "")}
                  components={cdRow.new_component_yamls as SpecFile[] | undefined} />
              ) : <div className="text-xs text-gray-400">…</div>}
            </div>
            <details className="mb-5">
              <summary className="text-sm font-semibold text-gray-700 cursor-pointer">Snapshot metadata</summary>
              <pre className="text-11 font-mono bg-gray-50 border border-gray-200 rounded p-3 overflow-x-auto whitespace-pre-wrap mt-2">{cdRow ? JSON.stringify(cdRow.snapshot_json || {}, null, 2) : "…"}</pre>
            </details>
            <div className="border border-red-200 bg-red-50 rounded-lg p-4 mt-4">
              <div className="text-sm font-semibold text-red-700 mb-1">Danger zone</div>
              <p className="text-xs text-red-700/80 mb-3">Deletes only the cache row. Next request for this model either falls through to a built-in spec or triggers a fresh agent run.</p>
              <button type="button" className="text-xs px-3 py-1.5 bg-red-600 text-white rounded" onClick={async () => {
                if (!(await confirmDialog({ title: `Delete the cache entry for ${cd}?`, message: "This cannot be undone.",
                  confirmLabel: "Delete", danger: true }))) return;
                setCdDeleteStatus("Deleting…");
                void adminApi("DELETE", `/api/admin/cached-specs/${encodeURIComponent(cd)}`).then(() => {
                  setCdDeleteStatus("Deleted.");
                  setTimeout(() => { setCd(null); void loadTab("cache"); }, 600);
                }).catch((e) => {
                  setCdDeleteStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
                });
              }}>Delete cache entry</button>
              {cdDeleteStatus && <span className={`ml-3 text-xs ${cdDeleteStatus.startsWith("Failed") ? "text-red-600" : cdDeleteStatus === "Deleted." ? "text-green-600" : "text-gray-500"}`}>{cdDeleteStatus}</span>}
            </div>
          </aside>
        </div>
      )}

      {gd && (
        <div className="fixed inset-0 z-40" onClick={closeGenDetail}>
          <div className="absolute inset-0 bg-black/30" />
          <aside className="absolute right-0 top-0 h-full w-full md:w-[760px] bg-white shadow-xl overflow-y-auto p-6" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="text-sm font-mono font-semibold text-gray-900">{gdRow ? String(gdRow.model_id || "—") : "…"}</div>
                <div className="text-11 text-gray-500 mt-0.5">
                  {gdErr ? `Failed: ${gdErr}` : !gdRow ? "Loading…" : (
                    <span className="inline-flex items-center gap-1 flex-wrap">
                      <GenStatusBadge row={gdRow} prUrl={gdRow.pr_url ?? null} />
                      <LibraryBadge row={gdRow} />
                      <span>· {String(gdRow.agent_cli || "unknown agent")} · by {String(gdRow.generated_by_email || "anon")} · created {gdRow.created_at ? new Date(String(gdRow.created_at)).toLocaleString() : "—"}</span>
                    </span>
                  )}
                </div>
              </div>
              <button type="button" className="text-2xl text-gray-400" onClick={closeGenDetail}>×</button>
            </div>
            {gdRow && (
              <div className="mb-5">
                <div className="flex items-center gap-2 flex-wrap">
                  <a
                    href={`/inspector.html#model=${encodeURIComponent(String(gdRow.model_id || ""))}`}
                    target="_blank"
                    rel="noopener"
                    className="text-xs px-3 py-1.5 font-medium bg-amber-500 text-white rounded hover:bg-amber-600"
                  >
                    Open in Inspector ↗
                  </a>
                  {gdRow.pr_url ? (
                    <a href={String(gdRow.pr_url)} target="_blank" rel="noopener" className="text-xs px-3 py-1.5 border border-amber-300 text-amber-700 hover:bg-amber-50 rounded font-medium">View PR ↗</a>
                  ) : null}
                  {isPendingStatus(gdRow.status) ? (
                    <button
                      type="button"
                      disabled={gdRecheck}
                      className="text-xs px-3 py-1.5 border border-violet-300 text-violet-700 hover:bg-violet-50 rounded font-medium disabled:opacity-40 disabled:cursor-not-allowed"
                      title="Validate the saved spec again against the engine this server runs; no agent runs."
                      data-action="recheck"
                      onClick={() => void runAdminRecheck(gd)}
                    >
                      {gdRecheck ? "Re-checking…" : "Re-check"}
                    </button>
                  ) : null}
                  <button
                    type="button"
                    disabled={!!gdJob || isPendingStatus(gdRow.status)}
                    className="text-xs px-3 py-1.5 border border-gray-300 text-gray-700 hover:bg-gray-50 rounded font-medium disabled:opacity-40 disabled:cursor-not-allowed"
                    title={isPendingStatus(gdRow.status) ? "A pending record is re-checked, not reviewed." : gdJob ? "A review is already running on this record." : "Agent-review the spec against the model's real architecture (PASS / NO_PASS). Read-only."}
                    onClick={() => {
                      setGdActionStatus("");
                      setGdActionKind("");
                      void adminApi<{ status?: string; message?: string }>("POST", `/api/admin/spec-generations/${encodeURIComponent(gd)}/review`).then((r) => {
                        if (r.status === "started" || r.status === "already_running") pollAdminReview(gd);
                        else { setGdActionStatus(r.message || "Could not start the review."); setGdActionKind("err"); }
                      }).catch((e) => {
                        setGdActionStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
                        setGdActionKind("err");
                      });
                    }}
                  >
                    {gdJob ? "Reviewing…" : "Generate review"}
                  </button>
                  {gdActionStatus ? (
                    <span className={`text-xs ${gdActionKind === "err" ? "text-red-600" : gdActionKind === "ok" ? "text-green-600" : "text-gray-500"}`}>{gdActionStatus}</span>
                  ) : null}
                  {gdRow.status !== "merged" && gdRow.status !== "rejected" && gdRow.status !== "closed" && (
                    <button
                      type="button"
                      className="ml-auto text-xs px-3 py-1.5 border border-red-200 text-red-700 hover:bg-red-50 rounded font-medium"
                      title={isPendingStatus(gdRow.status) ? "A pending record is not a candidate: rejecting it deletes it." : undefined}
                      onClick={async () => {
                        if (!(await confirmDialog(isPendingStatus(gdRow.status)
                          ? { title: "Discard this pending record?", message: "It is deleted; this cannot be undone.",
                              confirmLabel: "Discard", danger: true }
                          : { title: "Reject this generation?", confirmLabel: "Reject", danger: true }))) return;
                        void adminApi("POST", `/api/admin/spec-generations/${encodeURIComponent(gd)}/reject`).then(() => {
                          closeGenDetail();
                          void loadTab("gens");
                        }).catch(alertError);
                      }}
                    >
                      {isPendingStatus(gdRow.status) ? "Discard" : "Reject"}
                    </button>
                  )}
                </div>
                <p className="text-11 text-gray-400 mt-2 leading-relaxed">
                  Admins review and decide (reject here, or merge the PR on GitHub). Fixing a spec or
                  opening / updating its PR is the contributor's own action, from their account page.
                </p>
                {gdRow.status === "blocked_by_engine" ? (
                  <div className="mt-3">
                    <h3 className="text-sm font-semibold text-violet-800 mb-2">Engine gaps</h3>
                    <EngineGapTable gaps={gdRow.engine_gaps as EngineGap[] | null} />
                  </div>
                ) : gdRow.status === "needs_repair" ? (
                  <div className="mt-3">
                    <h3 className="text-sm font-semibold text-amber-800 mb-2">Errors to repair</h3>
                    <ul className="text-xs text-gray-700 list-disc pl-5 space-y-1">
                      {((gdRow.repairable_errors as string[] | null) || []).map((e, i) => <li key={i}>{e}</li>)}
                    </ul>
                  </div>
                ) : null}
              </div>
            )}
            <div className="space-y-2 mb-5">
              {gdJob && <AgentJobCard kind="review" startedAt={gdJob.startedAt} steps={gdJob.steps} now={now} />}
              {gdReviews.map((r, i) => (
                <div key={i} className="border border-gray-200 rounded">
                  <div className="bg-gray-50 px-3 py-1.5 text-11 text-gray-600 border-b border-gray-200">
                    <ReviewBadge verdict={r.verdict} historical={!!gdRow?.library_spec} />
                    <span className="ml-2">{r.reviewer_role || ""} · {r.reviewer_email || ""} · {r.created_at ? new Date(r.created_at).toLocaleString() : ""}</span>
                  </div>
                  {r.verdict === "pass" ? <AdmissionLine row={r} /> : null}
                  <MarkdownReport text={r.report || ""} />
                </div>
              ))}
            </div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2 mt-2">Files</h3>
            <div className="mb-5">
              {gdRow ? (
                <SpecFiles modelId={String(gdRow?.model_id || "")} specYaml={String(gdRow.spec_yaml || "")}
                  components={gdRow.new_component_yamls as SpecFile[] | undefined} />
              ) : <div className="text-xs text-gray-400">…</div>}
            </div>
            <details className="mb-5">
              <summary className="text-sm font-semibold text-gray-700 cursor-pointer">Snapshot metadata</summary>
              <pre className="text-11 font-mono bg-gray-50 border border-gray-200 rounded p-3 overflow-x-auto whitespace-pre-wrap mt-2">{gdRow ? JSON.stringify(gdRow.snapshot_json || {}, null, 2) : "…"}</pre>
            </details>
          </aside>
        </div>
      )}
    </div>
  );
}
