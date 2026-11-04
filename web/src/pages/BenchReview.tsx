import { useEffect, useMemo, useState } from "react";
import { authJson, loadAuth, loginRedirect } from "../auth";
import { DataTable, type ColumnDef } from "../DataTable";
import { BenchOp } from "../BenchOp";

type BatchSummary = {
  row_count?: number;
  models?: string[];
  ops?: string[];
  gpus?: string[];
  frameworks?: string[];
  phases?: string[];
} & PeakCounts;

/** One op of the batch against shared, per op shape. */
type OverlapOp = { op?: string; op_shapes?: number; in_shared?: number; faster?: number };

/** One op shape of the batch shared already has: the batch's fastest and shared's. */
type OverlapShape = {
  signature?: string;
  physical_shape?: string;
  gpu?: string;
  phase?: string;
  latency_us?: number;
  framework?: string;
  kernel_impl?: string;
  shared_latency_us?: number;
  shared_framework?: string;
  shared_kernel_impl?: string;
  shared_candidates?: number;
};

/** The batch against the shared corpus, per op shape (`/api/bench/overlap`). */
type Overlap = {
  op_shapes?: number;
  in_shared?: number;
  new?: number;
  faster_than_shared?: number;
  by_op?: OverlapOp[];
  in_shared_list?: OverlapShape[];
};

type MeasRow = {
  id: string;
  model?: string;
  op?: string;
  kernel_impl?: string;
  phase?: string;
  shape_json?: unknown;
  latency_us?: number;
  /** The physical op the row's envelope keys on. */
  signature?: string;
  /** What happened: a measurement, or the refusal that took its place. */
  outcome?: string;
  outcome_detail?: string;
  achieved_tflops?: number;
  achieved_bw_gbps?: number;
  mfu?: number | null;
  mbu?: number | null;
  roof_status?: string | null;
};

const CARD = "bg-white border border-gray-200 rounded-2xl p-5 sm:p-6 shadow-sm";

/** One batch: a run or an upload of the caller's — or, for an admin
 *  reviewing a nomination, of `owner`'s (run ids are unique per owner). */
export type BenchReviewTarget = { kind: "run" | "upload"; id: string; view: string; owner?: string };

export function benchReviewTarget(kind: string, id: string, owner?: string): BenchReviewTarget {
  return { kind: kind === "run" ? "run" : "upload", id, view: "rows", ...(owner ? { owner } : {}) };
}

/** The batch as the bench API's query names it. */
function setQuery(t: BenchReviewTarget): string {
  const p = new URLSearchParams({ [t.kind === "run" ? "run_id" : "upload_id"]: t.id });
  if (t.owner) p.set("owner", t.owner);
  return p.toString();
}

/** The batch's rows as Kernel Bench lists them (`bench_browse` rows view):
 *  one page, each row with its physical op (`signature`) and MFU / MBU.
 *  `outcome=all`: the failures beside the measurements — a review that
 *  lists only the measured rows cannot see why a batch has no check. */
const SAMPLE_ROWS = 200;
function sampleRowsUrl(t: BenchReviewTarget): string {
  const p = new URLSearchParams({ view: "rows", source: "mine", [t.kind]: t.id,
                                  page: "0", page_size: String(SAMPLE_ROWS),
                                  outcome: "all" });
  if (t.owner) p.set("owner", t.owner);
  return `/api/bench/browse?${p}`;
}

/** The kernel-corpus page filtered to this one batch. */
export function benchCorpusHref(t: BenchReviewTarget): string {
  // Reviewing a batch reads every measured row, not the envelopes they fold into.
  return `/kernel-corpus.html#${new URLSearchParams({ view: "rows", source: "mine", [t.kind]: t.id, ...(t.owner ? { owner: t.owner } : {}) })}`;
}

/** A batch summary's roofline and refusal counts (`sets._COUNT_SQL`): rows
 *  over the recorded compute / bandwidth peak, rows outside the roofline
 *  check, how many rows could be checked at all, and how the refused rows
 *  (no measurement, no peaks) divide by outcome. */
export type PeakCounts = { mfu_over?: number; mbu_over?: number; anomalous?: number; checked?: number;
  measured?: number; refused?: number;
  refused_oom?: number; refused_mismatch?: number; refused_error?: number;
  refused_crash?: number; refused_timeout?: number };

/** "error×3, oom×1" — what the refused rows of a batch recorded. */
export function refusedBreakdown(u: PeakCounts): string {
  return (["error", "mismatch", "oom", "crash", "timeout"] as const)
    .map((o) => {
      const n = u[`refused_${o}`] || 0;
      return n ? `${o}×${n.toLocaleString()}` : "";
    })
    .filter(Boolean).join(", ");
}

/** How a batch's rows divide: the measured ones split by the roofline
 *  check into ok and over peak, and the refused ones (no measurement, so
 *  neither ok nor over) beside them — ok + over + refused = rows. */
export function rowSplit(u: PeakCounts, rows: number): { measured: number; ok: number; over: number; refused: number } {
  const refused = u.refused || 0;
  const measured = u.measured ?? Math.max(0, rows - refused);
  const over = u.anomalous || 0;
  return { measured, ok: Math.max(0, measured - over), over, refused };
}

function overCell(n: number | undefined, checked: number | undefined) {
  if (n) return <span className="text-amber-800">{n.toLocaleString()} over</span>;
  if ((checked || 0) > 0) return <span className="text-green-700">ok</span>;
  return "–";
}

/** The Check column's cell: the roofline verdict, or — when no row could be
 *  checked — what the batch's rows did instead. */
function checkCell(u: PeakCounts) {
  if ((u.checked || 0) > 0) return overCell(u.anomalous, u.checked);
  if ((u.refused || 0) > 0)
    return <span className="text-red-700" title={refusedBreakdown(u)}>{(u.refused || 0).toLocaleString()} failed</span>;
  return "–";
}

/** The MFU / MBU / Check columns of a table of batch summaries. */
export function peakCheckColumns<T extends PeakCounts>(): ColumnDef<T>[] {
  return [
    { key: "mfu", header: "MFU", align: "right",
      render: (u) => overCell(u.mfu_over, u.checked), sortValue: (u) => (u.checked ? u.mfu_over ?? 0 : null) },
    { key: "mbu", header: "MBU", align: "right",
      render: (u) => overCell(u.mbu_over, u.checked), sortValue: (u) => (u.checked ? u.mbu_over ?? 0 : null) },
    { key: "roof", header: "Check",
      render: (u) => checkCell(u), sortValue: (u) => (u.checked ? u.anomalous ?? 0 : null) },
  ];
}

function chips(items: string[] | undefined, empty = "—") {
  const list = [...new Set((items || []).filter(Boolean))].sort();
  if (!list.length) return <span className="text-gray-400">{empty}</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {list.map((x) => (
        <span key={x} className="text-11 px-2 py-0.5 rounded-full bg-gray-100 text-gray-700">{x.includes("/") ? x.split("/").pop() : x}</span>
      ))}
    </div>
  );
}

function fmtPct(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x) || x <= 0) return "–";
  return (x > 0 && x < 0.001) ? "<0.1%" : `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
}

function roofLabel(status: string | null | undefined) {
  if (!status) return "–";
  if (status === "ok") return <span className="text-green-700">ok</span>;
  const text = status === "mfu" ? "MFU>1" : status === "mbu" ? "MBU>1" : "MFU+MBU>1";
  return <span className="text-amber-800">{text}</span>;
}

/** What happened to a row that is not a measurement (`outcome_detail` is
 *  the reason the run recorded); a measurement leaves the cell empty. */
function outcomeLabel(r: MeasRow) {
  const o = r.outcome || "measured";
  if (o === "measured") return "";
  return <span className="text-red-700" title={r.outcome_detail || undefined}>{o}</span>;
}

function shapeShort(v: unknown): string {
  try {
    const d = (typeof v === "object" && v) ? v as Record<string, unknown> : JSON.parse(String(v));
    return Object.entries(d).map(([k, x]) => `${k}=${x}`).join(" ");
  } catch {
    return String(v || "");
  }
}

function query(): BenchReviewTarget | null {
  const p = new URLSearchParams(window.location.search);
  const run = p.get("run") || "";
  const upload = p.get("upload") || "";
  const owner = p.get("owner") || undefined;
  if (run) return { kind: "run", id: run, view: p.get("view") || "", owner };
  if (upload) return { kind: "upload", id: upload, view: p.get("view") || "", owner };
  return null;
}

/** Summary + sample rows for one batch. Rendered by the standalone page
 *  (deep links) and by the workspace's in-page dialog. */
function BenchReviewBody({ q }: { q: BenchReviewTarget | null }) {
  const [summary, setSummary] = useState<BatchSummary | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);
  const [rows, setRows] = useState<MeasRow[] | null>(null);
  const [rowsErr, setRowsErr] = useState("");
  const [rowsLoading, setRowsLoading] = useState(false);
  const [overlap, setOverlap] = useState<Overlap | null>(null);
  const [overlapErr, setOverlapErr] = useState("");
  const [overlapLoading, setOverlapLoading] = useState(false);

  useEffect(() => {
    if (!q) {
      setErr("Missing run or upload id.");
      setLoading(false);
      return;
    }
    let cancelled = false;
    (async () => {
      const auth = await loadAuth();
      if (cancelled) return;
      if (auth.authEnabled && !auth.user) {
        loginRedirect();
        return;
      }
      const key = setQuery(q);
      try {
        const data = await authJson<BatchSummary>("GET", `/api/bench/summary?${key}`);
        if (!cancelled) setSummary(data);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
      if (!cancelled) setOverlapLoading(true);
      try {
        const ov = await authJson<Overlap>("GET", `/api/bench/overlap?${key}`);
        if (!cancelled) setOverlap(ov);
      } catch (e) {
        if (!cancelled) setOverlapErr(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setOverlapLoading(false);
      }
      if (q.view === "rows") {
        if (!cancelled) setRowsLoading(true);
        try {
          const data = await authJson<{ items?: MeasRow[] }>("GET", sampleRowsUrl(q));
          if (!cancelled) setRows(data.items || []);
        } catch (e) {
          if (!cancelled) setRowsErr(e instanceof Error ? e.message : String(e));
        } finally {
          if (!cancelled) setRowsLoading(false);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [q]);

  const loadRows = async () => {
    if (!q || rowsLoading) return;
    setRowsLoading(true);
    setRowsErr("");
    try {
      const data = await authJson<{ items?: MeasRow[] }>("GET", sampleRowsUrl(q));
      setRows(data.items || []);
    } catch (e) {
      setRowsErr(e instanceof Error ? e.message : String(e));
    } finally {
      setRowsLoading(false);
    }
  };

  const n = summary?.row_count || 0;
  const { measured, ok, over: bad, refused } = rowSplit(summary ?? {}, n);
  const measCols: ColumnDef<MeasRow>[] = [
    { key: "model", header: "Model", className: "text-11", render: (r) => String(r.model || "").split("/").pop(), sortValue: (r) => String(r.model || "").split("/").pop() },
    { key: "op", header: "Op", render: (r) => <BenchOp signature={r.signature} op={r.op} />, sortValue: (r) => r.signature || r.op },
    { key: "kernel", header: "Kernel", className: "font-mono text-11 text-gray-500", render: (r) => r.kernel_impl, sortValue: (r) => r.kernel_impl },
    { key: "phase", header: "Phase", className: "text-gray-500", sortValue: (r) => r.phase },
    { key: "shape", wrap: true, header: "Shape", className: "font-mono text-11 text-gray-500", render: (r) => shapeShort(r.shape_json), sortValue: (r) => shapeShort(r.shape_json) },
    { key: "latency", header: "latency µs", align: "right", render: (r) => (r.latency_us != null ? Number(r.latency_us).toFixed(1) : "–"), sortValue: (r) => (r.latency_us != null ? Number(r.latency_us) : null) },
    { key: "tflops", header: "TFLOP/s", align: "right", className: "text-gray-500", render: (r) => (r.achieved_tflops != null ? Number(r.achieved_tflops).toFixed(2) : "–"), sortValue: (r) => (r.achieved_tflops != null ? Number(r.achieved_tflops) : null) },
    { key: "bw", header: "GB/s", align: "right", className: "text-gray-500", render: (r) => (r.achieved_bw_gbps != null ? Number(r.achieved_bw_gbps).toFixed(0) : "–"), sortValue: (r) => (r.achieved_bw_gbps != null ? Number(r.achieved_bw_gbps) : null) },
    { key: "mfu", header: "MFU", align: "right",
      className: (r) => (r.mfu != null && r.mfu > 1 ? "text-amber-800" : "text-gray-500"),
      render: (r) => fmtPct(r.mfu), sortValue: (r) => r.mfu },
    { key: "mbu", header: "MBU", align: "right",
      className: (r) => (r.mbu != null && r.mbu > 1 ? "text-amber-800" : "text-gray-500"),
      render: (r) => fmtPct(r.mbu), sortValue: (r) => r.mbu },
    { key: "roof", header: "Check", render: (r) => roofLabel(r.roof_status), sortValue: (r) => r.roof_status },
    { key: "outcome", header: "Outcome", render: (r) => outcomeLabel(r), sortValue: (r) => r.outcome || "measured" },
  ];

  // No batch named: nothing below has a batch to read, so only say so.
  if (!q) return <p className="text-sm text-red-600">{err}</p>;

  return (
    <>
      <p className="text-11 font-mono text-gray-400 break-all">{q.id}</p>
      {loading && <p className="text-sm text-gray-400">Loading summary…</p>}
      {err && <p className="text-sm text-red-600">{err}</p>}
      {summary && (
        <section className={CARD}>
          <p className="text-sm text-gray-800 mb-4">
            <span className="font-medium">{n.toLocaleString()} rows</span>
            {measured === 0 ? null : bad === 0
              ? <span className="text-green-700"> · {refused ? "all measured" : "all"} at or under roofline (MFU≤1, MBU≤1)</span>
              : <span className="text-amber-800"> · {ok.toLocaleString()} ok, {bad.toLocaleString()} over peak ({(summary.mfu_over || 0).toLocaleString()} MFU, {(summary.mbu_over || 0).toLocaleString()} MBU)</span>}
            {refused > 0 &&
              <span className="text-red-700" title={refusedBreakdown(summary)}> · {refused.toLocaleString()} failed: {refusedBreakdown(summary)}</span>}
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <p className="text-xs text-gray-500 mb-1">Models</p>
              {chips(summary.models)}
            </div>
            <div>
              <p className="text-xs text-gray-500 mb-1">GPUs</p>
              {chips(summary.gpus)}
            </div>
            <div>
              <p className="text-xs text-gray-500 mb-1">Frameworks</p>
              {chips(summary.frameworks)}
            </div>
            <div>
              <p className="text-xs text-gray-500 mb-1">Phases</p>
              {chips(summary.phases)}
            </div>
            <div className="sm:col-span-2">
              <p className="text-xs text-gray-500 mb-1" title="As the model's graph names them; the tables against shared key on the physical op">Ops in the model graph</p>
              {chips(summary.ops)}
            </div>
          </div>
        </section>
      )}
      <section className={CARD}>
        <h2 className="text-sm font-semibold text-gray-900 mb-1">Against shared</h2>
        <p className="text-xs text-gray-500 mb-3">
          This batch&apos;s op shapes (the key estimates look measurements up by) against the shared corpus, computed now.
        </p>
        {overlapLoading && <p className="text-sm text-gray-400">Comparing…</p>}
        {overlapErr && <p className="text-sm text-red-600">{overlapErr}</p>}
        {overlap && (
          <>
            <p id="overlap-summary" className="text-sm text-gray-800 mb-3">
              <span className="font-medium">{(overlap.new || 0).toLocaleString()} new</span>
              <span className="text-gray-600"> · {(overlap.in_shared || 0).toLocaleString()} already in shared</span>
              <span className="text-gray-600"> · faster than shared on {(overlap.faster_than_shared || 0).toLocaleString()}</span>
              <span className="text-gray-400"> · {(overlap.op_shapes || 0).toLocaleString()} op shapes</span>
            </p>
            {(overlap.by_op || []).length > 0 && (
              <DataTable
                className="mb-3"
                columns={[
                  { key: "op", header: "Physical op", sortValue: (r) => r.op },
                  { key: "op_shapes", header: "Op shapes", align: "right", sortValue: (r) => r.op_shapes },
                  { key: "in_shared", header: "In shared", align: "right", sortValue: (r) => r.in_shared },
                  { key: "faster", header: "Faster than shared", align: "right", sortValue: (r) => r.faster },
                ]}
                rows={overlap.by_op || []}
                rowKey={(r, i) => `${r.op}:${i}`}
                empty=""
              />
            )}
            {(overlap.in_shared_list || []).length > 0 && (
              <>
                <p className="text-xs text-gray-500 mb-1">
                  {(overlap.in_shared_list || []).length} of {(overlap.in_shared || 0).toLocaleString()} op shapes shared already has, this batch&apos;s fastest relative to shared&apos;s first
                </p>
                <DataTable
                  columns={[
                    // Keyed by the physical op the envelope matches on, which a
                    // fused kernel's row names differently (flash_attention
                    // timing a gqa_attention shape).
                    { key: "op", header: "Physical op", sortValue: (r) => r.signature },
                    { key: "gpu", header: "GPU", sortValue: (r) => r.gpu },
                    { key: "phase", header: "Phase", className: "text-gray-500", sortValue: (r) => r.phase },
                    { key: "shape", wrap: true, header: "Shape", className: "font-mono text-11 text-gray-500", render: (r) => shapeShort(r.physical_shape), sortValue: (r) => shapeShort(r.physical_shape) },
                    { key: "mine", header: "This batch µs", align: "right", render: (r) => Number(r.latency_us || 0).toFixed(1), sortValue: (r) => r.latency_us },
                    { key: "mine_kernel", header: "Kernel", className: "font-mono text-11 text-gray-500", render: (r) => `${r.framework || ""}/${r.kernel_impl || ""}`, sortValue: (r) => r.kernel_impl },
                    { key: "shared", header: "Shared µs", align: "right", render: (r) => Number(r.shared_latency_us || 0).toFixed(1), sortValue: (r) => r.shared_latency_us },
                    { key: "shared_kernel", header: "Shared kernel", className: "font-mono text-11 text-gray-500", render: (r) => `${r.shared_framework || ""}/${r.shared_kernel_impl || ""}`, sortValue: (r) => r.shared_kernel_impl },
                    { key: "ratio", header: "vs shared", align: "right", render: (r) => r.shared_latency_us ? `${((Number(r.latency_us) / Number(r.shared_latency_us) - 1) * 100).toFixed(0)}%` : "–", sortValue: (r) => r.shared_latency_us ? Number(r.latency_us) / Number(r.shared_latency_us) : null },
                  ]}
                  rows={overlap.in_shared_list || []}
                  rowKey={(r, i) => `${r.signature}:${r.physical_shape}:${r.gpu}:${r.phase}:${i}`}
                  empty=""
                />
              </>
            )}
            {(overlap.in_shared || 0) === 0 && (
              <p className="text-xs text-green-700">Shared has none of this batch&apos;s op shapes: all of it is new.</p>
            )}
          </>
        )}
      </section>
      <section className={CARD}>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-900">Sample rows</h2>
          {rows == null && (
            <button type="button" className="text-xs px-2 py-1 border rounded hover:bg-gray-50" onClick={() => void loadRows()} disabled={rowsLoading}>
              {rowsLoading ? "Loading…" : `Load ${SAMPLE_ROWS}`}
            </button>
          )}
        </div>
        {rowsErr && <p className="text-sm text-red-600 mb-2">{rowsErr}</p>}
        {rows && (
          <>
            <p className="text-xs text-gray-500 mb-2">{rows.length} shown</p>
            <DataTable columns={measCols} rows={rows} rowKey={(r) => r.id} empty="No rows." />
          </>
        )}
        {rows == null && !rowsLoading && (
          <p className="text-xs text-gray-400">A sample of this batch, not the full sweep.</p>
        )}
      </section>
    </>
  );
}

export function BenchReview() {
  const q = useMemo(query, []);
  useEffect(() => {
    document.title = "Batch review · HuggingArch";
    return () => { document.title = "HuggingArch"; };
  }, []);
  return (
    <div className="bg-gray-50 min-h-screen p-4 sm:p-6">
      <div className="max-w-3xl mx-auto space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-10 font-semibold uppercase tracking-wider text-amber-700">Kernel benchmarks</p>
            <h1 className="text-lg font-semibold text-gray-900">Batch review</h1>
          </div>
          <a href="/account/workspace.html#benchmarks" className="text-xs text-gray-500 hover:text-amber-700">Workspace</a>
        </div>
        <BenchReviewBody q={q} />
      </div>
    </div>
  );
}

export function BenchReviewModal({ target, onClose }: { target: BenchReviewTarget; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-gray-50 rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-200 bg-white rounded-t-xl">
          <div>
            <p className="text-10 font-semibold uppercase tracking-wider text-amber-700">Kernel benchmarks</p>
            <h2 className="text-base font-semibold text-gray-900">Batch review</h2>
          </div>
          <button type="button" className="text-gray-400 hover:text-gray-700 text-2xl leading-none shrink-0" onClick={onClose}>&times;</button>
        </div>
        <div className="flex-1 overflow-auto p-4 sm:p-5 space-y-4">
          <BenchReviewBody key={`${target.owner || ""}:${target.kind}:${target.id}:${target.view}`} q={target} />
        </div>
      </div>
    </div>
  );
}
