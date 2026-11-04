import { useEffect, useState, type ReactNode } from "react";
import type { EChartsOption } from "echarts";
import { fetchJson } from "../../api";
import { ComputeProgress } from "../../ProgressBar";
import { DataTable, type ColumnDef } from "../../DataTable";
import { AXIS, HaChart } from "../../charts/HaChart";
import { fmtCompact, fmtDuration, fmtSeqLabel, fmtTime, formatBytes, formatSI } from "../../format";
import { Section } from "./Section";

type PhaseKey = "prefill" | "decode";

/** One length of one phase (backend `latency_sweep` point): the bench's
 *  timing totals there, and what the phase bound (`s_query` new tokens per
 *  request against `s_kv` positions). */
type LatencyPoint = {
  len: number;
  batch: number;
  s_query: number;
  s_kv: number;
  t_sol_s: number;
  t_measured_s: number | null;
  t_compute_s: number;
  t_memory_s: number;
  t_comm_s: number;
  measured_rows: number;
  total_rows: number;
  /** Σ over the same kernels: FLOPs, HBM access, and their ratio (null
   *  when the step moves no bytes). */
  flops: number;
  access_bytes: number;
  arithmetic_intensity: number | null;
};
type LatencyRow = {
  entry: string;
  component: string;
  model_id: string | null;
  phases: Record<PhaseKey, LatencyPoint[]>;
};
type LatencyResponse = {
  success: boolean;
  message?: string;
  contract: { batch: number; gpu_sku: string; lens: number[]; unit: string };
  results: LatencyRow[];
  errors: { entry?: string; component: string; error: string }[];
};

/** How a bench entry is drawn here: the label and color it has elsewhere on
 *  the page. */
export type LatencyEntryStyle = { label: string; color: string };

function fmtIntensity(ai: number | null | undefined): string {
  return ai == null || !Number.isFinite(ai) ? "—" : `${fmtCompact(ai)} FLOPs/B`;
}

function pointTitle(p: LatencyPoint): string {
  const parts = [
    `${p.s_query.toLocaleString("en-US")} new token${p.s_query === 1 ? "" : "s"}/request against ${p.s_kv.toLocaleString("en-US")} positions, batch ${p.batch}`,
    `Σ compute ${fmtTime(p.t_compute_s)} · Σ memory ${fmtTime(p.t_memory_s)}${p.t_comm_s > 0 ? ` · Σ comm ${fmtTime(p.t_comm_s)}` : ""}`,
    p.t_measured_s != null
      ? `measured ${fmtTime(p.t_measured_s)} (every kernel)`
      : `measured ${p.measured_rows}/${p.total_rows} kernels — no measured total`,
    `AI ${fmtIntensity(p.arithmetic_intensity)} = ${formatSI(p.flops, "FLOPs")} / ${formatBytes(p.access_bytes)}`,
  ];
  return parts.join("\n");
}

/** Prefill and decode time of each entry across the Scaling sweep's
 *  lengths, on the selected GPU at the bench's batch. Every number is the
 *  backend's (`/api/playground/module-latency-sweep`, the bench's own
 *  timing at each length); `body` is the bench request it extends, null
 *  while the bench is still settling. */
export function LatencySection({ body, styleOf, phase, onPhase, atLen, serving }: {
  body: string | null;
  styleOf: (entry: string) => LatencyEntryStyle | undefined;
  phase: PhaseKey;
  onPhase: (p: PhaseKey) => void;
  atLen: number;
  serving: ReactNode;
}) {
  const [data, setData] = useState<LatencyResponse | null>(null);
  const [status, setStatus] = useState<{ text: string; progress?: { done: number; total: number }; failed?: boolean } | null>(null);
  useEffect(() => {
    if (body == null) return;
    const ctl = new AbortController();
    setStatus({ text: "Timing each entry at every length…" });
    fetchJson<LatencyResponse>("/api/playground/module-latency-sweep", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal: ctl.signal,
    }, (info) => {
      const p = info.progress;
      setStatus(p
        ? { text: `Still timing — ${p.done}/${p.total}…`, progress: p }
        : { text: "Still timing…" });
    }).then((d) => {
      if (!d.success) {
        setStatus({ text: `Failed: ${d.message || "unknown error"}`, failed: true });
        return;
      }
      setData(d);
      setStatus(null);
    }).catch((e: unknown) => {
      if (ctl.signal.aborted) return;
      setStatus({ text: `Failed: ${e instanceof Error ? e.message : String(e)}`, failed: true });
    });
    return () => ctl.abort();
  }, [body]);

  const btn = (k: PhaseKey) =>
    "px-2 py-1 text-xs " + (phase === k ? "bg-gray-700 text-white" : "bg-white text-gray-600 hover:bg-gray-50");
  const rows = (data?.results ?? []).filter((r) => r.phases[phase]?.length);
  const lens = data?.contract.lens ?? [];
  const valueOf = (r: LatencyRow, i: number) => r.phases[phase][i]?.t_sol_s ?? NaN;
  const minOf = lens.map((_, i) => Math.min(...rows.map((r) => valueOf(r, i))));
  const label = (r: LatencyRow) => styleOf(r.entry)?.label ?? r.entry;
  const color = (r: LatencyRow) => styleOf(r.entry)?.color ?? "#888";

  const columns: ColumnDef<LatencyRow>[] = [{
    key: "entry",
    header: "entry",
    className: "whitespace-nowrap",
    sortValue: label,
    render: (r) => (
      <>
        <span className="inline-block w-2.5 h-2.5 rounded-sm mr-1.5 align-middle" style={{ background: color(r) }} />
        <span className="font-mono text-gray-800">{label(r)}</span>
      </>
    ),
  }];
  lens.forEach((n, i) => {
    const atL = n === atLen;
    const best = (r: LatencyRow) => rows.length > 1 && valueOf(r, i) === minOf[i];
    columns.push({
      key: `len-${n}`,
      header: <span title={`${n.toLocaleString("en-US")} tokens`}>{fmtSeqLabel(n)}</span>,
      align: "right",
      headerClassName: atL ? "bg-amber-50 text-amber-800" : undefined,
      className: (r) =>
        `pg-lat-cell font-mono whitespace-nowrap ${atL ? "bg-amber-50" : ""} ${best(r) ? "text-emerald-700 font-semibold" : "text-gray-800"}`,
      cellProps: (r) => {
        const p = r.phases[phase][i];
        return { title: p ? pointTitle(p) : undefined, "data-value": p?.t_sol_s };
      },
      render: (r) => {
        const p = r.phases[phase][i];
        if (!p) return "—";
        return (
          <>
            {fmtTime(p.t_sol_s)}
            {p.t_measured_s != null && (
              <div className="text-11 font-normal text-gray-500">meas. {fmtTime(p.t_measured_s)}</div>
            )}
          </>
        );
      },
      sortValue: (r) => r.phases[phase][i]?.t_sol_s ?? null,
    });
  });

  const xAxis = (): EChartsOption["xAxis"] => ({
    type: "log", logBase: 2, min: Math.max(1, lens[0] || 1),
    name: phase === "prefill" ? "prompt length (tokens)" : "cache_len (tokens)",
    nameLocation: "middle", nameGap: 24, ...AXIS,
    axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => fmtSeqLabel(v) },
    axisPointer: { label: { formatter: ({ value }) => fmtSeqLabel(Number(value)) } },
  });
  const seriesOf = (pick: (p: LatencyPoint | undefined) => number) => rows.map((r) => ({
    type: "line" as const,
    name: label(r),
    data: lens.map((n, i) => [n, pick(r.phases[phase][i])]),
    lineStyle: { color: color(r), width: 2 },
    itemStyle: { color: color(r) },
    symbolSize: 5,
  }));

  const option: EChartsOption = {
    grid: { left: 72, right: 16, top: 40, bottom: 40 },
    tooltip: { trigger: "axis", order: "valueDesc", valueFormatter: (v) => fmtDuration(Number(v)) },
    legend: { top: 0, textStyle: { fontSize: 9 } },
    xAxis: xAxis(),
    // Times span decades (µs at 1K to seconds at 1M): log scale.
    yAxis: { type: "log", ...AXIS, axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => fmtDuration(v) } },
    series: seriesOf((p) => p?.t_sol_s ?? NaN),
  };
  // Intensity spans decades too (≈1 FLOP/B for a decode step's reads to
  // thousands for a long prefill): log scale.
  const aiOption: EChartsOption = {
    grid: { left: 72, right: 16, top: 40, bottom: 40 },
    tooltip: { trigger: "axis", order: "valueDesc", valueFormatter: (v) => fmtIntensity(Number(v)) },
    legend: { top: 0, textStyle: { fontSize: 9 } },
    xAxis: xAxis(),
    yAxis: { type: "log", ...AXIS, name: "FLOPs / byte",
             axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => fmtCompact(v) } },
    series: seriesOf((p) => p?.arithmetic_intensity ?? NaN),
  };

  return (
    <Section id="pg-latency" scope="per layer · whole batch · roofline on the selected GPU" title="Latency" lead={serving}
      footnote={data && <>
        t SOL = Σ over the step's kernels of max(compute, memory, comm, launch) on {data.contract.gpu_sku}, one call of the module, batch {data.contract.batch}; kernels serial, no overlap.
        {" "}Prefill at L: a prompt of L tokens per request. Decode at L: one new token per request against L cached tokens.
        {" "}Hover a cell for the compute / memory split; "meas." appears only where every kernel has a measurement.
        {" "}Arithmetic intensity = Σ FLOPs / Σ HBM access bytes over the same kernels.
      </>}>
      <div className="flex items-baseline justify-between flex-wrap gap-2 mt-3">
        <div className="text-sm font-medium text-gray-700">
          {phase === "prefill" ? "Prefill time vs prompt length" : "Decode step time vs context length"}{" "}
          <span className="text-xs font-normal text-gray-400">
            {status && (status.failed
              ? status.text
              : <ComputeProgress compact label={status.text} progress={status.progress} />)}
          </span>
        </div>
        <div className="inline-flex rounded border border-gray-200 overflow-hidden" role="tablist" aria-label="Latency phase">
          <button id="pg-lat-prefill" type="button" className={btn("prefill")} aria-selected={phase === "prefill"} onClick={() => onPhase("prefill")}>Prefill</button>
          <button id="pg-lat-decode" type="button" className={btn("decode")} aria-selected={phase === "decode"} onClick={() => onPhase("decode")}>Decode</button>
        </div>
      </div>
      {data && data.errors.length > 0 && (
        <div className="text-11 text-amber-700 mt-1">
          {data.errors.map((e) => `${styleOf(e.entry ?? "")?.label ?? e.entry ?? e.component}: ${e.error}`).join(" · ")}
        </div>
      )}
      {rows.length > 0 && (
        <>
          <div className="mt-2">
            <DataTable columns={columns} rows={rows} rowKey={(r) => r.entry} framed={false} />
            <div className="text-11 text-gray-400 mt-1">green = fastest in its column</div>
          </div>
          <div className="mt-3">
            <HaChart option={option} height={260} />
          </div>
          <div className="text-sm font-medium text-gray-700 mt-4">
            {phase === "prefill" ? "Prefill" : "Decode"} arithmetic intensity vs {phase === "prefill" ? "prompt length" : "context length"}
          </div>
          <div className="mt-2">
            <HaChart option={aiOption} height={240} />
          </div>
        </>
      )}
    </Section>
  );
}
