import { useEffect, useState } from "react";
import type { EChartsOption } from "echarts";
import { AXIS, HaChart, interactiveBits } from "../charts/HaChart";
import { fetchJson, isAbort } from "../api";
import { SWEEP_COLORS, SWEEP_SEQ_LENS, type SweepHit } from "../compare";
import { fmtCompact, formatNumber, fmtSeqLabel } from "../format";

const BATCH_OPTS = [
  { value: "all", label: "all (curves)" },
  { value: "max", label: "max" },
  { value: "b1", label: "decode 1" },
  { value: "b16", label: "decode 16" },
  { value: "b64", label: "decode 64" },
] as const;

const DEFAULT_SEQ = SWEEP_SEQ_LENS[Math.floor(SWEEP_SEQ_LENS.length / 2)];

/** One row's decode point at one (context, batch key), as the backend reads
 *  it off the row's sweep (`CapacitySweepPlan.decode_cloud`). */
type CloudPt = {
  row: number;
  context: number;
  batch_key: string;
  batch?: number | null;
  tpot_s?: number;
  tok_s_per_gpu?: number;
  tpot_measured_s?: number;
  tok_s_per_gpu_measured?: number;
  refused?: string;
  reason?: string;
};
type Front = { group: { context: number; batch_key?: string; row?: number }; frontier: CloudPt[] };
/** `/inference/estimate-sweep/frontier`: every row's points and, per track,
 *  its views at every context — the front across the rows at each batch
 *  point, each row's own line over its batch points, and their envelope. */
type CompareFronts = {
  points: CloudPt[];
  fronts: Record<"sol" | "measured", { by_batch: Front[]; by_row: Front[]; envelope: Front[] }>;
};

type Pt = {
  key: number;
  label: string;
  x: number;
  y: number;
  color: string;
  onFront: boolean;
  ratio: number | null;
  isBase: boolean;
};

export function CompareTradeoff({
  hits,
  labels,
  source,
  onSource,
}: {
  hits: SweepHit[];
  labels: string[];
  source: string;
  onSource: (v: string) => void;
}) {
  const [seq, setSeq] = useState(DEFAULT_SEQ);
  const [batch, setBatch] = useState("all");
  const [baseIdx, setBaseIdx] = useState(0);
  // The rows with an answer, in the order the view request names them.
  const answered = hits.flatMap((h, i) => (h.result && h.sent ? [i] : []));
  const reqBody = JSON.stringify({ sweeps: answered.map((i) => hits[i].sent) });
  const [fronts, setFronts] = useState<{ key: string; data: CompareFronts } | null>(null);
  const [failed, setFailed] = useState("");
  useEffect(() => {
    if (!answered.length) return;
    const ctrl = new AbortController();
    setFailed("");
    fetchJson<CompareFronts>("/inference/estimate-sweep/frontier", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: reqBody,
      signal: ctrl.signal,
    })
      .then((data) => setFronts({ key: reqBody, data }))
      .catch((e) => {
        if (!isAbort(e)) setFailed(e instanceof Error ? e.message : String(e));
      });
    return () => ctrl.abort();
    // `answered` is read off `hits`, which `reqBody` names.
  }, [reqBody]);

  if (!hits.length) return null;

  const baseKey = Math.min(baseIdx, hits.length - 1);
  const seqIdx = SWEEP_SEQ_LENS.indexOf(seq);
  const si = seqIdx >= 0 ? seqIdx : Math.floor(SWEEP_SEQ_LENS.length / 2);
  const seqVal = SWEEP_SEQ_LENS[si];

  // The backend's points and front at this (context, batch key), on this
  // track; a view drawn for other rows is not shown.
  const data = fronts?.key === reqBody ? fronts.data : null;
  const measured = source === "measured";
  const at = (p: CloudPt) => p.context === seqVal && p.batch_key === batch;
  const views = data?.fronts[measured ? "measured" : "sol"];
  const curves = batch === "all";
  const front = new Set((views?.by_batch || [])
    .find((f) => f.group.context === seqVal && f.group.batch_key === batch)
    ?.frontier.map((p) => p.row) || []);
  // A point on this track, as drawn: TPOT in ms against tokens/s per GPU.
  const xy = (p: CloudPt): [number, number] | null => {
    const tpot = measured ? p.tpot_measured_s : p.tpot_s;
    const thru = measured ? p.tok_s_per_gpu_measured : p.tok_s_per_gpu;
    return tpot == null || thru == null ? null : [tpot * 1000, thru];
  };
  const rowLabel = (row: number) => labels[answered[row]] || `#${answered[row] + 1}`;
  const rowColor = (row: number) => SWEEP_COLORS[answered[row] % SWEEP_COLORS.length];
  const pointTip = (p: CloudPt, v: [number, number]) =>
    `${rowLabel(p.row)} · batch ${p.batch ?? "—"}: ${v[0].toFixed(2)} ms, ${formatNumber(Math.round(v[1]))} tok/s per GPU`;
  const raw: Omit<Pt, "ratio" | "isBase">[] = [];
  for (const p of (data?.points || []).filter(at)) {
    const v = xy(p);
    if (!v) continue;
    raw.push({
      key: answered[p.row],
      label: rowLabel(p.row),
      x: v[0],
      y: v[1],
      color: rowColor(p.row),
      onFront: front.has(p.row),
    });
  }

  const base = raw.find((p) => p.key === baseKey) || raw[0];
  const pts: Pt[] = raw.map((p) => ({
    ...p,
    ratio: base && base !== p && base.y > 0 ? p.y / base.y : null,
    isBase: !!base && p.key === base.key,
  }));
  const frontPts = pts.filter((p) => p.onFront).sort((a, b) => a.x - b.x);
  const batchLab = batch === "max" ? "max" : batch.slice(1);
  const head = curves
    ? `One line per row through its batch points (1 → max) · ◆ = on the envelope over every row and batch · seq ${fmtSeqLabel(seqVal)}`
    : `◆ = Pareto-optimal · ×N = per-GPU throughput vs baseline · seq ${fmtSeqLabel(seqVal)}, batch ${batchLab}`;
  // Each row's line at this context and the envelope over them all.
  const rowLines = (views?.by_row || []).filter((f) => f.group.context === seqVal)
    .map((f) => ({ row: f.group.row as number, pts: f.frontier.flatMap((p) => {
      const v = xy(p);
      return v ? [{ p, v }] : [];
    }).sort((a, b) => a.v[0] - b.v[0]) }))
    .filter((l) => l.pts.length);
  const envelope = ((views?.envelope || []).find((f) => f.group.context === seqVal)?.frontier || [])
    .flatMap((p) => {
      const v = xy(p);
      return v ? [{ p, v }] : [];
    }).sort((a, b) => a.v[0] - b.v[0]);
  const logAxis = (name: string, gap: number) => ({
    type: "log" as const, name, nameLocation: "middle" as const, nameGap: gap, ...AXIS,
    min: (v: { min: number }) => v.min / 1.3, max: (v: { max: number }) => v.max * 1.3,
    axisLabel: { ...AXIS.axisLabel, formatter: (v: string | number) => fmtCompact(Number(v)),
                 showMinLabel: false, showMaxLabel: false },
  });
  const curvesOption: EChartsOption = {
    ...interactiveBits(),
    grid: { left: 72, right: 160, top: 32, bottom: 48 },
    tooltip: {
      trigger: "item",
      formatter: (raw: unknown) => (raw as { data?: { tip?: string } }).data?.tip || "",
    },
    xAxis: logAxis("TPOT (ms, log) — lower is better", 28),
    yAxis: logAxis("decode throughput (tok/s per GPU, log) — higher is better", 56),
    series: [
      ...(envelope.length ? [{
        type: "line" as const,
        name: "envelope",
        data: envelope.map(({ p, v }) => ({ value: v, tip: `envelope · ${pointTip(p, v)}` })),
        symbol: "diamond",
        symbolSize: 11,
        lineStyle: { color: "rgba(16,185,129,0.55)", width: 2, type: "dashed" as const },
        itemStyle: { color: "rgba(16,185,129,0.9)" },
        z: 1,
      }] : []),
      ...rowLines.map(({ row, pts: line }) => ({
        type: "line" as const,
        name: rowLabel(row),
        data: line.map(({ p, v }) => ({ value: v, tip: pointTip(p, v) })),
        symbol: "circle",
        symbolSize: 6,
        lineStyle: { color: rowColor(row), width: 1.8 },
        itemStyle: { color: rowColor(row) },
        endLabel: { show: true, formatter: rowLabel(row), fontSize: 11, color: "#374151" },
        z: 2,
      })),
    ],
  };

  const option: EChartsOption = {
    ...interactiveBits(),
    grid: { left: 64, right: 48, top: 32, bottom: 48 },
    tooltip: {
      trigger: "item",
      formatter: (raw: unknown) => {
        const p = raw as { data?: { tip?: string } };
        return p.data?.tip || "";
      },
    },
    xAxis: { type: "value", name: "TPOT (ms) — lower is better", nameLocation: "middle", nameGap: 28, ...AXIS },
    yAxis: { type: "value", name: "decode throughput (tok/s per GPU) — higher is better", nameLocation: "middle", nameGap: 48, ...AXIS },
    series: [
      ...(frontPts.length > 1 ? [{
        type: "line" as const,
        data: frontPts.map((p) => [p.x, p.y]),
        showSymbol: false,
        lineStyle: { color: "rgba(16,185,129,0.45)", width: 1.5, type: "dashed" as const },
        tooltip: { show: false },
        silent: true,
      }] : []),
      ...pts.map((p) => ({
        type: "scatter" as const,
        name: p.label,
        data: [{
          value: [p.x, p.y],
          tip: `${p.label}: ${p.x.toFixed(2)} ms, ${formatNumber(Math.round(p.y))} tok/s per GPU${
            p.ratio ? ` · ×${p.ratio.toFixed(2)} vs base` : p.isBase ? " · baseline" : ""
          }`,
        }],
        symbol: p.onFront ? "diamond" : "circle",
        symbolSize: 12,
        itemStyle: { color: p.color },
        label: {
          show: true,
          formatter: p.label + (p.ratio ? `  ×${p.ratio.toFixed(2)}` : p.isBase ? "  (base)" : ""),
          fontSize: 11,
          fontWeight: 600 as const,
          color: "#374151",
          position: "right" as const,
        },
      })),
    ],
  };

  return (
    <section className="card p-5 mt-4">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h3 className="font-semibold text-gray-900">
          Trade-off
          <span className="text-xs text-gray-400 font-normal ml-2">
            latency vs throughput
          </span>
        </h3>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <label className="text-gray-500" htmlFor="ic-source">source</label>
          <select
            id="ic-source"
            className="px-2 py-1 border border-gray-300 rounded"
            value={source}
            onChange={(e) => onSource(e.target.value)}
          >
            <option value="sol">SOL</option>
            <option value="measured">Measured</option>
          </select>
          <label className="text-gray-500 ml-1" htmlFor="ic-scatter-seq">seq</label>
          <select
            id="ic-scatter-seq"
            className="px-2 py-1 border border-gray-300 rounded"
            value={seqVal}
            onChange={(e) => setSeq(Number(e.target.value))}
          >
            {SWEEP_SEQ_LENS.map((s) => (
              <option key={s} value={s}>{fmtSeqLabel(s)}</option>
            ))}
          </select>
          <label className="text-gray-500 ml-1" htmlFor="ic-scatter-batch">batch</label>
          <select
            id="ic-scatter-batch"
            className="px-2 py-1 border border-gray-300 rounded"
            value={batch}
            onChange={(e) => setBatch(e.target.value)}
          >
            {BATCH_OPTS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
          {curves ? null : <>
          <label className="text-gray-500 ml-1" htmlFor="ic-scatter-baseline">baseline</label>
          <select
            id="ic-scatter-baseline"
            className="px-2 py-1 border border-gray-300 rounded"
            value={baseKey}
            onChange={(e) => setBaseIdx(Number(e.target.value))}
          >
            {hits.map((h, i) => (
              <option key={`${h.body.model_id}-${i}`} value={i}>
                {labels[i] || `#${i + 1}`}
              </option>
            ))}
          </select>
          </>}
        </div>
      </div>
      <div id="ic-chart-scatter">
        <div className="text-11 text-gray-500 mb-2">{head}</div>
        {failed ? (
          <div className="text-xs text-red-600 py-6">Trade-off load failed: {failed}</div>
        ) : !answered.length ? (
          <div className="text-xs text-gray-400 py-6">No row has an answer to compare.</div>
        ) : !data ? (
          <div className="text-xs text-gray-400 py-6">Loading…</div>
        ) : curves ? (
          rowLines.length
            ? <HaChart option={curvesOption} height={380} />
            : <div className="text-xs text-gray-400 py-6">No row serves any batch at this seq. Pick a smaller seq.</div>
        ) : !pts.length ? (
          <div className="text-xs text-gray-400 py-6">
            No group fits at this (seq, batch) — every group OOM'd here. Pick a smaller seq/batch.
          </div>
        ) : (
          <HaChart option={option} height={320} />
        )}
      </div>
    </section>
  );
}
