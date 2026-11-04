import type { EChartsOption } from "echarts";
import { AXIS, HaChart, interactiveBits } from "../charts/HaChart";
import { phaseSeconds, phaseTokPerSec, SWEEP_COLORS, SWEEP_SEQ_LENS, type SweepHit } from "../compare";
import type { Estimate, PhaseMetrics } from "./ResultSummary";

const SMALL_B = [
  { key: "b1", label: "decode B = 1", B: 1 },
  { key: "b16", label: "decode B = 16", B: 16 },
  { key: "b64", label: "decode B = 64", B: 64 },
] as const;

type Series = { key: string; label: string; color: string; points: Array<{ x: number; v: number }> };
type Phase = "prefill" | "decode";

function tpotMs(phase: PhaseMetrics | undefined, source: string): number | null {
  const t = phaseSeconds(phase, source);
  return t != null ? t * 1000 : null;
}

function Plot({ series, compact }: { series: Series[]; compact?: boolean }) {
  const pts = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.v) && p.v > 0 && p.x > 0);
  if (!pts.length) {
    return <div className="text-xs text-gray-400 py-6">no curve</div>;
  }
  const option: EChartsOption = {
    color: series.map((s) => s.color),
    ...interactiveBits(),
    grid: { left: 52, right: 16, top: 28, bottom: 40 },
    tooltip: { trigger: "axis" },
    legend: { type: "scroll", bottom: 0, textStyle: { fontSize: 10 } },
    xAxis: { type: "log", name: "seq (log)", nameLocation: "middle", nameGap: 22, ...AXIS },
    yAxis: { type: "value", ...AXIS },
    series: series.map((s) => ({
      type: "line" as const,
      name: s.label,
      // A context with no value (the measured track where nothing was
      // measured, a batch past the cap) is a null: the line breaks there.
      data: s.points.filter((p) => p.x > 0).map((p) => [p.x, Number.isFinite(p.v) && p.v > 0 ? p.v : null]),
      connectNulls: false,
      showSymbol: true,
      symbolSize: 4,
      lineStyle: { width: 1.6 },
    })),
  };
  return <HaChart option={option} height={compact ? 160 : 220} />;
}

function Chart({
  id,
  title,
  subtitle,
  series,
  compact,
  nested,
}: {
  id?: string;
  title: string;
  subtitle: string;
  series: Series[];
  compact?: boolean;
  nested?: boolean;
}) {
  const inner = (
    <div id={id} className={nested ? "mb-3" : undefined}>
      <div className="text-sm font-medium text-gray-800 mb-0">{title}</div>
      {subtitle ? <div className="text-11 text-gray-500 mb-2">{subtitle}</div> : null}
      <Plot series={series} compact={compact} />
    </div>
  );
  if (nested) return inner;
  return <section className="card p-5 mt-4">{inner}</section>;
}

function MultiBSplit({
  id,
  title,
  subtitleSmall,
  subtitleMax,
  small,
  maxSeries,
}: {
  id: string;
  title: string;
  subtitleSmall: string;
  subtitleMax: string;
  small: Array<{ key: string; label: string; series: Series[] }>;
  maxSeries: Series[];
}) {
  return (
    <section className="card p-5 mt-4">
      <div id={id}>
        <div className="text-sm font-medium text-gray-800">{title} · small batch</div>
        <div className="text-11 text-gray-500 mb-2">{subtitleSmall}</div>
        {small.map((b) => (
          <Chart
            key={b.key}
            title={b.label}
            subtitle=""
            series={b.series}
            compact
            nested
          />
        ))}
        <Chart
          title={`${title} · capacity max batch`}
          subtitle={subtitleMax}
          series={maxSeries}
          nested
        />
      </div>
    </section>
  );
}

export function CompareCharts({
  hits,
  labels,
  source = "sol",
  phase = "decode",
  onPhase,
}: {
  hits: SweepHit[];
  labels: string[];
  source?: string;
  phase?: Phase;
  onPhase?: (p: Phase) => void;
}) {
  const ok = hits
    .map((h, i) => ({ h, i }))
    .filter(({ h }) => h.result?.results_by_batch && !h.error);
  if (!ok.length) return null;

  const overlay = (
    batchKey: string,
    pick: (r?: Estimate) => number,
    clip?: { phase: Phase; B: number | null; maxLabel?: boolean },
  ): Series[] =>
    ok.map(({ h, i }) => {
      const row = h.result?.results_by_batch?.[batchKey] || [];
      const maxRow = h.result?.results_by_batch?.max || [];
      const base = labels[i] || `#${i + 1}`;
      return {
        key: `${h.body.model_id}-${i}-${batchKey}`,
        label: clip?.maxLabel ? `${base} · B = max` : base,
        color: SWEEP_COLORS[i % SWEEP_COLORS.length],
        points: SWEEP_SEQ_LENS.map((seq, si) => {
          if (clip) {
            const phMax = maxRow[si]?.phases?.[clip.phase];
            const Bmax = phMax?.batch || 0;
            const B = clip.B == null ? Bmax : clip.B;
            if (Bmax <= 0 || B < 1 || B > Bmax) return { x: seq, v: NaN };
          }
          return { x: seq, v: pick(row[si]) };
        }),
      };
    });

  const smallOf = (ph: Phase, pick: (r?: Estimate) => number) =>
    SMALL_B.map((b) => ({
      key: b.key,
      label: b.label,
      series: overlay(b.key, pick, { phase: ph, B: b.B }),
    }));

  const tabCls = (on: boolean) =>
    on
      ? "px-3 py-1.5 text-xs font-medium bg-amber-500 text-white"
      : "px-3 py-1.5 text-xs font-medium bg-white text-gray-600 hover:bg-gray-50";

  return (
    <div className="mt-4 space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden" role="tablist" aria-label="Phase">
          <button
            id="ic-phase-prefill"
            type="button"
            data-phase="prefill"
            role="tab"
            aria-selected={phase === "prefill"}
            className={tabCls(phase === "prefill")}
            onClick={() => onPhase?.("prefill")}
          >
            Prefill
          </button>
          <button
            id="ic-phase-decode"
            type="button"
            data-phase="decode"
            role="tab"
            aria-selected={phase === "decode"}
            className={tabCls(phase === "decode")}
            onClick={() => onPhase?.("decode")}
          >
            Decode
          </button>
        </div>
        <span className="text-11 text-gray-500">Prefill / decode</span>
      </div>

      {phase === "prefill" ? (
        <>
          <Chart
            id="ic-chart-maxbatch-prefill"
            title="Max batch / DP rank vs seq · prefill"
            subtitle="Activation-bound"
            series={overlay("max", (r) => r?.max_prefill_batch_per_gpu ?? r?.max_prefill_batch ?? NaN)}
          />
          <MultiBSplit
            id="ic-chart-prefill"
            title="Prefill throughput vs seq"
            subtitleSmall="tok/s per GPU at batch 1 / 16 / 64"
            subtitleMax="tok/s per GPU at max batch"
            small={smallOf("prefill", (r) => phaseTokPerSec(r?.phases?.prefill, source) ?? NaN)}
            maxSeries={overlay("max", (r) => phaseTokPerSec(r?.phases?.prefill, source) ?? NaN, { phase: "prefill", B: null, maxLabel: true })}
          />
        </>
      ) : (
        <>
          <Chart
            id="ic-chart-maxbatch-decode"
            title="Max batch / DP rank vs seq · decode"
            subtitle="KV-bound"
            series={overlay("max", (r) => r?.max_decode_batch_per_gpu ?? NaN)}
          />
          <Chart
            id="ic-chart-kvcache"
            title="Per-request KV cache vs seq"
            subtitle="Per rank"
            series={overlay("max", (r) => r?.kv_cache?.per_request_kv_bytes ?? NaN)}
          />
          <MultiBSplit
            id="ic-chart-kv-read"
            title="TPOT (ms / generated token) vs seq"
            subtitleSmall="Step time at batch 1 / 16 / 64"
            subtitleMax="Step time at max batch"
            small={smallOf("decode", (r) => tpotMs(r?.phases?.decode, source) ?? NaN)}
            maxSeries={overlay("max", (r) => tpotMs(r?.phases?.decode, source) ?? NaN, { phase: "decode", B: null, maxLabel: true })}
          />
          <MultiBSplit
            id="ic-chart-decode"
            title="Decode throughput vs seq"
            subtitleSmall="tok/s per GPU at batch 1 / 16 / 64"
            subtitleMax="tok/s per GPU at max batch"
            small={smallOf("decode", (r) => phaseTokPerSec(r?.phases?.decode, source) ?? NaN)}
            maxSeries={overlay("max", (r) => phaseTokPerSec(r?.phases?.decode, source) ?? NaN, { phase: "decode", B: null, maxLabel: true })}
          />
        </>
      )}
    </div>
  );
}
