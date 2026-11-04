import type { ReactNode } from "react";
import { deployFromPayload, phaseTokPerSec, SWEEP_SEQ_LENS, type SweepHit } from "../compare";
import { DataTable, type ColumnDef } from "../DataTable";
import { formatGiB, formatNumber, fmtSeqLabel } from "../format";
import { BOUND_PILL } from "../bound";
import type { Estimate, PhaseMetrics, RowBound } from "./ResultSummary";

const BASICS_TIPS = {
  deploy:
    "Deployment this row was computed under — GPU · TP/EP/CP/DP · cluster — from the payload actually sent. If a row OOMs, this config caused it.",
  attn: "Attention variant + hybrid marker. Same KV/token can scale differently with seq: SWA / DSA top-K cap reads, dense reads the full cache.",
  weights: "Weight bytes one GPU holds under this deployment, as the model ships.",
  maxTok:
    "Max KV-tokens per GPU after weights + activations. Per-request KV is the \"Per-request KV vs context\" curve.",
};

function firstMaxEstimate(h: SweepHit): { est: Estimate | null; anyMissing: boolean } {
  const row = h.result?.results_by_batch?.max || [];
  let est: Estimate | null = null;
  let anyMissing = !row.length;
  for (const cell of row) {
    if (!cell) {
      anyMissing = true;
      continue;
    }
    if (!est) est = cell;
  }
  if (h.error) anyMissing = true;
  return { est, anyMissing };
}

function Dash({ tip }: { tip: string }) {
  return (
    <span className="text-gray-300" title={tip}>
      —
    </span>
  );
}

const FETCH_FAIL_TIP =
  "Backend request failed (HTTP error / network / 120s timeout). Retry, or check the backend logs.";
const OOM_FALLBACK =
  "Capacity check returned no admissible batch at this seq (note absent — likely weights + KV exceed budget).";

type SplitKey = RowBound;
type SplitPart = { key: SplitKey; label: string; pct: number };
type DecSplit = { dominant: SplitPart; parts: SplitPart[] };

type SeqCell = {
  max_batch: number;
  max_tokens: number | null;
  pre_tps: number | null;
  dec_tps: number | null;
  act_peak: number | null;
  dec_split: DecSplit | null;
  oom: boolean;
  note: string;
};


/** The decode step's time by the term that set each op's time, as the
 *  backend split it: each part's share of `t_sol_s` (`t_sol_by_bound_share`). */
function decodeSplit(dec?: PhaseMetrics): DecSplit | null {
  const share = dec?.t_sol_by_bound_share;
  if (!share) return null;
  const parts: SplitPart[] = [
    { key: "compute", label: "Comp", pct: share.compute },
    { key: "memory", label: "Mem", pct: share.memory },
    { key: "comm", label: "Comm", pct: share.comm },
    { key: "launch", label: "Launch", pct: share.launch },
  ];
  parts.sort((a, b) => b.pct - a.pct);
  return { dominant: parts[0], parts };
}

function seqCell(est: Estimate | undefined): SeqCell | null {
  if (!est) return null;
  const maxB = est.max_decode_batch_per_gpu || 0;
  const dec = est.phases?.decode;
  return {
    max_batch: maxB,
    max_tokens: est.max_tokens_per_gpu ?? null,
    pre_tps: phaseTokPerSec(est.phases?.prefill),
    dec_tps: phaseTokPerSec(dec),
    act_peak: est.activation_peak?.peak_activation_bytes_per_rank ?? null,
    dec_split: decodeSplit(dec),
    oom: maxB <= 0,
    note: est.phases?.prefill?.note || est.phases?.prefill?.time_unpublished_reason || "",
  };
}

function PivotTable<T>({
  label,
  labelTip,
  hits,
  labels,
  rows,
  getCell,
  fmt,
  cellTip,
  sortBy = (v) => (typeof v === "number" ? v : null),
}: {
  label: string;
  labelTip: string;
  hits: SweepHit[];
  labels: string[];
  rows: Array<Array<SeqCell | null>>;
  getCell: (c: SeqCell) => T | null | undefined;
  fmt: (v: T) => ReactNode;
  cellTip: (name: string, seq: number, c: SeqCell, v: T) => string;
  /** What a cell sorts by; defaults to the value itself when it is a number. */
  sortBy?: (v: T) => number | string | null;
}) {
  type PivotRow = { h: SweepHit; i: number; name: string };
  const cols: ColumnDef<PivotRow>[] = [
    {
      key: "model",
      header: "Model",
      className: "text-gray-700 font-mono",
      render: (r) => r.name,
      sortValue: (r) => r.name,
    },
    ...SWEEP_SEQ_LENS.map(
      (s, si): ColumnDef<PivotRow> => ({
        key: `seq-${s}`,
        header: (
          <span title={`avg input tokens per request = ${formatNumber(s)} (the seq this column was computed at)`}>
            {fmtSeqLabel(s)}
          </span>
        ),
        align: "right",
        className: "font-mono",
        render: (r) => {
          const c = rows[r.i]?.[si] ?? null;
          if (!c) return <Dash tip={FETCH_FAIL_TIP} />;
          const v = getCell(c);
          if (c.oom || v == null || v === (0 as T)) return <Dash tip={c.note || OOM_FALLBACK} />;
          return <span title={cellTip(r.name, s, c, v)}>{fmt(v)}</span>;
        },
        sortValue: (r) => {
          const c = rows[r.i]?.[si] ?? null;
          const v = c ? getCell(c) : null;
          if (!c || c.oom || v == null || v === (0 as T)) return null;
          return sortBy(v);
        },
      }),
    ),
  ];
  return (
    <div>
      <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2" title={labelTip}>
        {label}
      </div>
      <DataTable
        columns={cols}
        rows={hits.map((h, i) => ({ h, i, name: labels[i] || `#${i + 1}` }))}
        rowKey={(r) => `${r.h.body.model_id}-${r.i}-${label}`}
        framed={false}
        tableClassName="min-w-[760px]"
      />
    </div>
  );
}

function FmtSplit({ s }: { s: DecSplit }) {
  const dom = s.dominant;
  const pctTxt = `${Math.round(dom.pct * 100)}%`;
  const breakdown = s.parts.map((p) => `${p.label} ${Math.round(p.pct * 100)}%`).join(" / ");
  return (
    <span title={`Decode-step time split: ${breakdown}`}>
      <span className={`inline-block px-1.5 py-0.5 text-10 font-medium ${BOUND_PILL[dom.key]} rounded`}>
        {dom.label}
      </span>{" "}
      {pctTxt}
    </span>
  );
}

type BasicsRow = {
  h: SweepHit;
  label: string;
  emptyTip: string;
  dep: ReturnType<typeof deployFromPayload>;
  attn: string | null;
  hybrid: boolean;
  weights: number | undefined;
  maxTok: number | null | undefined;
};

export function CompareSummary({ hits, labels }: { hits: SweepHit[]; labels: string[] }) {
  if (!hits.length) return null;

  const rows = hits.map((h) => {
    const arr = h.result?.results_by_batch?.max || [];
    return SWEEP_SEQ_LENS.map((_, si) => seqCell(arr[si]));
  });

  const basicsRows: BasicsRow[] = hits.map((h, i) => {
    const { est, anyMissing } = firstMaxEstimate(h);
    return {
      h,
      label: labels[i] || `#${i + 1}`,
      emptyTip: anyMissing
        ? "no successful estimate at any seq — capacity infeasible under this config"
        : "Backend returned a response but this field was null.",
      dep: deployFromPayload(h),
      attn: est?.kv_cache?.attention_variant || est?.kv_cache?.attention_type || null,
      hybrid: !!est?.kv_cache?.is_hybrid_attention,
      weights: est?.memory?.weight_bytes_per_gpu,
      maxTok: est?.max_tokens_per_gpu,
    };
  });

  const basicsCols: ColumnDef<BasicsRow>[] = [
    {
      key: "model",
      header: "Model",
      className: "text-gray-700 font-mono",
      render: (r) => r.label,
      sortValue: (r) => r.label,
    },
    {
      key: "deployment",
      header: <span title={BASICS_TIPS.deploy}>Deployment</span>,
      sortValue: (r) => `${r.dep.gpu} · EP${r.dep.ep}/DP${r.dep.dpAttn}`,
      render: (r) => (
        <span
          className="font-mono text-11 text-gray-600"
          title={`Sent to the backend for ${r.h.body.model_id}: ${r.dep.label}`}
        >
          {r.dep.gpu} ·{" "}
          <span className={r.dep.ep > 1 ? "text-emerald-700" : "text-amber-700 font-medium"}>
            EP{r.dep.ep}
          </span>
          /DP{r.dep.dpAttn}
        </span>
      ),
    },
    {
      key: "attn",
      header: <span title={BASICS_TIPS.attn}>Attn</span>,
      sortValue: (r) => r.attn,
      className: "font-mono",
      render: (r) =>
        r.attn ? (
          <>
            <span
              title={
                r.hybrid
                  ? `Attention variant: ${r.attn} (hybrid — mixed attention classes per block; see KV-read chart for the per-step read footprint).`
                  : `Attention variant: ${r.attn}.`
              }
            >
              {r.attn}
            </span>
            {r.hybrid ? (
              <span className="ml-1 px-1.5 py-0.5 text-10 font-medium text-amber-700 bg-amber-50 border border-amber-200 rounded">
                hybrid
              </span>
            ) : null}
          </>
        ) : (
          <Dash tip={r.emptyTip} />
        ),
    },
    {
      key: "weights",
      header: <span title={BASICS_TIPS.weights}>Weights / GPU</span>,
      sortValue: (r) => r.weights,
      align: "right",
      className: "font-mono",
      render: (r) =>
        r.weights != null ? (
          <span title={`Per-GPU weight bytes for ${r.label} as the model ships, after TP/EP sharding.`}>
            {formatGiB(r.weights)}
          </span>
        ) : (
          <Dash tip={r.emptyTip} />
        ),
    },
    {
      key: "maxTok",
      header: <span title={BASICS_TIPS.maxTok}>Max tokens / GPU</span>,
      sortValue: (r) => r.maxTok,
      align: "right",
      className: "font-mono",
      render: (r) =>
        r.maxTok != null ? (
          <span title={`Max KV-tokens that fit per GPU for ${r.label} after weights + activations (= max_batch × seq budget).`}>
            {formatNumber(r.maxTok)}
          </span>
        ) : (
          <Dash tip={r.emptyTip} />
        ),
    },
  ];

  return (
    <section className="card p-5 mt-4">
      <h3 className="font-semibold text-gray-900 mb-3">
        Summary
        <span className="text-xs text-gray-400 font-normal ml-2">per-group basics + per-seq sweep</span>
      </h3>
      <div id="ic-summary" className="space-y-6">
        <div>
          <div
            className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2"
            title="Model + parallel invariants — identical across the 9 seq points."
          >
            Model basics <span className="text-gray-400 normal-case font-normal ml-1">(seq-invariant)</span>
          </div>
          <DataTable
            columns={basicsCols}
            rows={basicsRows}
            rowKey={(r, i) => `${r.h.body.model_id}-${i}`}
            framed={false}
          />
        </div>
        <PivotTable
          label="Max batch / DP rank"
          labelTip="Max concurrent sequences one DP rank (the TP × PP × CP group serving the same requests) can admit at this seq, as the capacity solver found it."
          hits={hits}
          labels={labels}
          rows={rows}
          getCell={(c) => c.max_batch}
          fmt={(v) => formatNumber(v)}
          cellTip={(name, seq, c, v) =>
            `${name} @ seq=${fmtSeqLabel(seq)} → ${formatNumber(v)} sequences/GPU (${c.max_tokens == null ? "the cache does not grow with length" : `${formatNumber(c.max_tokens)} tokens of KV per GPU`}).`
          }
        />
        <PivotTable
          label="Prefill tok/s per GPU @ max batch"
          labelTip="Aggregate prefill throughput across the whole cluster at the capacity-admitted batch (effective_tokens_per_step ÷ t_sol, summed over GPUs)."
          hits={hits}
          labels={labels}
          rows={rows}
          getCell={(c) => c.pre_tps}
          fmt={(v) => formatNumber(Math.round(v))}
          cellTip={(name, seq, _c, v) =>
            `${name} @ seq=${fmtSeqLabel(seq)} → ${formatNumber(Math.round(v))} tok/s per GPU prefill (at the capacity-admitted batch).`
          }
        />
        <PivotTable
          label="Decode tok/s per GPU @ max batch"
          labelTip="Aggregate decode throughput across the whole cluster at the capacity-admitted batch (1 token per step × batch ÷ t_sol, summed over GPUs)."
          hits={hits}
          labels={labels}
          rows={rows}
          getCell={(c) => c.dec_tps}
          fmt={(v) => formatNumber(Math.round(v))}
          cellTip={(name, seq, _c, v) =>
            `${name} @ seq=${fmtSeqLabel(seq)} → ${formatNumber(Math.round(v))} tok/s per GPU decode (at the capacity-admitted batch).`
          }
        />
        <PivotTable
          label="Activation peak / GPU"
          labelTip="Peak activation HBM / rank in decode's worst step. OOMs when weights + KV + act_peak > HBM, even if KV alone fits."
          hits={hits}
          labels={labels}
          rows={rows}
          getCell={(c) => c.act_peak}
          fmt={(v) => formatGiB(v)}
          cellTip={(name, seq, _c, v) =>
            `${name} @ seq=${fmtSeqLabel(seq)} → ${formatGiB(v)} activation peak per rank (decode worst-step live set).`
          }
        />
        <PivotTable
          label="Decode bottleneck"
          labelTip="Dominant of compute / memory / comm at the capacity batch. Memory at low batch, compute at high, comm across NVLink/IB."
          hits={hits}
          labels={labels}
          rows={rows}
          getCell={(c) => c.dec_split}
          fmt={(s) => <FmtSplit s={s} />}
          sortBy={(s) => `${s.dominant.label} ${Math.round(s.dominant.pct * 100)}`}
          cellTip={(name, seq, _c, s) =>
            `${name} @ seq=${fmtSeqLabel(seq)} → decode is ${s.dominant.label}-bound (${Math.round(s.dominant.pct * 100)}% of t_step).`
          }
        />
      </div>
    </section>
  );
}
