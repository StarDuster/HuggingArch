import { useEffect, useState } from "react";
import { fetchJson } from "../api";
import { fmtTime, formatBytes, formatGiB, formatFlops, formatNumber } from "../format";
import { BenchJobsExport } from "./BenchExport";
import { ComputeGraph, type GraphTarget } from "./ComputeGraph";
import { PerOpTable } from "./PhaseTable";
import type { EncodeResult, EncodeStack, Media } from "./ResultSummary";

/** One axis of a media input, as `/inference/media-inputs` declares it: the
 *  items per request (`items`), or one item's extent — fixed by the config,
 *  or a whole number of `least` (an image's resize factor, one audio frame). */
export type MediaAxis = { name: string; items: boolean; fixed: number | null; least: number | null };

export type MediaInput = {
  name: string;
  kind: string;
  stack: string | null;
  axes: MediaAxis[];
  per_item_of: string | null;
  /** Why the engine does not walk this input; a request states 0 items. */
  unmodeled: string | null;
};

export type MediaInputs = {
  model: string;
  inputs: MediaInput[];
  encoders: Array<{ stack: string; aux_kind: string | null; inputs: string[] }>;
};

/** The media form as typed: per input, whether the request carries it and,
 *  per group of items of one size, each free axis's text. */
export type MediaDraft = Record<string, { on: boolean; groups: Array<Record<string, string>> }>;

/** The model's media inputs (`/inference/media-inputs`); null until answered
 *  or for a model the endpoint cannot resolve. */
export function useMediaInputs(modelId: string): MediaInputs | null {
  const [spec, setSpec] = useState<MediaInputs | null>(null);
  useEffect(() => {
    setSpec(null);
    if (!modelId) return;
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => {
      fetchJson<MediaInputs>(
        `/inference/media-inputs?model_id=${encodeURIComponent(modelId)}`,
        { signal: ctrl.signal },
      )
        .then((j) => { if (!ctrl.signal.aborted) setSpec(j); })
        .catch(() => { if (!ctrl.signal.aborted) setSpec(null); });
    }, 300);
    return () => {
      window.clearTimeout(timer);
      ctrl.abort();
    };
  }, [modelId]);
  return spec;
}

/** The request's `media` from the form: every input the request carries,
 *  one `{axis: value}` per group of items of one size (a list when there are
 *  several), each free axis a whole number; an input whose items belong to a
 *  carried one (`per_item_of`) is stated with no item unless the form gives
 *  some, as the estimate requires. An error names what is missing. */
export function requestMedia(spec: MediaInputs | null, draft: MediaDraft
): { media: Media } | { error: string } {
  const media: Media = {};
  if (!spec) return { media };
  for (const input of spec.inputs) {
    const entry = draft[input.name];
    if (input.unmodeled !== null || !entry?.on) continue;
    const groups: Record<string, number>[] = [];
    for (const row of entry.groups) {
      const given: Record<string, number> = {};
      for (const axis of input.axes) {
        if (axis.fixed !== null) continue;
        const text = (row[axis.name] ?? "").trim();
        const n = Number(text);
        if (!text || !Number.isInteger(n) || n < 0) {
          return { error: `${input.name}: give ${axis.name} as a whole number.` };
        }
        given[axis.name] = n;
      }
      groups.push(given);
    }
    if (!groups.length) continue;
    media[input.name] = groups[0];
  }
  for (const input of spec.inputs) {
    if (input.per_item_of && media[input.per_item_of] && !media[input.name]) {
      media[input.name] = { [input.axes[0].name]: 0 };
    }
  }
  return { media };
}

/** The encoder tab's inputs: the media each request carries, at the
 *  processor's output size (or converted from a raw image size by the
 *  model's own processor, `/inference/processed-media`), one size per input
 *  size each, and the requests one encode step runs. */
export function MediaForm({ spec, modelId, draft, onDraft, batch, onBatch, onEstimate, status }: {
  spec: MediaInputs;
  modelId: string;
  draft: MediaDraft;
  onDraft: (d: MediaDraft) => void;
  batch: string;
  onBatch: (v: string) => void;
  onEstimate: () => void;
  status: string;
}) {
  const [raw, setRaw] = useState<Record<string, { height: string; width: string; note: string }>>({});
  const entryOf = (name: string) => draft[name] ?? { on: false, groups: [{}] };
  const setEntry = (name: string, next: { on: boolean; groups: Array<Record<string, string>> }) =>
    onDraft({ ...draft, [name]: next });
  const setCell = (name: string, g: number, axis: string, value: string) => {
    const cur = entryOf(name);
    setEntry(name, { ...cur, groups: cur.groups.map((row, i) => (i === g ? { ...row, [axis]: value } : row)) });
  };
  const convert = async (input: MediaInput, g: number) => {
    const r = raw[`${input.name}#${g}`] ?? { height: "", width: "", note: "" };
    const items = entryOf(input.name).groups[g]?.[input.axes[0].name] ?? "";
    const key = `${input.name}#${g}`;
    try {
      const body = await fetchJson<{ media: Media; rules: Record<string, { processor: string; source: string }> }>(
        "/inference/processed-media", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_id: modelId,
            raw: { [input.name]: { [input.axes[0].name]: Number(items || 1), height: Number(r.height), width: Number(r.width) } },
          }),
        });
      const processed = body.media[input.name] as Record<string, number>;
      const cur = entryOf(input.name);
      setEntry(input.name, {
        ...cur,
        groups: cur.groups.map((row, i) => (i === g
          ? Object.fromEntries(Object.entries({ ...row, ...processed }).map(([k, v]) => [k, String(v)]))
          : row)),
      });
      const rule = body.rules[input.name];
      setRaw({ ...raw, [key]: { ...r, note: rule ? `${rule.processor} (${rule.source})` : "" } });
    } catch (e) {
      setRaw({ ...raw, [key]: { ...r, note: e instanceof Error ? e.message : String(e) } });
    }
  };
  return (
    <section id="infer-encode-media" className="card px-4 sm:px-6 py-4 mb-6">
      <h3 className="font-semibold text-gray-900 mb-1">Media per request</h3>
      <p className="text-xs text-gray-500 mb-3">
        Sizes are what the processor hands the encoder (after its resize); a raw image size can be converted by the
        model's own processor. One size per input (items of different sizes in one request are not modelled yet). The
        encoders run in their own encode step;
        the language model's prompt is its own text workload and does not include these.
      </p>
      <div className="space-y-3">
        {spec.inputs.map((input) => {
          const parentOn = input.per_item_of ? !!draft[input.per_item_of]?.on : false;
          if (input.unmodeled !== null) {
            return (
              <div key={input.name} id={`infer-encode-input-${input.name}`} className="text-xs text-gray-500">
                <span className="font-mono text-gray-700">{input.name}</span>{" "}
                ({input.axes.map((a) => a.name).join(", ")}) — not modelled; a request carries none
                {parentOn ? ` (${input.axes[0].name} = 0)` : ""}. <span className="italic">{input.unmodeled}</span>
              </div>
            );
          }
          const entry = entryOf(input.name);
          const sides = input.axes.filter((a) => !a.items && a.fixed === null);
          const convertible = input.kind === "image" && sides.length > 0;
          return (
            <div key={input.name} id={`infer-encode-input-${input.name}`} className="text-sm">
              <label className="inline-flex items-center gap-2">
                <input type="checkbox" checked={entry.on} onChange={(e) => setEntry(input.name, { ...entry, on: e.target.checked })} />
                <span className="font-mono">{input.name}</span>
                <span className="text-xs text-gray-400">
                  {input.kind}{input.stack ? ` → ${input.stack}` : ""}{input.per_item_of ? ` · per item of ${input.per_item_of}` : ""}
                </span>
              </label>
              {entry.groups.map((row, g) => {
                const key = `${input.name}#${g}`;
                const r = raw[key] ?? { height: "", width: "", note: "" };
                return (
                  <div key={g} className="flex flex-wrap items-center gap-3 mt-1 ml-6">
                    {input.axes.map((axis) => (
                      axis.fixed !== null ? (
                        <span key={axis.name} className="text-xs text-gray-500">
                          {axis.name} = <span className="font-mono">{axis.fixed}</span> (fixed by the config)
                        </span>
                      ) : (
                        <label key={axis.name} className="inline-flex items-center gap-1 text-xs text-gray-600">
                          {axis.name}
                          <input
                            type="number"
                            className="w-24 px-2 py-1 border border-gray-300 rounded"
                            disabled={!entry.on}
                            min={axis.items ? 0 : (axis.least ?? 1)}
                            step={axis.least ?? 1}
                            placeholder={axis.items ? "items" : axis.least && axis.least > 1 ? `× ${axis.least}` : ""}
                            title={axis.items ? "items per request"
                              : axis.least && axis.least > 1 ? `a whole number of ${axis.least} (the processor's resize factor)` : undefined}
                            value={row[axis.name] ?? ""}
                            onChange={(e) => setCell(input.name, g, axis.name, e.target.value)}
                          />
                        </label>
                      )
                    ))}
                    {convertible && entry.on && (
                      <span className="inline-flex items-center gap-1 text-xs text-gray-500">
                        raw
                        <input type="number" min={1} className="w-20 px-2 py-1 border border-gray-300 rounded" placeholder="height"
                          value={r.height} onChange={(e) => setRaw({ ...raw, [key]: { ...r, height: e.target.value } })} />
                        ×
                        <input type="number" min={1} className="w-20 px-2 py-1 border border-gray-300 rounded" placeholder="width"
                          value={r.width} onChange={(e) => setRaw({ ...raw, [key]: { ...r, width: e.target.value } })} />
                        <button type="button" className="px-2 py-1 border border-gray-300 rounded hover:bg-gray-50"
                          onClick={() => void convert(input, g)}>convert</button>
                        {r.note ? <span className="italic">{r.note}</span> : null}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-3 mt-4 text-sm">
        <label className="inline-flex items-center gap-2 text-xs text-gray-600">
          Requests per encode step
          <input
            id="infer-encode-batch"
            type="number"
            min={1}
            className="w-24 px-2 py-1 border border-gray-300 rounded"
            placeholder="prefill batch"
            value={batch}
            onChange={(e) => onBatch(e.target.value)}
          />
        </label>
        <button id="infer-encode-estimate" type="button" className="px-4 py-1.5 text-sm font-medium bg-amber-500 text-white rounded-lg hover:bg-amber-600" onClick={onEstimate}>
          Estimate
        </button>
        {status ? <span className="text-xs text-gray-500">{status}</span> : null}
      </div>
    </section>
  );
}

function Metric({ label, value, id }: { label: string; value: string; id?: string }) {
  return (
    <div id={id} className="flex justify-between gap-4 text-sm border-b border-gray-100 py-1">
      <span className="text-gray-600">{label}</span>
      <span className="font-mono text-gray-900">{value}</span>
    </div>
  );
}

const dash = (v: number | null | undefined, f: (n: number) => string) =>
  v === null || v === undefined ? "—" : f(v);

/** The encoder's own capacity: its weights, its encode step's activation
 *  peak, and the output cache the framework keeps — none of it in the
 *  language model's memory picture. */
function CapacityCard({ s, encode }: { s: EncodeStack; encode: EncodeResult }) {
  const c = encode.output_cache;
  return (
    <div id={`infer-encode-capacity-${s.stack}`} className="mt-4">
      <div className="text-xs font-medium text-gray-500 mb-1">Encoder memory per rank</div>
      <div className="grid sm:grid-cols-2 gap-x-8">
        <div>
          <Metric id={`infer-encode-weights-${s.stack}`} label="Weights"
            value={s.reruns_weights_of ? `those of ${s.reruns_weights_of}` : formatGiB(s.weight_bytes_per_rank)} />
          <Metric
            id={`infer-encode-act-peak-${s.stack}`}
            label={`Activation peak${s.activation_peak?.worst_block ? ` (${s.activation_peak.worst_block})` : ""}`}
            value={dash(s.activation_peak?.bytes_per_rank, formatGiB)}
          />
        </div>
        <div id="infer-encode-output-cache">
          <Metric label="Output cache (all encoders)" value={dash(c.bytes_per_rank, formatGiB)} />
          <Metric label="Cached tokens" value={c.tokens === null ? "no token budget" : formatNumber(c.tokens)} />
          <Metric label="Scheduler token budget" value={dash(c.scheduler_tokens, formatNumber)} />
          <Metric label="Tokens of the largest item" value={formatNumber(c.max_tokens_per_item)} />
          <Metric label="Bytes per cached token" value={formatGiB(c.bytes_per_token)} />
        </div>
      </div>
    </div>
  );
}

function StackCard({ s }: { s: EncodeStack }) {
  return (
    <div id={`infer-encode-stack-${s.stack}`} className="grid sm:grid-cols-2 gap-x-8">
      <div>
        <Metric label="Items in the step" value={formatNumber(s.items)} />
        {Object.entries(s.media_rows).map(([name, r]) => (
          <div key={name}>
            <Metric label={`${name}: segments`} value={formatNumber(r.segments)} />
            <Metric label={`${name}: rows per segment (computed)`} value={formatNumber(r.rows_per_segment)} />
            <Metric label={`${name}: rows per segment (the item's own)`} value={formatNumber(r.valid_rows_per_segment)} />
          </div>
        ))}
        {Object.entries(s.handoff_tokens_per_request).map(([name, n]) => (
          <Metric key={name} label={`${name}: prompt positions per request`} value={formatNumber(n)} />
        ))}
      </div>
      <div>
        <Metric label="FLOPs per rank" value={dash(s.flops_per_rank, formatFlops)} />
        <Metric label="Bytes per rank" value={dash(s.bytes_per_rank, formatBytes)} />
        <Metric label="t_compute" value={dash(s.t_compute, fmtTime)} />
        <Metric label="t_memory" value={dash(s.t_memory, fmtTime)} />
        <Metric label="t_comm" value={dash(s.t_comm, fmtTime)} />
        <Metric id={`infer-encode-tsol-${s.stack}`} label="t_sol (step)" value={dash(s.t_sol, fmtTime)} />
        <Metric label="t_measured (step)" value={dash(s.t_measured, fmtTime)} />
        <Metric label="Time per item" value={dash(s.t_sol_per_item, fmtTime)} />
        <Metric label="Items per second" value={dash(s.items_per_s, (n) => formatNumber(Math.round(n)))} />
      </div>
    </div>
  );
}

/** The encode phase of an estimate: each encoder's own step, never added to
 *  the language model's prefill / decode. */
export function EncoderResults({ encode, target }: { encode: EncodeResult; target: GraphTarget }) {
  const [pick, setPick] = useState(encode.stacks[0]?.stack ?? "");
  const s = encode.stacks.find((x) => x.stack === pick) ?? encode.stacks[0];
  if (!s) return null;
  const tabCls = (k: string) =>
    "px-3 py-1.5 text-xs font-medium " + (s.stack === k ? "bg-amber-500 text-white" : "bg-white text-gray-600 hover:bg-gray-50");
  const stepTarget: GraphTarget = {
    ...target,
    encode: { stack: s.stack, media: encode.media as Record<string, Record<string, number>>, batch: encode.batch },
  };
  return (
    <div id="infer-encode-results" className="space-y-6">
      <section className="card p-5 mb-6">
        <div className="flex items-baseline justify-between mb-3 flex-wrap gap-2">
          <h3 className="font-semibold text-gray-900">
            Encode step <span className="text-xs text-gray-400 font-normal">per rank · {encode.batch} request(s)</span>
          </h3>
          {encode.stacks.length > 1 && (
            <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden" role="tablist" aria-label="Encoder">
              {encode.stacks.map((x) => (
                <button key={x.stack} id={`infer-encode-tab-${x.stack}`} type="button" role="tab"
                  className={tabCls(x.stack)} aria-selected={s.stack === x.stack} onClick={() => setPick(x.stack)}>
                  {x.stack}{x.aux_kind && x.aux_kind !== x.stack ? ` (${x.aux_kind})` : ""}
                </button>
              ))}
            </div>
          )}
          {/* The encode step as measurable jobs. One export for the step:
           *  the planner plans every encoder the media names, which is what
           *  this card's tabs are showing. */}
          <BenchJobsExport target={stepTarget} phase="encode" />
        </div>
        {s.items === 0 ? (
          <p className="text-sm text-gray-500">This request carries no media for {s.stack}; it runs no item.</p>
        ) : <StackCard s={s} />}
        <CapacityCard s={s} encode={encode} />
        {encode.subtotal && (
          <div id="infer-encode-subtotal" className="mt-4">
            <div className="text-xs font-medium text-gray-500 mb-1">Encoders only, all encoders together</div>
            <Metric label="FLOPs per rank" value={dash(encode.subtotal.flops_per_rank, formatFlops)} />
            <Metric label="t_sol" value={dash(encode.subtotal.t_sol, fmtTime)} />
          </div>
        )}
        <div className="text-11 text-amber-700 mt-3">
          ⚠ <span className="font-medium">No-overlap roofline.</span> t_sol sums per-leaf t_op = max(t_compute, t_memory, t_comm, t_launch) — kernels assumed serial, no compute/comm overlap.
        </div>
      </section>
      {s.items > 0 && (
        <section className="card p-5 mb-6">
          <PerOpTable
            key={s.stack}
            rows={s.per_op_timing}
            moduleRows={s.flops_by_module}
            total={s.op_table_total}
            byDeclaration={s.per_op_timing_by_declaration ?? []}
            target={stepTarget}
            phase="encode"
            id="infer-encode-breakdown"
            measured={s.t_measured ?? null}
          />
        </section>
      )}
      {s.items > 0 && (
        <ComputeGraph key={s.stack} target={stepTarget} />
      )}
    </div>
  );
}
