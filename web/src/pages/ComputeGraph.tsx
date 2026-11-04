import { useEffect, useRef, useState } from "react";
import { ApiError, fetchJson } from "../api";
import type { Parallel } from "../parallel";
import { setPrecisionQuery } from "../precision";
import { textHtml, type BlockEdge, type BlockNode } from "./BlockDag";
import { GraphShell } from "./Structure";
import type { Estimate } from "./ResultSummary";

// The inference page's serving graph (comm injected, kernels fused), drawn by
// the same renderer as the Inspector: the backend sends the real edges at every
// scope (`edges`, each compound's `child_edges`) and the styles, so this page
// only fetches, picks the phase / stage, and summarises.
export type GraphResp = {
  model?: string;
  framework?: string;
  /** The release the framework binding's facts were read at. */
  framework_audited_at?: string;
  phase?: string;
  parallel?: Partial<Parallel>;
  stage?: { rank?: number; pp?: number; layers?: number[] };
  graph?: { nodes?: BlockNode[]; edges?: BlockEdge[] };
};

/** The fused compound standing for `fusedOp` — its children are the
 *  pre-fusion subgraph it replaced. */
export function findFusedCompound(nodes: BlockNode[], fusedOp: string): BlockNode | null {
  for (const n of nodes) {
    if (n.style === "fused" && (n.op === fusedOp || n.fused === fusedOp)) return n;
    if (n.children) {
      const hit = findFusedCompound(n.children, fusedOp);
      if (hit) return hit;
    }
  }
  return null;
}

/** The graph's nodes by id, at every depth — how a node a fused subgraph
 *  reads from outside it is named (its backend `label`). */
export function nodesById(nodes: BlockNode[], out: Map<string, BlockNode> = new Map()): Map<string, BlockNode> {
  for (const n of nodes) {
    if (n.id) out.set(n.id, n);
    if (n.children) nodesById(n.children, out);
  }
  return out;
}

/** Flatten a fused compound into a popover-sized DAG: the backend's labels,
 *  no inspector card extras, producers outside it as source pills named by
 *  their own node (`graph`, the whole graph by id). */
export function compactFusedSubgraph(
  compound: BlockNode,
  graph: Map<string, BlockNode>,
  absorbed: Array<{ name?: string; op?: string; shape_out?: number[] }> = [],
): { nodes: BlockNode[]; edges: BlockEdge[] } {
  const children = compound.children || [];
  const ids = new Set(children.map((c) => c.id).filter((x): x is string => !!x));
  const shapeByName = new Map<string, string>();
  for (const a of absorbed) {
    if (a.name && a.shape_out) shapeByName.set(a.name, `[${a.shape_out.join(",")}]`);
  }

  const nodes: BlockNode[] = children.map((ch) => ({
    id: ch.id,
    op: ch.op,
    label: ch.label,
    style: ch.style,
    subtitle: (ch.label && shapeByName.get(ch.label)) || undefined,
  }));

  const extOf = new Map<string, string>();
  const addExt = (ref: string): string => {
    const had = extOf.get(ref);
    if (had) return had;
    const eid = `ext:${extOf.size}`;
    extOf.set(ref, eid);
    nodes.push({ id: eid, op: "source", label: graph.get(ref)?.label, style: "io", hoverHtml: textHtml(ref) });
    return eid;
  };

  const edges: BlockEdge[] = [];
  const seen = new Set<string>();
  const push = (src: string, dst: string, extra?: BlockEdge) => {
    const key = `${src}->${dst}`;
    if (seen.has(key) || src === dst) return;
    seen.add(key);
    edges.push({ src, dst, tensor: extra?.tensor, shape: extra?.shape, port_label: extra?.port_label, kind: extra?.kind });
  };

  for (const e of compound.child_edges || []) {
    if (!e.src || !e.dst || !ids.has(e.dst)) continue;
    push(ids.has(e.src) ? e.src : addExt(e.src), e.dst, e);
  }
  for (const ch of children) {
    if (!ch.id) continue;
    (ch.inputs || []).forEach((inp, i) => {
      const port = (ch.input_ports || [])[i];
      const producer = port && inp.endsWith(`.${port}`) ? inp.slice(0, -(port.length + 1)) : inp;
      if (!producer || ids.has(producer)) return;
      push(addExt(producer), ch.id!);
    });
  }
  return { nodes, edges };
}

/** The step an estimate describes, as the estimate reports it: the workload
 *  the backend's phase binding turns into each phase's query / cache length,
 *  and the batch each phase ran at. */
export type GraphWorkload = {
  avgInputTokens?: number;
  avgOutputTokens?: number;
  prefixCachedTokens?: number;
  speculativeDraftTokens?: number;
  batch: { prefill?: number; decode?: number };
};

/** One encoder's step, as the estimate's encode phase ran it: the stack, the
 *  media each request carried, and the requests in the step. */
export type EncodeStep = {
  stack: string;
  media: Record<string, Record<string, number>>;
  batch: number;
};

/** The phase a graph is drawn for: a language-model step, or an encoder's. */
export type GraphPhase = "prefill" | "decode" | "encode";

/** The deployment a compute graph is drawn for: the model, its parallelism,
 *  the framework/version picked for calibration (empty = the server's default
 *  binding), the precision preset the estimate ran under and the estimate's
 *  step — an encoder's, when `encode` names one. */
export type GraphTarget = {
  modelId: string;
  gpuSku: string;
  par: Parallel;
  cpScheme: string;
  framework: string;
  precision: string;
  /** ρ = max/mean expert load the estimate ran under
   *  (`Workload.ep_load_factor`). Part of the step, not of the mesh: it
   *  decides how many rows the busiest rank's expert cell runs. */
  epLoadFactor?: number;
  workload: GraphWorkload;
  encode?: EncodeStep;
};

/** The workload of the estimate `data`, read off its response. */
export function graphWorkload(data: Estimate): GraphWorkload {
  const pre = data.phases?.prefill;
  const dec = data.phases?.decode;
  return {
    avgInputTokens: data.avg_input_tokens,
    avgOutputTokens: data.avg_output_tokens,
    prefixCachedTokens: pre?.binding?.s_prefix_cached,
    speculativeDraftTokens: dec?.binding?.speculative_draft_tokens,
    batch: { prefill: pre?.batch, decode: dec?.batch },
  };
}

/** Query for `/inference/compute-graph` — shared by the full graph and the
 *  fused-op subgraph so both render the same deployment and step. A workload
 *  field the estimate did not report is left to the endpoint's default. */
export function computeGraphQuery(t: GraphTarget, phase: GraphPhase): URLSearchParams {
  const { par, workload: w } = t;
  const qs = new URLSearchParams({ model_id: t.modelId.trim() });
  if (t.framework) qs.set("framework", t.framework);
  if (t.cpScheme) qs.set("cp_scheme", t.cpScheme);
  setPrecisionQuery(qs, t.precision);
  for (const [k, v] of Object.entries({
    pp: par.pp,
    tp_attention: par.tp_attention,
    tp_dense_ffn: par.tp_dense_ffn,
    tp_moe_expert: par.tp_moe_expert,
    ep_moe: par.ep_moe,
    cp: par.cp,
    dp_attention: par.dp_attention,
    dp_replicas: par.dp_replicas,
  })) qs.set(k, String(v));
  qs.set("phase", phase);
  if (phase === "encode") {
    // The encoder's own step: its media and batch, none of the text workload.
    if (!t.encode) throw new Error("an encode graph needs the encoder's step");
    qs.set("stack", t.encode.stack);
    qs.set("media", JSON.stringify(t.encode.media));
    qs.set("batch_size", String(t.encode.batch));
    return qs;
  }
  for (const [k, v] of Object.entries({
    avg_input_tokens: w.avgInputTokens,
    avg_output_tokens: w.avgOutputTokens,
    prefix_cached_tokens: w.prefixCachedTokens,
    speculative_draft_tokens: w.speculativeDraftTokens,
    batch_size: w.batch[phase],
  })) if (typeof v === "number" && Number.isFinite(v)) qs.set(k, String(Math.round(v)));
  return qs;
}

/** What to say when the graph endpoint refuses. The one refusal worth its own
 *  sentence is a step whose tokens the sequence world cannot split — the
 *  endpoint names it with `reason`; everything else is shown as it came. */
export function graphFailureMessage(e: unknown, prefix: string): string {
  if (e instanceof ApiError && e.reason === "sequence_not_divisible") {
    const { extent, world, degree } = e.body;
    return `Cannot draw graph: ${String(extent)} tokens do not divide ${String(world)}=${String(degree)} (${e.message}).`;
  }
  return `${prefix}: ${e instanceof Error ? e.message : String(e)}`;
}

export function ComputeGraph({ target }: { target: GraphTarget }) {
  const { modelId, par } = target;
  // An encoder's graph is its encode step alone; a language model's has two.
  const encoder = target.encode !== undefined;
  const [lmPhase, setPhase] = useState<"prefill" | "decode">("prefill");
  const phase: GraphPhase = encoder ? "encode" : lmPhase;
  const [stage, setStage] = useState(0);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState("");
  const [nodes, setNodes] = useState<BlockNode[]>([]);
  const [edges, setEdges] = useState<BlockEdge[]>([]);
  // The query of the graph on screen (null: none drawn yet). A graph drawn
  // for another phase, stage or estimate is never left standing as this one.
  const [drawnFor, setDrawnFor] = useState<string | null>(null);
  // Latest draw; a response to an older one is dropped.
  const drawSeq = useRef(0);

  // The graph is one GPU of the plan above: its TP / EP / CP shards are the
  // plan's, and every rank of a shard group runs the same graph. Pipeline
  // stages are the one axis whose ranks run different layers, so a plan with
  // PP > 1 names which stage's GPU is drawn.
  const stages = encoder ? 1 : par.pp;
  // A stage picked under another PP may not exist under this one.
  useEffect(() => { setStage(0); }, [stages]);

  const query = () => {
    const qs = computeGraphQuery(target, phase);
    if (stages > 1) qs.set("pp_rank", String(stage));
    return qs;
  };
  const wanted = modelId.trim() ? query().toString() : "";

  const tabCls = (k: "prefill" | "decode") =>
    "px-3 py-1.5 text-xs font-medium " + (phase === k ? "bg-amber-500 text-white" : "bg-white text-gray-600 hover:bg-gray-50");

  const load = async () => {
    const my = ++drawSeq.current;
    setNodes([]);
    setEdges([]);
    setSummary("");
    const id = modelId.trim();
    if (!id) {
      setDrawnFor(null);
      setLoading(false);
      setStatus("Pick a model.");
      return;
    }
    // A phase the estimate published no step for (its deployment admits no
    // request) has no graph: the endpoint would draw its default batch.
    if (!encoder && target.workload?.batch?.[lmPhase] == null) {
      setDrawnFor(query().toString());
      setLoading(false);
      setStatus(`No ${lmPhase} step runs on this deployment, so there is no graph of one.`);
      return;
    }
    const qs = query();
    const key = qs.toString();
    setDrawnFor(key);
    setLoading(true);
    setStatus(`Drawing the ${phase} graph…`);
    try {
      const data = await fetchJson<GraphResp>(`/inference/compute-graph?${qs}`);
      if (my !== drawSeq.current) return;
      setLoading(false);
      const raw = data.graph?.nodes || [];
      if (!raw.length) {
        setStatus("Empty graph.");
        return;
      }
      const blockCount = raw.filter((n) => n.op === "block_stack").length;
      const p = data.parallel || {};
      const stageText = data.stage
        ? ` · stage ${data.stage.rank}/${data.stage.pp} layers [${data.stage.layers?.[0]}, ${data.stage.layers?.[1]})`
        : "";
      setSummary(
        `${data.model} · ${data.framework} (audited at ${data.framework_audited_at}) · ${data.phase || "prefill"} · `
        + `TP_attn=${p.tp_attention} TP_ffn=${p.tp_dense_ffn} EP=${p.ep_moe}`
        + stageText
        + ` — ${raw.length} nodes (${blockCount} blocks)`,
      );
      setNodes(raw);
      setEdges(data.graph?.edges || []);
      setStatus("");
    } catch (e) {
      if (my !== drawSeq.current) return;
      setLoading(false);
      setStatus(graphFailureMessage(e, "Failed"));
    }
  };

  // Once a graph is drawn, a new phase, stage or estimate redraws it: the
  // graph on screen is always the one the controls name, or a loading line.
  useEffect(() => {
    if (drawnFor !== null && drawnFor !== wanted) void load();
  }, [wanted]);

  const placeholder = status
    || `Click “Draw graph” to draw the ${phase} step of the estimate above. Click a block to expand it.`;

  return (
    <section className="card p-4 sm:p-6 mb-6">
      <details id="infer-graph-details" className="group">
        <summary className="cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden text-lg font-semibold text-gray-900 flex flex-wrap items-center gap-2">
          <span aria-hidden="true" className="inline-block text-gray-400 text-sm transition-transform group-open:rotate-90">▶</span>
          Compute graph
          <span className="text-xs font-normal text-gray-500 group-open:hidden">click to expand</span>
        </summary>
        <div className="mt-4">
          <div className="flex items-center gap-3 flex-wrap mb-3 text-xs text-gray-600">
            {!encoder && (
            <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden" role="tablist" aria-label="Phase">
              <button id="infer-graph-phase-prefill" type="button" className={tabCls("prefill")} role="tab" aria-selected={phase === "prefill"} onClick={() => setPhase("prefill")}>Prefill</button>
              <button id="infer-graph-phase-decode" type="button" className={tabCls("decode")} role="tab" aria-selected={phase === "decode"} onClick={() => setPhase("decode")}>Decode</button>
            </div>
            )}
            {stages > 1 && (
              <label id="infer-graph-stage-wrap" className="inline-flex items-center gap-1">
                PP stage
                <select id="infer-graph-stage" className="px-2 py-1 border border-gray-300 rounded bg-white text-gray-700" value={stage} onChange={(e) => setStage(Number(e.target.value))}>
                  {Array.from({ length: stages }, (_, r) => (
                    <option key={r} value={r}>{r} of {stages}</option>
                  ))}
                </select>
              </label>
            )}
            <button id="infer-graph-refresh" type="button" disabled={loading} className="px-3 py-1.5 text-sm font-medium bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-60 disabled:cursor-wait" onClick={() => void load()}>
              {loading ? "Drawing…" : drawnFor === null ? "Draw graph" : "Redraw"}
            </button>
          </div>
          {summary ? <div id="infer-graph-summary" className="text-xs text-gray-500 mb-2">{summary}</div> : null}
          <div id="infer-compute-graph" className="dag-container p-3 min-h-[320px] overflow-auto" aria-busy={loading}>
            {nodes.length ? (
              <GraphShell rows={[{ label: "serving", nodes, edges }]} />
            ) : (
              <p id="infer-graph-status" className="p-4 text-xs text-slate-500">{placeholder}</p>
            )}
          </div>
        </div>
      </details>
    </section>
  );
}
