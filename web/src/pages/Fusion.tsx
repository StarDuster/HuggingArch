import { useEffect, useMemo, useRef, useState } from "react";
import ELK from "elkjs/lib/elk.bundled.js";
import type { ElkEdgeSection, ElkExtendedEdge, ElkNode } from "elkjs";
import { fetchJson } from "../api";
import { DataTable, type ColumnDef } from "../DataTable";
import { SourceAnchor, type CitedAnchor } from "../SourceAnchor";

/** `style`: the class the compute graph draws the same op in (the
 *  backend's op style table; `comm` for a collective). */
type PatternNode = { op?: string; op_choices?: string[] | null; style?: string; inputs?: string[] };
type NaivePattern = {
  kind?: string;
  nodes?: Record<string, PatternNode>;
  edges?: Array<{ src?: string; dst?: string }>;
  io_mapping?: Record<string, string>;
  external_inputs?: string[];
  parallel?: {
    op?: string;
    op_choices?: string[] | null;
    style?: string;
    shared_inputs?: string[];
    min_count?: number;
    max_count?: number | null;
  };
};
type BackendProv = {
  library?: string;
  kernel_impl?: string;
  provisioning?: string;
  pip?: string;
};
type Plan = {
  fused_op: string;
  plan_file?: string;
  description?: string;
  naive_pattern?: NaivePattern;
  backends?: { signature?: string; providers?: BackendProv[] } | null;
  backends_note?: string;
};
type BindingOp = { fused_op?: string; source_anchor?: CitedAnchor | null };
type Binding = {
  framework: string;
  /** "vllm (audited at v0.30.0)" — the release the binding's facts were read at. */
  label?: string;
  ops?: BindingOp[];
};
type Registry = { plans?: Plan[]; bindings?: Binding[] };

type DagNode = { id: string; op: string; label: string; subtitle?: string | null; style: string };

const elk = new ELK();
/** Tailwind classes per style class the backend sends (`style`), plus the
 *  pattern boundary's `source` and `sink` nodes. An unknown class draws as
 *  `op`. */
const STYLE: Record<string, string> = {
  source: "bg-gray-100 text-gray-700 border border-gray-300",
  sink: "bg-amber-50 text-amber-900 border border-amber-300",
  attn: "bg-indigo-100 text-indigo-900",
  ffn: "bg-emerald-100 text-emerald-900",
  moe: "bg-violet-100 text-violet-900",
  norm: "bg-sky-100 text-sky-900",
  comm: "bg-amber-100 text-amber-900",
  view: "bg-white text-gray-600 border border-dashed border-gray-300",
  res: "bg-white text-red-700 border border-dashed border-red-300",
  embed: "bg-teal-100 text-teal-900",
  op: "bg-gray-200 text-gray-800",
};

function extId(name: string): string {
  return `ext_${name}`;
}
function sinkId(port: string): string {
  return `sink_${port}`;
}
function refToNodeId(ref: string): string {
  if (ref.startsWith("external.")) return extId(ref.slice("external.".length));
  const dot = ref.indexOf(".");
  return dot >= 0 ? ref.slice(0, dot) : ref;
}
function outputRefToNodeId(pattern: NaivePattern, ref: string): string {
  if (pattern.kind === "parallel" && ref.startsWith("parallel.")) return "siblings";
  return refToNodeId(ref);
}
function sourceNode(name: string): DagNode {
  return {
    id: extId(name),
    op: `external.${name}`,
    label: name,
    subtitle: "source",
    style: "source",
  };
}
function addSinks(pattern: NaivePattern, nodes: DagNode[], edges: Array<{ src: string; dst: string }>): void {
  for (const [port, target] of Object.entries(pattern.io_mapping || {})) {
    if (target.startsWith("external.")) continue;
    const id = sinkId(port);
    nodes.push({ id, op: target, label: port, subtitle: "sink", style: "sink" });
    edges.push({ src: outputRefToNodeId(pattern, target), dst: id });
  }
}

export function patternToDag(pattern: NaivePattern | undefined): { nodes: DagNode[]; edges: Array<{ src: string; dst: string }> } {
  if (!pattern) return { nodes: [], edges: [] };
  const edges = (pattern.edges || []).map((e) => ({
    src: refToNodeId(e.src || ""),
    dst: refToNodeId(e.dst || ""),
  })).filter((e) => e.src && e.dst);
  if (pattern.kind === "parallel") {
    const nodes: DagNode[] = (pattern.external_inputs || []).map(sourceNode);
    const pp = pattern.parallel || {};
    const opLabel = pp.op || (pp.op_choices || []).join(" | ");
    const countLabel = pp.max_count ? `N=${pp.min_count}..${pp.max_count}` : `N≥${pp.min_count}`;
    nodes.push({
      id: "siblings",
      op: opLabel,
      label: `${countLabel} × ${opLabel}`,
      subtitle: "parallel siblings (interchangeable)",
      style: pp.style || "op",
    });
    addSinks(pattern, nodes, edges);
    return { nodes, edges };
  }
  const nodes: DagNode[] = (pattern.external_inputs || []).map(sourceNode);
  for (const [pnName, pnSpec] of Object.entries(pattern.nodes || {})) {
    const opLabel = pnSpec.op || (pnSpec.op_choices || []).join(" | ");
    nodes.push({
      id: pnName,
      op: opLabel,
      label: pnName,
      subtitle: pnSpec.op_choices ? `choices: ${pnSpec.op_choices.join(", ")}` : null,
      style: pnSpec.style || "op",
    });
  }
  addSinks(pattern, nodes, edges);
  return { nodes, edges };
}

function edgePath(sec: ElkEdgeSection): string {
  const pts = [sec.startPoint, ...(sec.bendPoints || []), sec.endPoint];
  return pts.map((p, i) => `${i ? "L" : "M"} ${p.x} ${p.y}`).join(" ");
}
function dagNodeWidth(n: DagNode): number {
  const chars = Math.max(n.label.length, n.subtitle?.length || 0);
  return Math.max(160, Math.min(280, 24 + chars * 7));
}

function Subgraph({ pattern }: { pattern: NaivePattern | undefined }) {
  const { nodes, edges } = useMemo(() => patternToDag(pattern), [pattern]);
  const [laid, setLaid] = useState<ElkNode | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!nodes.length) {
      setLaid(null);
      return;
    }
    let cancelled = false;
    const graph: ElkNode = {
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "DOWN",
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.layered.spacing.nodeNodeBetweenLayers": "36",
        "elk.spacing.nodeNode": "20",
      },
      children: nodes.map((n) => ({ id: n.id, width: dagNodeWidth(n), height: 40 })),
      edges: edges.map((e, i) => ({ id: `e${i}`, sources: [e.src], targets: [e.dst] })),
    };
    elk.layout(graph).then((r) => { if (!cancelled) { setLaid(r); setErr(""); } })
      .catch((e) => { if (!cancelled) { setLaid(null); setErr(e instanceof Error ? e.message : String(e)); } });
    return () => { cancelled = true; };
  }, [nodes, edges]);

  if (!nodes.length) {
    return <p className="p-3 text-xs text-gray-400">No subgraph available (parallel-form patterns surface as a single sibling cluster — see the description for details).</p>;
  }
  if (err) {
    return (
      <ul className="text-xs font-mono text-gray-600 space-y-1">
        {nodes.map((n) => <li key={n.id}>{n.label} <span className="text-gray-400">{n.op}</span></li>)}
        {edges.map((e, i) => <li key={i} className="text-gray-400">{e.src} → {e.dst}</li>)}
      </ul>
    );
  }
  if (!laid) return <p className="p-3 text-xs text-gray-400">ELK layout…</p>;
  const w = Math.max(laid.width || 0, 280);
  const h = Math.max(laid.height || 0, 180);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const allEdges: ElkExtendedEdge[] = [];
  const walk = (n: ElkNode) => {
    for (const e of (n.edges || []) as ElkExtendedEdge[]) allEdges.push(e);
    for (const c of n.children || []) walk(c);
  };
  walk(laid);
  return (
    <div className="relative overflow-auto" style={{ minHeight: 240 }}>
      <svg width={w} height={h} className="absolute inset-0 pointer-events-none">
        {allEdges.flatMap((e) => (e.sections || []).map((sec, i) => (
          <path key={`${e.id}-${i}`} d={edgePath(sec)} fill="none" stroke="#9ca3af" strokeWidth="1.3" />
        )))}
      </svg>
      <div className="relative" style={{ width: w, height: h }}>
        {(laid.children || []).map((c) => {
          const n = byId.get(c.id);
          if (!n) return null;
          return (
            <div
              key={c.id}
              className={`absolute rounded-lg px-2 py-1 text-11 font-mono overflow-hidden ${STYLE[n.style] || STYLE.op}`}
              style={{ left: c.x, top: c.y, width: c.width, height: c.height }}
              title={[n.label, n.op !== n.label ? n.op : "", n.subtitle ?? ""].filter(Boolean).join("\n")}
            >
              <div className="font-medium truncate">{n.label}</div>
              {n.subtitle ? <div className="opacity-70 italic truncate text-10">{n.subtitle}</div> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

type NoteBlock =
  | { kind: "heading"; text: string }
  | { kind: "prose"; text: string }
  | { kind: "code"; lines: string[] };

/** A plan's notes are the comment block at the top of its yaml file: prose
 *  hard-wrapped at ~80 columns, indented lines for formulas, lists and file
 *  references, a `----` underline under a heading. Prose lines join into one
 *  paragraph; indented lines keep their layout. */
function noteBlocks(text: string): NoteBlock[] {
  const out: NoteBlock[] = [];
  for (const para of text.split(/\n\s*\n/)) {
    const lines = para.split("\n");
    let prose: string[] = [];
    let code: string[] = [];
    const flush = () => {
      if (prose.length) out.push({ kind: "prose", text: prose.join(" ") });
      if (code.length) out.push({ kind: "code", lines: code });
      prose = [];
      code = [];
    };
    lines.forEach((line, i) => {
      if (!line.trim()) return;
      if (/^\s*[-=]{3,}\s*$/.test(line)) return; // underline, taken with the line above
      if (/^\s*[-=]{3,}\s*$/.test(lines[i + 1] ?? "")) {
        flush();
        out.push({ kind: "heading", text: line.trim() });
        return;
      }
      if (/^\s/.test(line)) {
        if (prose.length) { out.push({ kind: "prose", text: prose.join(" ") }); prose = []; }
        code.push(line);
      } else {
        if (code.length) { out.push({ kind: "code", lines: code }); code = []; }
        prose.push(line.trim());
      }
    });
    flush();
  }
  return out;
}

/** `name` spans in the notes are code. */
function NoteText({ text }: { text: string }) {
  return (
    <>
      {text.split(/(`[^`]+`)/).map((part, i) =>
        part.startsWith("`") && part.endsWith("`") && part.length > 1
          ? <code key={i} className="font-mono text-11 bg-gray-100 text-gray-800 rounded px-1">{part.slice(1, -1)}</code>
          : part)}
    </>
  );
}

function PlanNotes({ text }: { text: string }) {
  const blocks = noteBlocks(text);
  const indent = (lines: string[]) =>
    Math.min(...lines.map((l) => l.match(/^\s*/)?.[0].length ?? 0));
  return (
    <div className="plan-notes mt-2 space-y-2 bg-gray-50 rounded p-3 max-w-3xl">
      {blocks.map((b, i) => b.kind === "heading"
        ? <h4 key={i} className="text-xs font-semibold text-gray-800 pt-1"><NoteText text={b.text} /></h4>
        : b.kind === "prose"
          ? <p key={i}><NoteText text={b.text} /></p>
          : (
            <pre key={i} className="font-mono text-11 text-gray-700 whitespace-pre-wrap break-words">
              {b.lines.map((l) => l.slice(indent(b.lines))).join("\n")}
            </pre>
          ))}
    </div>
  );
}

const opInHash = () => new URLSearchParams(window.location.hash.replace(/^#/, "")).get("op");

function fwKey(b: Binding): string {
  return b.framework;
}

export function Fusion() {
  const [reg, setReg] = useState<Registry | null>(null);
  const [err, setErr] = useState("");
  const [filter, setFilter] = useState<Set<string>>(() => new Set());
  const [selected, setSelected] = useState<string | null>(null);

  // The URL hash names the selected plan (`#op=…`). A click is a new
  // history entry, so Back / Forward walk the plans viewed; a selection the
  // page makes itself (the first plan, a filtered-out one replaced) only
  // rewrites the current entry. An edited hash or Back / Forward selects
  // the plan it names.
  const userPick = useRef(false);

  useEffect(() => {
    fetchJson<Registry>("/fusion/registry")
      .then((j) => {
        setReg(j);
        const first = (j.plans || [])[0]?.fused_op;
        const hash = opInHash();
        setSelected(hash && (j.plans || []).some((p) => p.fused_op === hash) ? hash : (first || null));
      })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    const onHash = () => {
      const op = opInHash();
      if (op && (reg?.plans || []).some((p) => p.fused_op === op)) setSelected(op);
    };
    window.addEventListener("hashchange", onHash);
    window.addEventListener("popstate", onHash);
    return () => {
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("popstate", onHash);
    };
  }, [reg]);

  useEffect(() => {
    if (!selected) return;
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    params.set("op", selected);
    const next = `#${params.toString()}`;
    const push = userPick.current;
    userPick.current = false;
    if (next === window.location.hash) return;
    const url = `${window.location.pathname}${window.location.search}${next}`;
    if (push) history.pushState(null, "", url);
    else history.replaceState(null, "", url);
  }, [selected]);

  const pick = (op: string) => {
    if (op === selected) return;
    userPick.current = true;
    setSelected(op);
  };

  const plans = reg?.plans || [];
  const bindings = reg?.bindings || [];
  const visible = useMemo(() => {
    if (filter.size === 0) return plans;
    return plans.filter((p) => bindings.some((b) =>
      filter.has(fwKey(b)) && (b.ops || []).some((op) => op.fused_op === p.fused_op),
    ));
  }, [plans, bindings, filter]);

  useEffect(() => {
    if (selected && !visible.find((p) => p.fused_op === selected) && visible[0]) {
      setSelected(visible[0].fused_op);
    }
  }, [visible, selected]);

  const plan = plans.find((p) => p.fused_op === selected) || null;
  const frameworksFor = (op: string) =>
    [...new Set(bindings.filter((b) => (b.ops || []).some((o) => o.fused_op === op)).map((b) => b.framework))];

  const toggleFw = (key: string) => {
    setFilter((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  if (err) {
    return (
      <main className="w-full page-shell py-6 flex-1">
        <p className="text-sm text-red-600">Failed to load registry: {err}</p>
      </main>
    );
  }

  if (!reg) {
    return (
      <main className="w-full page-shell py-6 flex-1">
        <p className="text-sm text-gray-400">Loading…</p>
      </main>
    );
  }

  return (
    <main className="w-full page-shell py-6 flex-1">
      <section className="card p-4 mb-4">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-xs font-medium text-gray-600">Framework:</span>
          <div id="fusion-framework-chips" className="flex items-center gap-2 flex-wrap">
            {bindings.map((b) => {
              const key = fwKey(b);
              const active = filter.has(key);
              return (
                <button
                  key={key}
                  type="button"
                  className={`px-2.5 py-1 text-11 font-mono rounded-full border transition ${active ? "bg-amber-500 text-white border-amber-500" : "bg-white text-gray-700 border-gray-300 hover:border-amber-400"}`}
                  onClick={() => toggleFw(key)}
                >
                  {b.label || key}
                </button>
              );
            })}
          </div>
          <span className="text-11 text-gray-400 ml-auto">Empty filter shows all.</span>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)] xl:grid-cols-[280px_1fr] gap-4">
        <section className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-800">Fusion Plans</h3>
            <span id="fusion-plan-count" className="text-10 text-gray-400">{visible.length} / {plans.length}</span>
          </div>
          <ul id="fusion-plan-list" className="space-y-1">
            {visible.map((p) => {
              const fws = frameworksFor(p.fused_op);
              const active = p.fused_op === selected;
              return (
                <li key={p.fused_op}>
                  <button
                    type="button"
                    className={`w-full text-left flex items-center gap-2 px-2.5 py-1.5 rounded text-xs ${active ? "bg-amber-50 text-amber-800 font-semibold" : "hover:bg-gray-50 text-gray-700"}`}
                    onClick={() => pick(p.fused_op)}
                  >
                    <span className="font-mono truncate" title={p.fused_op}>{p.fused_op}</span>
                    <span className="ml-auto flex items-center gap-0.5 shrink-0">
                      {fws.length
                        ? fws.map((fw) => <span key={fw} className="text-[9px] font-mono px-1 py-0.5 rounded bg-gray-100 text-gray-500">{fw}</span>)
                        : <span className="text-[9px] italic text-amber-600" title="No framework fuses this into one kernel">unbound</span>}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>

        <section id="fusion-plan-detail" className="card p-5">
          {!plan ? (
            <p className="text-sm text-gray-400">Pick a plan.</p>
          ) : (
            <PlanDetail plan={plan} bindings={bindings} />
          )}
        </section>
      </div>
    </main>
  );
}

function PlanDetail({ plan, bindings }: { plan: Plan; bindings: Binding[] }) {
  const pattern = plan.naive_pattern;
  const rows = Object.entries(pattern?.io_mapping || {});
  const corpusHref = (library?: string, impl?: string) => {
    const p = new URLSearchParams({ view: "envelope", signature: plan.backends?.signature || "" });
    if (library) p.set("framework", library);
    if (impl) p.set("kernel", impl);
    return `/kernel-corpus.html#${p.toString()}`;
  };
  const providers = plan.backends?.providers || [];
  const ioColumns: ColumnDef<[string, string]>[] = [
    { key: "op", header: "Fused operand", className: "font-mono text-gray-800", render: (r) => r[0], sortValue: (r) => r[0] },
    { key: "ref", header: "Subgraph port", className: "font-mono text-gray-500", render: (r) => r[1], sortValue: (r) => r[1] },
  ];
  const hitFor = (b: Binding) => (b.ops || []).find((op) => op.fused_op === plan.fused_op);
  const bindingColumns: ColumnDef<Binding>[] = [
    {
      key: "framework",
      header: "Framework",
      className: (b) => `font-mono${hitFor(b) ? " text-gray-800" : ""}`,
      render: (b) => b.label || b.framework,
      sortValue: (b) => b.label || b.framework,
    },
    {
      key: "anchor", wrap: true,
      header: "source_anchor",
      // a dotted path has no spaces: it breaks anywhere rather than widen the page
      className: (b) => (hitFor(b) ? "font-mono text-11 text-gray-500 break-all" : "italic"),
      cellProps: (b) => ({ title: hitFor(b)?.source_anchor?.text }),
      render: (b) => {
        const hit = hitFor(b);
        if (!hit) return "— unbound —";
        return hit.source_anchor ? <SourceAnchor anchor={hit.source_anchor} /> : "—";
      },
      sortValue: (b) => hitFor(b)?.source_anchor?.text,
    },
  ];
  const providerColumns: ColumnDef<BackendProv>[] = [
    {
      key: "library",
      render: (p) => <span className="px-1.5 py-0.5 rounded bg-gray-100 text-gray-700 text-11 font-mono">{p.library}</span>,
      sortValue: (p) => p.library,
    },
    {
      key: "kernel_impl",
      className: "font-mono text-11",
      render: (p) => <a href={corpusHref(p.library, p.kernel_impl)} className="text-blue-600 hover:underline">{p.kernel_impl}</a>,
      sortValue: (p) => p.kernel_impl,
    },
    {
      key: "provisioning",
      render: (p) => (
        <>
          {p.provisioning === "source_build"
            ? <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 text-10">source-build</span>
            : <span className="px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 text-10">pip</span>}
          {p.pip ? <span className="text-10 text-gray-400 font-mono ml-1">{p.pip}</span> : null}
        </>
      ),
      sortValue: (p) => (p.provisioning === "source_build" ? "source-build" : "pip"),
    },
  ];
  return (
    <>
      <header className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <h2 className="text-lg font-semibold text-gray-900 font-mono">{plan.fused_op}</h2>
        <span className="min-w-0 break-all sm:break-normal text-10 text-gray-400 font-mono">{plan.plan_file}</span>
      </header>
      {plan.description ? (
        <details className="mb-4 text-xs text-gray-600 leading-relaxed">
          <summary className="cursor-pointer text-gray-500 hover:text-gray-800">Notes</summary>
          <PlanNotes text={plan.description} />
        </details>
      ) : null}

      <h3 className="text-sm font-semibold text-gray-700 mb-2">Naive subgraph</h3>
      <div className="mb-4 rounded-xl bg-gray-50" style={{ minHeight: 240 }}>
        <Subgraph pattern={pattern} />
      </div>

      <h3 className="text-sm font-semibold text-gray-700 mb-2">I/O mapping</h3>
      <DataTable columns={ioColumns} rows={rows} rowKey={(r) => r[0]} className="max-w-md mb-5" />

      <h3 className="text-sm font-semibold text-gray-700 mt-5 mb-2">Framework bindings</h3>
      <DataTable
        columns={bindingColumns}
        rows={bindings}
        rowKey={(b) => b.framework}
        rowClassName={(b) => (hitFor(b) ? undefined : "text-gray-400")}
      />

      <h3 className="text-sm font-semibold text-gray-700 mt-5 mb-2">
        Supported backends
      </h3>
      {!providers.length ? (
        plan.backends_note
          ? <p className="text-xs text-gray-500 leading-relaxed bg-gray-50 rounded p-2.5 border border-gray-100">{plan.backends_note}</p>
          : <p className="text-xs text-gray-400">No measurable backend yet.</p>
      ) : (
        <>
          <p className="text-11 text-gray-500 mb-1">
            signature <span className="font-mono text-gray-700">{plan.backends?.signature}</span>
            {" · "}{providers.length} {providers.length === 1 ? "library" : "libraries"}, envelope = fastest
            {" · "}<a href={corpusHref()} className="text-blue-600 hover:underline">Kernel Bench ↗</a>
          </p>
          <DataTable columns={providerColumns} rows={providers} rowKey={(p) => `${p.library}-${p.kernel_impl}`} />
        </>
      )}
    </>
  );
}
