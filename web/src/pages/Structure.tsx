// L1 structure view: the backend groups a component's detailed graph into
// boxes by the operand each node feeds (backend/views/abstraction.py). This
// file only draws what it returns — the grouping, the sums and the refusal
// reasons are all computed server-side over the same nodes the detailed (L2)
// graph shows, so the two views can never disagree on a number.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BlockElkDag, DagLegend, ENTRY_PALETTE, ScopeL1Context, textHtml, type BlockNode, type BlockEdge, type CompareRow, type InnerL1Mode, type KVMark } from "./BlockDag";
import { DataTable, tbl, type ColumnDef } from "../DataTable";
import { formatCount, formatGiB, formatNumber, formatSI, escapeHtml } from "../format";

export type L1Box = {
  id: string;
  members: string[];
  /** null: no member states the fact (a serving graph carries topology
   *  without per-op cost), not zero. */
  flops: number | null;
  params: number | null;
  weight_bytes: number | null;
  ops: Record<string, number>;
  flops_symbolic?: string | null;
  /** Whether `flops_symbolic` still reads workload axes (backend-decided). */
  flops_is_law?: boolean;
  // Named and ordered by the backend from the declared operand roles
  // (backend/views/abstraction.py `_named`); the page only shows them.
  label: string;
  rank: number;
  // The cached tensors the box's members write and read (their `kv` union,
  // gathered server-side).
  kv?: KVMark | null;
};
export type L1Data = {
  available: boolean;
  reason?: string | null;
  boxes: L1Box[];
  node_box: Record<string, string>;
  /** The module's own totals the boxes reconcile with; a fact with no
   *  independent figure is null (not compared). */
  totals: { flops: number | null; params: number | null; weight_bytes: number | null };
  /** What `totals.weight_bytes` means: checkpoint storage on Declared L1,
   * weights read by this runtime step on Fused L1. */
  weight_semantics?: "stored" | "weight_read_step";
  core?: "attention" | "experts" | "recurrence" | null;
  other_merges?: boolean;
};

/** One compared entry as the page holds it: its label and its own graph. */
export type GraphData = { nodes: BlockNode[]; edges: BlockEdge[] };
export type DiffRow = {
  label: string;
  nodes: BlockNode[];
  edges: BlockEdge[];
  l1?: L1Data;
  fused?: GraphData;
  fusedL1?: L1Data;
};

// One line naming what a box holds, so the name can be checked against the
// ops inside without opening it.
function contents(members: BlockNode[]): string {
  // In dataflow order (a node after the members it reads), not by name.
  const ids = new Set(members.map((n) => n.id));
  const placed = new Set<string | undefined>();
  const ordered: BlockNode[] = [];
  while (ordered.length < members.length) {
    const ready = members.filter((n) => !placed.has(n.id)
      && (n.inputs || []).every((i) => !ids.has(i) || placed.has(i)));
    const next = ready.length ? ready : members.filter((n) => !placed.has(n.id));
    next.forEach((n) => { placed.add(n.id); ordered.push(n); });
  }
  const names = ordered.map((n) => n.label || "?");
  return names.length <= 3 ? names.join(" · ") : `${names.slice(0, 3).join(" · ")} +${names.length - 3}`;
}

const styleOf = (id: string) => (id === "input" ? "io" : id === "core" ? "attn" : "op");

function prettyLaw(expr: string): string {
  return expr.replace(/\*\*/g, "^").replace(/\*/g, "·");
}

/** Params stay a count (independent of B/S). FLOPs are the same walk's
 *  formula when the report still has workload axes; a bound integer only
 *  when the formula collapsed to a number. No line when the graph carries
 *  no cost. */
function boxSubtitle(b: L1Box, _members: BlockNode[]): string | undefined {
  if (b.id === "input" || b.params == null || b.flops == null) return undefined;
  const params = `${formatCount(b.params)} params`;
  if (b.flops_is_law && b.flops_symbolic) {
    const law = prettyLaw(b.flops_symbolic);
    const flops = law.length > 48 ? `${law.slice(0, 47)}…` : law;
    return `${params} · ${flops}`;
  }
  return `${params} · ${formatSI(b.flops, "FLOPs")}`;
}

// Boxes as expandable compounds over the original L2 nodes: a box's children
// are its members and its child_edges the L2 edges inside it. Root edges are
// the L2 edges between boxes, one per (source node, destination box), with the
// source node carried as `via`; between closed boxes BlockElkDag draws them as
// one edge per box pair. BlockElkDag already expands a compound in place, so
// "open one box" needs nothing new.
export function l1Graph(nodes: BlockNode[], edges: BlockEdge[], l1: L1Data, scopeId = "root"): { nodes: BlockNode[]; edges: BlockEdge[] } {
  const byId = new Map<string, BlockNode>();
  nodes.forEach((n) => { if (n.id) byId.set(n.id, n); });
  // A member may read a node nested inside a sibling compound (the MoE
  // combine reads the router's topk_scale inside `mlp.gate`); the scope's
  // edge for that read names the sibling as `src` and the nested node as
  // `via`, so a nested producer resolves to the member that holds it.
  const holder = new Map<string, string>();
  for (const e of edges) if (e.via && e.src && e.via !== e.src) holder.set(e.via, e.src);
  const boxOf = (id?: string) => (id ? l1.node_box[id] ?? l1.node_box[holder.get(id) ?? ""] : undefined);
  const member = (id: string) => (l1.node_box[id] ? id : holder.get(id) ?? id);
  const inner = new Map<string, BlockEdge[]>();
  const between = new Map<string, BlockEdge>();
  const feeds = new Map<string, Set<string>>();
  for (const e of edges) {
    const a = boxOf(e.src), b = boxOf(e.dst);
    if (!a || !b) continue;
    if (a === b) {
      if (!inner.has(a)) inner.set(a, []);
      inner.get(a)!.push(e);
    } else {
      // One root edge per source NODE, carried as `via`: an opened source
      // box then leaves from that member, not from every exit it has (MLA's
      // K + V hands K two tensors and V one); between closed boxes they are
      // one edge.
      // Per source PORT too: one node handing a box two tensors (the fused
      // rope's q_out and its cached k_out) is two edges once the boxes open,
      // each with its own kind and label.
      const key = `${e.src}${e.src_port ? `:${e.src_port}` : ""}->${b}`;
      // The tensor rides along: an opened pair draws the L2 edge's own label;
      // a closed pair standing for several different tensors draws none.
      const had = between.get(key);
      if (!had) {
        between.set(key, {
          id: `l1:${scopeId}:${key}`, src: `l1:${scopeId}:${a}`, dst: `l1:${scopeId}:${b}`, via: e.src,
          src_port: e.src_port,
          tensor: e.tensor, shape: e.shape, port_label: e.port_label, kind: e.kind,
        });
      } else if (had.tensor !== e.tensor || had.shape !== e.shape
                 || had.port_label !== e.port_label || had.kind !== e.kind) {
        // One source node feeding a box through several different tensors:
        // no single label (or line style) is true of the merged edge.
        between.set(key, { id: had.id, src: had.src, dst: had.dst, via: had.via,
                           src_port: had.src_port });
      }
      if (!feeds.has(b)) feeds.set(b, new Set());
      feeds.get(b)!.add(`l1:${scopeId}:${a}`);
    }
  }
  const readOutside = new Set<string>();
  for (const e of edges) {
    const a = boxOf(e.src), b = boxOf(e.dst);
    if (a && b && a !== b && e.src) readOutside.add(e.src);
  }
  // What a box's members read from outside the component: the box lists it
  // as its own input, so an outer edge lands on the box that reads it (and,
  // opened, on the member) rather than on whichever boxes nothing feeds.
  const outside = new Map<string, { inputs: string[]; ports: string[] }>();
  for (const n of nodes) {
    const b = boxOf(n.id);
    if (!b) continue;
    (n.inputs || []).forEach((i, k) => {
      const port = (n.input_ports || [])[k] || "";
      const bare = port && i.endsWith(`.${port}`) ? i.slice(0, -(port.length + 1)) : i;
      if (boxOf(bare)) return;
      if (!outside.has(b)) outside.set(b, { inputs: [], ports: [] });
      const o = outside.get(b)!;
      if (o.inputs.includes(i)) return;
      o.inputs.push(i);
      o.ports.push(port);
    });
  }
  const boxes = [...l1.boxes].sort((x, y) => x.rank - y.rank);
  const boxNodes: BlockNode[] = boxes.map((b) => {
    // Members rewired for the box-to-box root edges: an input from another
    // box names that box, with the source node as its port, so an opened box
    // routes each edge to the member that reads that source (K + V reaches
    // the score for K and the apply for V, not just the box's first node);
    // a member read from outside the box answers to its own id as the port.
    // Nested members are rewired too (an opened member may read another box
    // from inside), and a ported input `<producer>.<port>` is looked up by its
    // bare producer (the backend's parallel `input_ports`).
    const rewire = (n: BlockNode): BlockNode => {
      const inputs: string[] = [];
      const ports: string[] = [];
      (n.inputs || []).forEach((i, k) => {
        const port = (n.input_ports || [])[k] || "";
        const bare = port && i.endsWith(`.${port}`) ? i.slice(0, -(port.length + 1)) : i;
        const other = boxOf(bare);
        if (other && other !== b.id) {
          inputs.push(`l1:${scopeId}:${other}`);
          ports.push(member(bare));
        } else {
          inputs.push(i);
          ports.push(port);
        }
      });
      return { ...n, inputs, input_ports: ports, children: n.children?.map(rewire) };
    };
    const children = b.members.map((m) => byId.get(m)).filter((n): n is BlockNode => !!n)
      .map((n) => ({
        ...rewire(n),
        output_port: readOutside.has(n.id || "") ? n.id : n.output_port,
      }));
    const leaf = b.id === "input";
    const subtitle = boxSubtitle(b, children);
    return {
      id: `l1:${scopeId}:${b.id}`,
      op: b.id,
      label: b.label,
      style: styleOf(b.id),
      subtitle,
      detail: leaf ? undefined : contents(children),
      expandable: !leaf && children.length > 0,
      children: leaf ? undefined : children,
      child_edges: inner.get(b.id) || [],
      inputs: [...(feeds.get(b.id) || []), ...(outside.get(b.id)?.inputs || [])],
      input_ports: [...[...(feeds.get(b.id) || [])].map(() => ""), ...(outside.get(b.id)?.ports || [])],
      kv: b.kv ?? undefined,
    } as BlockNode;
  });
  return { nodes: boxNodes, edges: [...between.values()] };
}

// One component's diagram with two orthogonal switches: which graph (declared
// or runtime/fused) and which granularity (boxes or every node). Structure is
// the default when the selected family produced a view; otherwise Detailed is
// shown with the reason there is no structure view.
function applyInnerL1(nodes: BlockNode[], innerMode: InnerL1Mode): BlockNode[] {
  return nodes.map((n) => {
    if (!n.children?.length) return n;
    const walked = applyInnerL1(n.children, innerMode);
    const mode = innerMode[n.id || ""] ?? ((n.l1 as L1Data | undefined)?.available ? "structure" : "detailed");
    if (mode === "structure" && n.l1 && (n.l1 as L1Data).available) {
      const g = l1Graph(walked, n.child_edges || [], n.l1 as L1Data, n.id || "scope");
      return { ...n, children: g.nodes, child_edges: g.edges };
    }
    return { ...n, children: walked };
  });
}

export function StructureGraph({
  nodes, edges, l1, fused, fusedL1,
}: {
  nodes: BlockNode[];
  edges: BlockEdge[];
  l1?: L1Data;
  fused?: GraphData;
  fusedL1?: L1Data;
}) {
  const inDomain = l1 !== undefined || fused !== undefined;
  // The abstraction switch exists only where some family has a Structure view.
  const hasL1 = !!l1?.available || !!fusedL1?.available;
  const [family, setFamily] = useState<"declared" | "fused">(
    nodes.length > 0 || l1 !== undefined ? "declared" : "fused");
  const activeL1 = family === "fused" ? fusedL1 : l1;
  const can = !!activeL1?.available;
  const canFused = !!(fused?.nodes?.length);
  const [level, setLevel] = useState<"structure" | "detailed">(
    can ? "structure" : "detailed");
  const [innerMode, setInnerModeState] = useState<InnerL1Mode>({});
  const setInnerMode = useCallback((id: string, m: "structure" | "detailed") => {
    setInnerModeState((prev) => (prev[id] === m ? prev : { ...prev, [id]: m }));
  }, []);
  const apply = useCallback((ns: BlockNode[]) => applyInnerL1(ns, innerMode), [innerMode]);
  const scopeL1 = useMemo(
    () => ({ innerMode, setInnerMode, apply }),
    [innerMode, setInnerMode, apply],
  );
  // A different component (or a re-bench) can change which views exist. Keep
  // the user's graph family across a same-shape re-bench, but never leave them
  // on a family or abstraction the new response does not carry.
  useEffect(() => {
    if (family === "fused" && !canFused) setFamily("declared");
  }, [family, canFused]);
  useEffect(() => {
    if (level === "structure" && !can) setLevel("detailed");
  }, [level, can]);
  // Switching families starts at that family's Structure view when it has one;
  // this also preserves the old rule that newly available L1 wins over Detail.
  useEffect(() => { if (can) setLevel("structure"); }, [family, can]);
  const graph = useMemo(() => {
    if (family === "fused" && canFused) {
      return can && level === "structure"
        ? l1Graph(fused!.nodes, fused!.edges, fusedL1!, "fused")
        : fused!;
    }
    return can && level === "structure"
      ? l1Graph(nodes, edges, l1!, "root")
      : { nodes, edges };
  }, [family, can, canFused, level, nodes, edges, l1, fused, fusedL1]);
  return (
    <div>
      {inDomain ? (
        <div className="flex flex-wrap items-center gap-2 text-xs mb-2">
          <div className="inline-flex rounded-md border border-gray-200 overflow-hidden"
            role="group" aria-label="Graph family">
            {(["declared", "fused"] as const).map((f) => (
              <button key={f} type="button" aria-pressed={family === f}
                disabled={f === "fused" && !canFused}
                className={`px-2.5 py-1 ${family === f ? "bg-gray-900 text-white" : "bg-white text-gray-600 hover:bg-gray-50"} disabled:opacity-40 disabled:cursor-not-allowed`}
                onClick={() => setFamily(f)}>
                {f === "declared" ? "Declared" : "Fused"}
              </button>
            ))}
          </div>
          {hasL1 ? <div className="inline-flex rounded-md border border-gray-200 overflow-hidden"
            role="group" aria-label="Graph abstraction">
            {(["structure", "detailed"] as const).map((m) => (
              <button key={m} type="button" aria-pressed={level === m}
                disabled={m === "structure" && !can}
                className={`px-2.5 py-1 ${level === m ? "bg-gray-900 text-white" : "bg-white text-gray-600 hover:bg-gray-50"} disabled:opacity-40 disabled:cursor-not-allowed`}
                onClick={() => setLevel(m)}>
                {m === "structure" ? "Structure" : "Detailed"}
              </button>
            ))}
          </div> : null}
          <span className="text-gray-400 min-w-0">
            {!can && activeL1 !== undefined
              ? `no structure view — ${activeL1?.reason || "not available"}`
              : level === "structure"
                ? `boxes group ops by the operand they feed · click a box to open it${family === "fused" ? " · box weights are this step's reads" : ""}`
                : family === "fused"
                  ? "one prefill forward after shipped fusions · resource totals price these kernels"
                  : "every declared op"}
          </span>
        </div>
      ) : null}
      <div className="p-4 overflow-x-auto dag-container">
        <ScopeL1Context.Provider value={scopeL1}>
          <BlockElkDag key={`${family}:${level}`} nodes={graph.nodes} edges={graph.edges} />
        </ScopeL1Context.Provider>
      </div>
    </div>
  );
}

function OwnGraphs({ rows }: { rows: DiffRow[] }) {
  const [open, setOpen] = useState(false);
  const [picked, setTab] = useState(0);
  const tab = Math.min(picked, Math.max(0, rows.length - 1));
  const row = rows[tab];
  return (
    <details className="mt-3" onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary className="text-xs text-gray-500 cursor-pointer">each entry's own graph</summary>
      {open && (
        <div className="mt-2">
          <div className="flex flex-wrap gap-1 mb-2">
            {rows.map((r, i) => (
              <button key={`${r.label}:${i}`} type="button" onClick={() => setTab(i)}
                className={`text-11 px-2 py-0.5 rounded border ${i === tab
                  ? "border-amber-500 bg-amber-50 text-amber-800"
                  : "border-gray-300 text-gray-600"}`}>
                {r.label}
              </button>
            ))}
          </div>
          {row && (row.nodes.length > 0
            ? <StructureGraph key={tab} nodes={row.nodes} edges={row.edges}
                l1={row.l1} fused={row.fused} fusedL1={row.fusedL1} />
            : emptyGraph(row))}
        </div>
      )}
    </details>
  );
}

function emptyGraph(row: DiffRow, named = true) {
  return (
    <div className="text-sm text-gray-400" data-graph-empty>
      {named && row.label ? <span className="font-mono mr-1">{row.label}</span> : null}
      no representative template / no graph
    </div>
  );
}

function oneGraph(row: DiffRow) {
  if (row.nodes.length === 0) return emptyGraph(row);
  return <StructureGraph nodes={row.nodes} edges={row.edges}
    l1={row.l1} fused={row.fused} fusedL1={row.fusedL1} />;
}

/** Caption for a stacked fallback graph: the entry identity (Playground's
 *  `component @ model`, Inspector's short name) so several unlabeled DAGs
 *  stacked after a refused overlay can still be told apart. */
function GraphCaption({ row, color }: { row: DiffRow; color: string }) {
  if (!row.label) return null;
  return (
    <div className="flex items-center gap-1.5 text-xs mb-1" data-graph-label={row.label}>
      <span className="inline-block w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: color }} />
      <span className="font-mono text-gray-800">{row.label}</span>
    </div>
  );
}

function stackedGraphs(rows: DiffRow[], colors: string[]) {
  return (
    <div className="mt-2 space-y-4">
      {rows.map((r, i) => (
        <div key={`${r.label}:${i}`}>
          <GraphCaption row={r} color={colors[i % colors.length]} />
          {r.nodes.length === 0 ? emptyGraph(r, false) : oneGraph(r)}
        </div>
      ))}
    </div>
  );
}

/** One graph: Declared / Fused, each with Structure / Detailed when the
 * backend produced the corresponding L1 view.
 *  Several: overlay + each entry's own graph. A passed `diff` is always
 *  rendered (including `available=false`); when the overlay is refused,
 *  each entry's graph is stacked under its label. Empty rows stay in place. */
export function GraphShell({
  rows, diff, colors,
}: {
  rows: DiffRow[];
  diff?: GraphDiffData | null;
  colors?: string[];
}) {
  const palette = colors && colors.length ? colors : ENTRY_PALETTE;
  const hasL1 = rows.some((r) => r.l1 !== undefined);
  if (diff) {
    return (
      <div className="min-w-0">
        <GraphDiffView gd={diff} rows={rows} />
        {diff.available ? (
          <>
            <OwnGraphs rows={rows} />
            {hasL1 ? (
              <details className="mt-3">
                <summary className="text-xs text-gray-500 cursor-pointer">per-box figures (params, FLOPs, weight bytes, change against the first entry)</summary>
                <StructureCompare diff={diff} cols={rows.map((r, i) => ({
                  key: r.label, label: r.label, l1: r.l1,
                  color: palette[i % palette.length],
                }))} />
              </details>
            ) : null}
          </>
        ) : rows.length === 1 ? oneGraph(rows[0]) : rows.length > 1 ? stackedGraphs(rows, palette) : null}
      </div>
    );
  }
  if (!rows.length) return null;
  if (rows.length === 1) return oneGraph(rows[0]);
  return <OwnGraphs rows={rows} />;
}

type Col = { key: string; label: string; color: string; l1?: L1Data };

// Role-aligned comparison: one row per box (the same operand role across
// variants), one column per variant, params / FLOPs / weight bytes per cell
// with the change against the first column — the change is the compute-graph
// diff's `diff.boxes[].delta`, never subtracted here. When the column totals
// agree and the rows do not, the diff says the weights were regrouped.
export function StructureCompare({ cols, diff }: { cols: Col[]; diff?: GraphDiffData }) {
  const ok = cols.filter((c) => c.l1?.available);
  if (ok.length < 2) return null;
  const cell = (c: Col, id: string) => c.l1?.boxes.find((b) => b.id === id);
  const inAny = (id: string) => ok.map((c) => cell(c, id)).filter((b): b is L1Box => !!b);
  const table = diff?.available && diff.boxes?.length ? diff.boxes : null;
  const ids = table ? table.map((b) => b.id).filter((id) => id !== "input")
    : [...new Set(ok.flatMap((c) => c.l1!.boxes.map((b) => b.id)))]
      .filter((id) => id !== "input")
      .sort((a, b) => Math.min(...inAny(a).map((x) => x.rank)) - Math.min(...inAny(b).map((x) => x.rank)));
  const deltaOf = (id: string, k: number) => table?.find((b) => b.id === id)?.delta[k];
  // A row reads every column's name for it; they differ only for the box
  // beside the core (a parallel branch in one variant, a dead end in another).
  const rowLabel = (id: string) => [...new Set(inAny(id).map((b) => b.label))].join(" / ");
  const first = cols[0];
  const delta = (d?: DiffSums) => {
    if (!d) return null;
    const one = (v: number, fmt: (x: number) => string) => (
      <span className={v > 0 ? "text-red-500" : v < 0 ? "text-green-600" : "text-gray-300"}>{signed(v, fmt)}</span>
    );
    return <div className="text-10" data-delta>{one(d.params, (x) => `${formatCount(x)} params`)} · {one(d.flops, (x) => formatSI(x, "FLOPs"))}</div>;
  };
  const columns: ColumnDef<string>[] = [
    {
      key: "part", wrap: true,
      header: "part",
      className: "text-gray-800",
      render: (id) => rowLabel(id),
      sortValue: null,
    },
    ...cols.map((c, k): ColumnDef<string> => ({
      key: c.key,
      align: "right",
      sortValue: null,
      header: (
        <>
          <span className="inline-flex items-center gap-1 justify-end">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: c.color }} />
            <span className="font-mono">{c.label}</span>
          </span>
          {!c.l1?.available && <div className="font-normal text-gray-400 normal-case">{c.l1?.reason || "no structure view"}</div>}
        </>
      ),
      className: (id) => {
        if (!c.l1?.available) return "text-gray-300";
        return cell(c, id) ? "font-mono text-gray-900" : "text-gray-400";
      },
      cellProps: (id) => {
        const b = c.l1?.available ? cell(c, id) : undefined;
        return { title: b?.flops_symbolic || undefined };
      },
      render: (id) => {
        if (!c.l1?.available) return "—";
        const b = cell(c, id);
        if (!b) {
          // A role with no box of its own may still be there, inside a
          // box it shares (absorbed MLA's V lives in "K + V").
          const within = !id.includes("+")
            ? c.l1!.boxes.find((x) => x.id.split("+").includes(id))
            : undefined;
          return within ? <>in <span className="text-gray-600">{within.label}</span></> : "absent";
        }
        return (
          <>
            {formatCount(b.params)} <span className="text-gray-400">params</span>
            <div className="text-10 text-gray-500">{formatSI(b.flops ?? NaN, "FLOPs")} · {formatGiB(b.weight_bytes ?? NaN)}</div>
            {c !== first && delta(deltaOf(id, k))}
          </>
        );
      },
    })),
  ];
  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-900 mb-1">Structure comparison</h3>
      <div className="text-11 text-gray-400 mb-3">
        Rows are derived from each op's declared operand role, not from names; each column's rows
        add up to that entry's totals in the summary. A box "Shared by K, V" is one path both roles read from.
      </div>
      <DataTable
        columns={columns}
        rows={ids}
        rowKey={(id) => id}
        framed={false}
        footer={diff?.available && diff.totals?.regrouped ? (
          <tr data-regrouped>
            <td colSpan={cols.length + 1} className={`${tbl.wrapCell} text-11 text-amber-700`}>
              weights regrouped — the totals agree, but the parts hold them differently
            </td>
          </tr>
        ) : undefined}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Compute-graph diff (backend/views/diff.py `GraphDiff`): several entries'
// graphs lined up into classes of corresponding nodes. The backend decides
// every pairing, every owner set, every path and alternative group, every
// changed attribute and its "because", and every sum; this page only arranges
// them into one union graph for BlockElkDag and formats the fields.

export type DiffBasis = { pair: [number, number]; basis: string | null };
export type DiffClass = {
  id: string;
  members: (string | null)[];
  basis: DiffBasis[];
  boxes: (string | null)[];
  diff: Record<string, unknown[]>;
  view: boolean;
  partition: number;
  because?: { dims: Record<string, (number | null)[]>; declaration: boolean;
    weight_dtype: (string | null)[] | null };
};
export type DiffEdge = { src: string; dst: string; port: string | number; owners: number[] };
type DiffSums = { params: number; flops: number; weight_bytes: number };
export type DiffPath = {
  id: string; classes: string[]; owners: number[]; from: string[];
  into: { class: string; port: string | number }[];
  boxes: string[]; chain: string[];
  sums: (DiffSums | null)[]; kv: (KVMark | null)[];
  group: string | null; partition: number;
};
export type DiffGroup = { id: string; into: { class: string; port: string | number }[]; paths: string[] };
export type SkeletonBox = { id: string; label: string; rank: number; partition: number; classes: string[]; chain: string[] };
export type DiffBoxRow = { id: string; label: string; rank: number; present: boolean[]; delta: DiffSums[] };
export type GraphDiffData = {
  available: boolean;
  reason?: string | null;
  entries?: string[];
  grouping?: string;
  classes?: DiffClass[];
  edges?: DiffEdge[];
  paths?: DiffPath[];
  groups?: DiffGroup[];
  skeleton_boxes?: SkeletonBox[];
  reads?: string[][];
  boxes?: DiffBoxRow[];
  totals?: { delta: DiffSums[]; regrouped: boolean };
};

// Past this many entries a node's per-entry lines fold into one; hover lists them.
const MAX_ROWS = 5;

const signed = (v: number, fmt: (x: number) => string) => (v > 0 ? `+${fmt(v)}` : v < 0 ? `−${fmt(-v)}` : "=");
const shown = (v: unknown): string => {
  if (v == null) return "—";
  if (typeof v === "number") return formatNumber(v);
  if (typeof v === "object") {
    const kv = v as { writes?: string[]; reads?: string[] };
    const parts = [kv.writes?.length ? `writes ${kv.writes.join(", ")}` : "", kv.reads?.length ? `reads ${kv.reads.join(", ")}` : ""];
    return parts.filter(Boolean).join("; ") || "—";
  }
  return String(v);
};
// The attributes a changed node's per-entry line shows, in this order; the
// hover lists every changed attribute.
const LINE_KEYS: [string, string, string][] = [["op", "", ""], ["tensor_in", "in ", ""], ["tensor_out", "out ", ""],
  ["params", "", " params"], ["weight_dtype", "W:", ""], ["flops", "", " FLOPs"]];

/** The diff's classes, paths, groups and skeleton boxes as a BlockElkDag tree. */
export function diffGraph(gd: GraphDiffData, rows: DiffRow[]): { nodes: BlockNode[]; edges: BlockEdge[] } {
  const n = rows.length;
  const classes = gd.classes || [];
  const byClass = new Map(classes.map((c) => [c.id, c]));
  const nodeIndex = rows.map((r) => new Map(r.nodes.map((x) => [x.id, x])));
  const owners = (c: DiffClass) => c.members.flatMap((m, i) => (m ? [i] : []));
  const member = (c: DiffClass, i: number) => (c.members[i] ? nodeIndex[i].get(c.members[i]!) : undefined);
  const distinct = (xs: (string | undefined)[]) => [...new Set(xs.filter((x): x is string => !!x))];
  const classLabel = (c: DiffClass) => distinct(owners(c).map((i) => member(c, i)?.label || member(c, i)?.op)).join(" | ");
  const boxLabel = (id: string) => rows.map((r) => r.l1?.boxes.find((b) => b.id === id)?.label).find(Boolean) || id;
  const tone = (own: number[]) => (own.length === 1 ? `diff-u${own[0] % 8}` : own.length === n ? "diff-shared" : "diff-partial");
  const names = (own: number[]) => own.map((i) => rows[i].label).join(", ");

  // Which drawn node holds each class: its path, else its skeleton box, else itself.
  const container = new Map<string, string>();
  for (const p of gd.paths || []) for (const c of p.classes) container.set(c, `dp:${p.id}`);
  const boxed = (gd.skeleton_boxes || []).filter((b) =>
    !b.classes.every((c) => byClass.get(c)!.members.every((m, i) => !m || nodeIndex[i].get(m)?.op === "input")));
  for (const b of boxed) for (const c of b.classes) container.set(c, `db:${b.id}`);
  const holder = (cid: string) => container.get(cid) || `dc:${cid}`;

  const classNode = (c: DiffClass): BlockNode => {
    const own = owners(c);
    const first = member(c, own[0])!;
    const changed = Object.keys(c.diff).length > 0;
    const ops = distinct(own.map((i) => member(c, i)?.op));
    const node: BlockNode = {
      id: `dc:${c.id}`, label: classLabel(c), op: ops.join(" | "),
      style: changed ? "diff-changed" : own.length === n ? (c.view ? "view" : "diff-shared") : tone(own),
      inputs: [], input_ports: [],
    };
    if (!changed) {
      Object.assign(node, { detail: first.detail, tensor_in: first.tensor_in, tensor_out: first.tensor_out,
                            weight_dtype: first.weight_dtype, has_bias: first.has_bias, kv: first.kv });
    } else {
      const lines: CompareRow[] = own.map((i) => ({
        entry: i,
        text: LINE_KEYS.filter(([k]) => k in c.diff).map(([k, pre, post]) => `${pre}${shown(c.diff[k][i])}${post}`).join(" · "),
        kv: "kv" in c.diff ? member(c, i)?.kv : undefined,
      }));
      if (!("kv" in c.diff)) node.kv = first.kv;
      node.compare = own.length > MAX_ROWS ? [{ text: `${own.length} entries differ` }] : lines;
    }
    if (!changed) {
      const hover = [`${node.label} — in ${own.length === n ? "every entry" : names(own)}`];
      const bases = distinct(c.basis.map((b) => b.basis || undefined));
      if (bases.length) {
        // One line per basis, so pairs sharing a basis don't each repeat it.
        const groups = new Map<string, string[]>();
        for (const b of c.basis) {
          const key = b.basis ?? "";
          const pairs = groups.get(key) || [];
          pairs.push(`${rows[b.pair[0]].label} ~ ${rows[b.pair[1]].label}`);
          groups.set(key, pairs);
        }
        for (const [basis, pairs] of groups) hover.push(`paired by${basis ? ` ${basis}` : ""}: ${pairs.join("; ")}`);
      }
      for (const [k, vals] of Object.entries(c.diff)) {
        hover.push(`${k}:`);
        own.forEach((i) => hover.push(`  ${rows[i].label}: ${shown(vals[i])}`));
      }
      if (c.because) {
        const dims = Object.entries(c.because.dims);
        // A dim reads as bound, unrounded (formatNumber would print 1536 as 1.54K).
        const why = dims.map(([d, vs]) => `${d} ${own.map((i) => String(vs[i] ?? "—")).join(" → ")}`);
        if (c.because.declaration) why.push("the declared op differs");
        // The resident width, not the saved one (`views/diff.py`): what the
        // rank holds. A framework that loads a checkpoint narrower than it
        // is stored makes the two differ, so this does not say "stored as".
        const loaded = c.because.weight_dtype;
        if (loaded) why.push(`runs at ${own.map((i) => loaded[i] ?? "—").join(" → ")}`);
        if (why.length) hover.push(`because: ${why.join("; ")}`);
      }
      node.hoverHtml = textHtml(hover.join("\n"));
    }
    return node;
  };
  const drawn = new Map(classes.map((c) => [c.id, classNode(c)]));

  // Edges: inside one holder they are the holder's own; across holders they
  // are root edges from the holder, with the source class as `via` (one per
  // source class and destination holder), and the destination class reads the
  // source holder with the source class as its port — the l1Graph scheme.
  const childEdges = new Map<string, BlockEdge[]>();
  const root = new Map<string, BlockEdge>();
  const readOutside = new Set<string>();
  for (const e of gd.edges || []) {
    const src = byClass.get(e.src)!, dst = byClass.get(e.dst)!;
    const hs = holder(e.src), hd = holder(e.dst);
    const kinds = distinct(e.owners.map((i) => rows[i].edges.find((x) => x.src === src.members[i] && x.dst === dst.members[i])?.kind || "activation"));
    const edge: BlockEdge = {
      port_label: typeof e.port === "string" ? e.port : undefined,
      kind: kinds.length === 1 && kinds[0] !== "activation" ? kinds[0] : undefined,
      tone: e.owners.length === 1 && n > 1 ? e.owners[0] : undefined,
    };
    const d = drawn.get(e.dst)!;
    if (hs === hd) {
      if (!childEdges.has(hs)) childEdges.set(hs, []);
      childEdges.get(hs)!.push({ ...edge, src: `dc:${e.src}`, dst: `dc:${e.dst}` });
      d.inputs!.push(`dc:${e.src}`);
      d.input_ports!.push("");
      continue;
    }
    readOutside.add(e.src);
    d.inputs!.push(hs);
    d.input_ports!.push(`dc:${e.src}`);
    const key = `dc:${e.src}->${hd}`;
    const had = root.get(key);
    if (!had) root.set(key, { ...edge, id: `d:${key}`, src: hs, dst: hd, via: `dc:${e.src}` });
    else if (had.port_label !== edge.port_label || had.kind !== edge.kind || had.tone !== edge.tone) {
      root.set(key, { id: had.id, src: had.src, dst: had.dst, via: had.via,
                      kind: had.kind === edge.kind ? had.kind : undefined,
                      tone: had.tone === edge.tone ? had.tone : undefined });
    }
  }
  for (const cid of readOutside) drawn.get(cid)!.output_port = `dc:${cid}`;

  const pathNode = (p: DiffPath): BlockNode => {
    const labels = p.boxes.map(boxLabel);
    const chain = p.chain.length > 4 ? `${p.chain.slice(0, 4).join(" → ")} … (${p.chain.length} ops)` : p.chain.join(" → ");
    const rowsOf: CompareRow[] = p.owners.map((i) => ({
      entry: i,
      text: `${formatCount(p.sums[i]?.params)} params · ${formatSI(p.sums[i]?.flops ?? NaN, "FLOPs")}`,
      kv: p.kv[i] ?? undefined,
    }));
    const into = p.into.map((t) => `${drawn.get(t.class)?.label}${typeof t.port === "string" ? `.${t.port}` : ""}`);
    return {
      id: `dp:${p.id}`,
      label: `${p.classes.length} node${p.classes.length === 1 ? "" : "s"}${labels.length ? ` · ${labels.slice(0, 2).join(", ")}${labels.length > 2 ? ` +${labels.length - 2}` : ""}` : ""}`,
      style: tone(p.owners), detail: chain || undefined, expandable: true, partition: p.partition,
      compare: p.owners.length > MAX_ROWS ? [{ text: `${p.owners.length} entries — hover` }] : rowsOf,
      hoverHtml: [
        `<span style="display:inline-block;padding:0 7px;border-radius:999px;background:#eef2f7;border:1px solid #cbd5e1;font-weight:600">only in ${p.owners.length}</span>`
        + ` <span style="color:#64748b">into ${escapeHtml(into.join(", ") || "—")}</span>`,
        ...p.owners.map((i) =>
          `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${ENTRY_PALETTE[i % ENTRY_PALETTE.length]};margin-right:5px"></span>`
          + `${escapeHtml(rows[i].label)}: ${escapeHtml(rowsOf.find((r) => r.entry === i)!.text)}`),
      ].join("<br/>"),
      children: p.classes.map((c) => drawn.get(c)!),
      child_edges: childEdges.get(`dp:${p.id}`) || [],
      inputs: [],
    };
  };
  const paths = new Map((gd.paths || []).map((p) => [p.id, pathNode(p)]));

  const top: BlockNode[] = [];
  const inBox = new Set(boxed.flatMap((b) => b.classes));
  const inPath = new Set((gd.paths || []).flatMap((p) => p.classes));
  for (const c of classes) {
    if (inBox.has(c.id) || inPath.has(c.id)) continue;
    top.push({ ...drawn.get(c.id)!, partition: c.partition });
  }
  for (const b of boxed) {
    const kids = b.classes.map((c) => drawn.get(c)!);
    const changed = b.classes.filter((c) => Object.keys(byClass.get(c)!.diff).length).length;
    // Per entry, the cached tensors this box's classes touch — names only.
    const kvText = (i: number) => {
      const ws = new Set<string>(), rs = new Set<string>();
      for (const c of b.classes) {
        const kv = member(byClass.get(c)!, i)?.kv;
        kv?.writes.forEach((w) => ws.add(w.tensor));
        kv?.reads.forEach((r) => rs.add(r.tensor));
      }
      return [ws.size ? `KV write ${[...ws].join(", ")}` : "", rs.size ? `KV read ${[...rs].join(", ")}` : ""].filter(Boolean).join(" · ");
    };
    const kvAll = rows.map((_, i) => kvText(i));
    const kvRows: CompareRow[] = kvAll.every((t) => t && t === kvAll[0]) ? [{ text: `${kvAll[0]} (every entry)` }]
      : kvAll.flatMap((text, i) => (text ? [{ entry: i, text }] : []));
    top.push({
      id: `db:${b.id}`, op: b.id, label: b.label, style: changed ? "diff-changed" : "diff-shared",
      subtitle: `shared · ${b.classes.length} node${b.classes.length === 1 ? "" : "s"}${changed ? ` · ${changed} differ` : ""}`,
      detail: b.chain.join(" → ") || undefined, expandable: true, partition: b.partition,
      compare: kvRows.length > MAX_ROWS ? undefined : kvRows,
      children: kids, child_edges: childEdges.get(`db:${b.id}`) || [], inputs: [],
    });
  }
  // An "alternatives into one port" group is a note, not a container: its
  // member paths can span several bands (Mamba's one path runs from the
  // input side to the output side), and a banded compound stretched over
  // them reads as a giant empty frame. Member paths stay ordinary top-level
  // nodes in their own bands; the note links from the destination's holder
  // so it lands beside the port it annotates.
  for (const g of gd.groups || []) {
    const into = g.into.map((t) => `${drawn.get(t.class)?.label}${typeof t.port === "string" ? `.${t.port}` : ""}`);
    const noteId = `dg:${g.id}`;
    top.push({
      id: noteId,
      label: `${g.paths.length} alternatives → ${into.slice(0, 3).join(", ")}${into.length > 3 ? ` +${into.length - 3}` : ""}`,
      style: "diff-partial",
      hoverHtml: textHtml(`paths of different entries that flow into the same port: ${into.join(", ")}`),
      inputs: [],
    });
    const hd = g.into.length ? holder(g.into[0].class) : undefined;
    if (hd) {
      root.set(`note:${noteId}`, { id: `note:${noteId}`, src: hd, dst: noteId });
    }
  }
  for (const p of gd.paths || []) top.push(paths.get(p.id)!);
  return { nodes: top, edges: [...root.values()] };
}

// What the diff drawing uses, read off the payload — only what is present.
function DiffLegend({ gd, rows, startOpen, graph }: {
  gd: GraphDiffData; rows: DiffRow[]; startOpen: boolean;
  // The renderer's own legend (KV marks, edge kinds) folds with this one.
  graph: { nodes: BlockNode[]; edges: BlockEdge[] };
}) {
  const n = rows.length;
  const classes = gd.classes || [];
  const paths = gd.paths || [];
  const skeleton = classes.some((c) => c.members.every(Boolean));
  const solo = [...new Set(paths.filter((p) => p.owners.length === 1).map((p) => p.owners[0]))].sort((a, b) => a - b);
  const subset = paths.some((p) => p.owners.length > 1 && p.owners.length < n);
  const changed = classes.some((c) => Object.keys(c.diff).length);
  const toned = (gd.edges || []).some((e) => e.owners.length === 1);
  const sw = (cls: string) => <span className={`dag-legend-swatch dag-node-box s-${cls}`} />;
  const [open, setOpen] = useState(startOpen);
  useEffect(() => { setOpen(startOpen); }, [startOpen]);
  if (!open) {
    return (
      <button type="button" className="text-11 text-slate-400 px-1 py-0.5"
        onClick={() => setOpen(true)}>
        Legend ▸
      </button>
    );
  }
  return (
    <div>
      {!startOpen && (
        <button type="button" className="text-11 text-slate-400 px-1 py-0.5"
          onClick={() => setOpen(false)}>
          Legend ▾
        </button>
      )}
      <div className="dag-legend" data-diff-legend>
        {rows.map((r, i) => (
          <span key={i} className="dag-legend-item" data-legend-item={`entry:${i}`}>{sw(`diff-u${i % 8}`)}<span className="font-mono">{r.label}</span></span>
        ))}
        {skeleton && <span className="dag-legend-item" data-legend-item="skeleton">{sw("diff-shared")}shared by every entry</span>}
        {solo.length > 0 && <span className="dag-legend-item" data-legend-item="only">a path only one entry has (its colour) · click to open</span>}
        {subset && <span className="dag-legend-item" data-legend-item="subset">{sw("diff-partial")}a path some entries share</span>}
        {changed && <span className="dag-legend-item" data-legend-item="changed">{sw("diff-changed")}paired, attributes differ</span>}
        {(gd.groups || []).length > 0 && <span className="dag-legend-item" data-legend-item="group"><span className="dag-legend-swatch dag-node-box" style={{ border: "1px dotted #d97706" }} />alternatives into one port</span>}
        {toned && <span className="dag-legend-item" data-legend-item="edge:entry">
          <svg width="28" height="8" aria-hidden><line x1="0" y1="4" x2="28" y2="4" stroke="#94a3b8" strokeWidth={1.4} strokeDasharray="6 4" /></svg>
          an edge only one entry has</span>}
      </div>
      <DagLegend nodes={graph.nodes} edges={graph.edges} />
    </div>
  );
}

const PHONE_MQ = "(max-width: 639px)";

/** The union graph of a compare, or the backend's reason there is none.
 *  The box has a fixed height (600 px; 55vh on a phone), so nothing below it
 *  moves when the layout lands. The graph mounts once, a frame after the
 *  placeholder paints; later responses re-lay it out in place, keeping the
 *  paths the reader opened and the box's scroll position. */
export function GraphDiffView({ gd, rows }: { gd: GraphDiffData; rows: DiffRow[] }) {
  const graph = useMemo(() => (gd.available ? diffGraph(gd, rows) : null), [gd, rows]);
  const [mountDag, setMountDag] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [phone, setPhone] = useState(
    () => typeof window !== "undefined" && window.matchMedia(PHONE_MQ).matches,
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  // How much of the drawing lies right of the box's visible edge.
  const [moreRight, setMoreRight] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia(PHONE_MQ);
    const on = () => setPhone(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  useEffect(() => {
    if (mountDag || !gd.available) return;
    const id = requestAnimationFrame(() => setMountDag(true));
    return () => cancelAnimationFrame(id);
  }, [gd.available, mountDag]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    let frame = 0;
    const check = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        // The renderer scrolls its drawing sideways itself; the box scrolls
        // down.
        const x = el.querySelector<HTMLElement>(".dag-elk-nodes")?.parentElement ?? el;
        setMoreRight(x.scrollLeft + x.clientWidth < x.scrollWidth - 1);
      });
    };
    check();
    // Scroll does not bubble; capture sees the renderer's own.
    el.addEventListener("scroll", check, { passive: true, capture: true });
    // The box keeps its size when the layout lands; the drawing's width
    // arrives as new nodes, so watch those as well.
    const ro = new ResizeObserver(check);
    ro.observe(el);
    const mo = new MutationObserver(check);
    mo.observe(el, { childList: true, subtree: true });
    return () => {
      cancelAnimationFrame(frame);
      el.removeEventListener("scroll", check, { capture: true });
      ro.disconnect();
      mo.disconnect();
    };
  }, [mountDag, graph]);

  if (!gd.available || !graph) {
    return <div className="text-sm text-amber-700" data-diff-refused>No compute-graph diff: {gd.reason || "not available"}</div>;
  }
  // Past 4 entries the overlay stops reading as a graph (bands stretch group
  // containers into giant empty frames) — say so instead of drawing it.
  const entryCount = gd.entries?.length || rows.length;
  if (entryCount > 4) {
    return (
      <div className="text-sm text-gray-500" data-diff-too-many>
        The compute-graph diff compares at most 4 entries — this one has {entryCount}.
        Remove a few entries and the graph comes back.
      </div>
    );
  }
  const reads = (gd.reads || []).flatMap((rs, i) => (rs.length ? [[i, rs] as const] : []));
  const boxH = fullscreen ? "100vh" : phone ? "55vh" : "600px";
  return (
    <div data-graph-diff>
      {gd.grouping !== "l1" && (
        <div className="text-11 text-gray-400 mb-1">some entry has no structure view — the shared part is laid out by topological layer, not by box</div>
      )}
      {reads.map(([i, rs]) => (
        <div key={i} className="text-11 text-gray-500" data-diff-reads={i}>
          <span className="font-mono">{rows[i].label}</span> reuses {rs.join(", ")} from another layer (declared reads)
        </div>
      ))}
      <div
        data-diff-box
        data-diff-fullscreen={fullscreen ? "" : undefined}
        className="relative min-w-0"
        style={{ height: boxH }}
      >
        <div ref={scrollRef} data-diff-scroll className="dag-container h-full" style={{ overflow: "auto" }}>
          <div>
            <div className="px-3 pt-2">
              <DiffLegend gd={gd} rows={rows} startOpen={!phone} graph={graph} />
            </div>
            {!mountDag ? (
              <div className="flex items-center justify-center py-16 text-sm text-slate-400">
                Laying out {rows.length} entries…
              </div>
            ) : (
              <div className="p-4">
                <BlockElkDag nodes={graph.nodes} edges={graph.edges} legend={false} />
              </div>
            )}
          </div>
        </div>
        {moreRight && (
          <div data-diff-more-right aria-hidden
            className="pointer-events-none absolute top-0 right-0 bottom-0 w-16 flex items-center justify-end pr-1"
            style={{ background: "linear-gradient(to right, rgba(248,250,252,0), rgba(248,250,252,0.95))" }}>
            <span className="text-11 text-slate-500 rounded px-1">scroll →</span>
          </div>
        )}
        {phone && (
          <button type="button" data-diff-fullscreen-toggle
            className="absolute bottom-2 right-2 z-10 text-11 px-2 py-1 rounded bg-white/90 text-slate-700 border border-slate-300 shadow-sm"
            onClick={() => setFullscreen((v) => !v)}>
            {fullscreen ? "Exit full screen ⤡" : "Full screen ⤢"}
          </button>
        )}
      </div>
    </div>
  );
}
