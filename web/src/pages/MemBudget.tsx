import { DataTable, type ColumnDef } from "../DataTable";
import { formatGiB } from "../format";
import { FormulaHover } from "../FormulaHover";
import type { FormulaFields } from "../tooltip";
import type { DeploymentBudget, WeightNode } from "./ResultSummary";

type Flat = { node: WeightNode; depth: number };

function flatten(node: WeightNode, depth: number, out: Flat[]) {
  out.push({ node, depth });
  for (const c of node.children || []) flatten(c, depth + 1, out);
}


function nodeFormula(node: WeightNode, result: string, totalBytes: number): FormulaFields | undefined {
  const symbolic = node.leaf_symbolic || node.shard_symbolic;
  const substituted = node.leaf_substituted || node.shard_substituted;
  if (!symbolic && !substituted) return undefined;
  const factors = node.leaf_copy_factors || [];
  const copies = factors.reduce((p, f) => p * (f.n || 1), 1);
  const cnt = copies > 1 ? copies : node.leaf_cnt || 0;
  const fields: FormulaFields = {
    title: node.name,
    symbolic,
    substituted,
    note: node.leaf_note || (node.weight_dtype ? `dt_w = ${node.weight_dtype}` : undefined),
    result,
  };
  if (cnt > 1 && totalBytes > 0) {
    fields.per_layer_result = formatGiB(totalBytes / cnt);
    if (factors.length) fields.copy_factors = factors;
    else {
      fields.cnt = cnt;
      fields.cnt_unit = "layers";
    }
  }
  return fields;
}

function namedSubterm(formula: FormulaFields | undefined, symbol: string): FormulaFields | undefined {
  return formula?.subterms?.find((st) => st.symbol === symbol);
}

/** One cluster's per-GPU budget. A disaggregated estimate draws one per
 *  cluster, each `cluster` naming whose it is. */
export function MemBudget({ result, cluster }: { result: DeploymentBudget; cluster?: "prefill" | "decode" }) {
  const idPrefix = cluster ? `infer-${cluster}` : "infer";
  const mem = result.memory || {};
  const wb = result.weight_breakdown || {};
  const frac = result.gpu_memory_fraction;
  const nGpus = result.cluster_size;
  const deployBytes = wb.deployment_total_bytes || 0;
  const publishedBytes = wb.published_checkpoint_bytes || 0;
  const leftOut = wb.left_out_stacks || [];
  const totalData = wb.deployment_data_bytes || 0;
  const totalScale = wb.deployment_scale_bytes || 0;

  const formulas = result.formulas || {};
  const chain: Array<{ label: string; val: number; cls: string; formula?: FormulaFields }> = [
    { label: "HBM capacity", val: mem.total_bytes || 0, cls: "text-gray-900", formula: formulas.mem_total },
    { label: frac != null ? `× fraction (${(frac * 100).toFixed(0)}%)` : "× fraction", val: mem.usable_bytes || 0, cls: "text-gray-700", formula: formulas.usable },
    { label: "− weights / GPU", val: mem.weight_bytes_per_gpu || 0, cls: "text-red-600", formula: formulas.weight_per_rank },
    {
      label: "= KV cache budget",
      val: mem.kv_budget_bytes || 0,
      cls: mem.fits ? "text-green-700 font-semibold" : "text-red-600 font-semibold",
      formula: formulas.kv_budget,
    },
  ];
  if (mem.host_offload && (mem.host_offload_bytes_per_gpu || 0) > 0) {
    // Beside the chain, not in it: the weights above already leave them out.
    chain.splice(3, 0, {
      label: "(Engram / PLE tables in CPU memory, not in the weights above)",
      val: mem.host_offload_bytes_per_gpu || 0,
      cls: "text-gray-400",
    });
  }
  if ((mem.weight_deficit_bytes || 0) > 0) {
    chain.push({
      label: "Short by",
      val: mem.weight_deficit_bytes || 0,
      cls: "text-red-600 font-semibold",
      formula: formulas.weight_deficit,
    });
  }

  const flat: Flat[] = [];
  for (const g of wb.groups || []) flatten(g, 0, flat);
  for (const m of wb.model_level || []) flatten(m, 0, flat);
  const totalFull = wb.deployment_total_bytes || 0;
  const totalPerRank = wb.total_bytes_per_rank || 0;

  type ChainRow = (typeof chain)[number];
  const chainCols: ColumnDef<ChainRow>[] = [
    { key: "label", colSpan: () => 2, className: "text-gray-600", render: (r) => r.label, sortValue: null },
    { key: "gap", header: "", sortValue: null },
    {
      key: "val",
      align: "right",
      sortValue: null,
      className: (r) => `font-mono ${r.cls}`,
      render: (r) => <FormulaHover value={formatGiB(r.val)} formula={r.formula} cls={r.cls} />,
    },
  ];

  type FlatRow = Flat & { i: number };
  const modRows: FlatRow[] = [];
  flat.forEach((f, i) => {
    const bf = f.node.bytes_full || 0;
    const hasChildren = (f.node.children || []).length > 0;
    if (bf < 1 && !hasChildren) return;
    modRows.push({ ...f, i });
  });
  const modCols: ColumnDef<FlatRow>[] = [
    {
      key: "op",
      sortValue: null,
      header: "Op",
      className: ({ node, depth }) =>
        `text-xs text-gray-700 ${(depth === 0 && !node.class_) || node.is_component ? "font-semibold" : "font-normal"}`,
      cellProps: ({ depth }) => ({ style: { paddingLeft: 8 + depth * 16 } }),
      render: ({ node, depth }) => (
        <>
          {node.name}
          {node.is_component && node.op && node.op !== node.name ? (
            <span className="ml-2 text-11 text-gray-400">{node.op}</span>
          ) : null}
          {node.class_ && !(depth === 0 && !node.class_) ? (
            <span className="ml-2 text-10 text-gray-400">[{node.class_}]</span>
          ) : null}
        </>
      ),
    },
    {
      key: "shape",
      sortValue: null,
      header: "Weight shape",
      align: "right",
      className: "text-11 font-mono text-gray-500",
      render: ({ node }) => node.weight_shape || "—",
    },
    {
      key: "dtype",
      sortValue: null,
      header: "Dtype",
      align: "right",
      className: "text-11 font-mono text-gray-500",
      render: ({ node }) => node.weight_dtype || "—",
    },
    {
      key: "full",
      sortValue: null,
      header: "Full",
      align: "right",
      className: "text-11 text-gray-500 font-mono",
      render: ({ node }) => {
        const bf = node.bytes_full || 0;
        return <FormulaHover value={formatGiB(bf)} formula={nodeFormula(node, formatGiB(bf), bf)} />;
      },
    },
    {
      key: "pct",
      sortValue: null,
      header: "% of total",
      align: "right",
      className: "text-11 text-gray-500",
      render: ({ node }) => {
        const bf = node.bytes_full || 0;
        const pct = totalFull > 0 ? (bf / totalFull) * 100 : 0;
        const barW = Math.min(100, Math.max(0, pct)).toFixed(1);
        return pct >= 0.05 ? (
          <>
            <span className="inline-block w-8 h-[5px] bg-gray-200 rounded-sm mr-1 align-middle">
              <span className="block h-full bg-blue-400 rounded-sm" style={{ width: `${barW}%` }} />
            </span>
            {pct.toFixed(1)}%
          </>
        ) : (
          "—"
        );
      },
    },
    {
      key: "divisor",
      sortValue: null,
      header: "Divisor",
      align: "right",
      className: "text-11 text-gray-400",
      render: ({ node }) => node.shard_label,
    },
    {
      key: "perRank",
      sortValue: null,
      header: "Per rank",
      align: "right",
      className: "font-mono text-xs text-gray-800",
      render: ({ node }) => {
        const br = node.bytes_per_rank || 0;
        return <FormulaHover value={formatGiB(br)} formula={nodeFormula(node, formatGiB(br), br)} />;
      },
    },
  ];

  return (
    <section className="card p-5 mb-6">
      <div className="flex items-baseline justify-between mb-3 flex-wrap gap-2">
        <h3 className="font-semibold text-gray-900">Per-GPU Memory Budget{cluster ? ` — ${cluster} cluster` : ""}</h3>
        <div id={`${idPrefix}-weights-source`} className="text-xs text-gray-500">
          Weights to load: <span className="font-mono text-gray-700">
            <FormulaHover value={formatGiB(deployBytes)} formula={formulas.deployment} />
          </span>
          {totalScale > 0 ? (
            <>
              {" = "}
              <span className="font-mono">
                <FormulaHover value={formatGiB(totalData)} formula={namedSubterm(formulas.deployment, "data")} />
              </span> data +{" "}
              <span className="font-mono text-amber-700">
                <FormulaHover value={formatGiB(totalScale)} formula={namedSubterm(formulas.deployment, "scale")} cls="text-amber-700" />
              </span> scale
            </>
          ) : null}
          {" at "}
          <span className="text-indigo-700 font-medium">{wb.weight_dtype || ""}</span>
          {", split over "}{nGpus ?? "—"}{nGpus === 1 ? " GPU" : " GPUs"}{" → "}
          <span className="font-mono">
            <FormulaHover value={formatGiB(mem.weight_bytes_per_gpu || 0)} formula={formulas.weight_per_rank} />
          </span>
          {" per GPU"}
          {(nGpus ?? 1) > 1 ? (
            <span className="text-gray-400"> (sharded weights are divided; replicated ones, such as norms, are copied to every GPU)</span>
          ) : null}
          {publishedBytes ? (
            <div className="text-gray-400 mt-0.5">
              Published checkpoint: {formatGiB(publishedBytes)}.
              {/* A left-out stack's bytes are what loading it would cost at
                  this precision (`stored_bytes_by_stack`), not its share of
                  the published file: under another precision the two differ. */}
              {leftOut.length ? (
                <>
                  {" Not in this budget: "}
                  {leftOut.map((s, i) => (
                    <span key={s.stack}>
                      {i > 0 ? ", " : ""}
                      {s.encoder ? `${s.stack} encoder` : s.stack} weights, {formatGiB(s.bytes)} at this precision
                      {s.encoder
                        ? " (counted with the encode phase)"
                        : " (aux module weights are off)"}
                    </span>
                  ))}
                  {"."}
                </>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
      <div id={`${idPrefix}-mem-table`}>
        <DataTable
          columns={chainCols}
          rows={chain}
          rowKey={(r) => r.label}
          hideHeader
          framed={false}
          tableClassName="text-sm"
        />
      </div>
      <details className="mt-4 group">
        <summary className="flex items-baseline justify-between mb-2 cursor-pointer select-none list-none hover:bg-gray-50 -mx-2 px-2 py-1 rounded">
          <div className="text-xs text-gray-400 uppercase tracking-wide"><span aria-hidden="true" className="inline-block mr-1 transition-transform group-open:rotate-90">▶</span>Per-Module Breakdown</div>
          <div className="text-11 text-gray-400">
            spec: <span className="font-mono">{wb.spec_name || "?"}</span>
          </div>
        </summary>
        <DataTable
          columns={modCols}
          rows={modRows}
          rowKey={(r) => `${r.node.name}-${r.i}`}
          framed={false}
          tableClassName="text-sm"
          empty="No module data"
          footer={
            <tr className="border-t-2 border-gray-300 bg-gray-50">
              <td className="py-1.5 text-xs text-gray-800 font-semibold" style={{ paddingLeft: 8 }}>Total</td>
              <td className="py-1.5 text-right text-11 text-gray-300">—</td>
              <td className="py-1.5 text-right text-11 text-gray-300">—</td>
              <td className="py-1.5 text-right text-11 text-gray-700 font-mono font-semibold">
                <FormulaHover value={formatGiB(totalFull)} formula={formulas.deployment} />
              </td>
              <td className="py-1.5 text-right text-11 text-gray-500">100%</td>
              <td className="py-1.5 text-right text-11 text-gray-300">—</td>
              <td className="py-1.5 text-right font-mono text-xs text-gray-900 font-semibold">
                <FormulaHover value={formatGiB(totalPerRank)} formula={formulas.weight_per_rank} />
              </td>
            </tr>
          }
        />
      </details>
    </section>
  );
}
