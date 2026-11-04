// Derived groups shown per declaration. A template reading its stack's
// `read: all` history is split by the backend into one invocation per
// history length (its "derived groups"); every row / card / node from one
// carries the fields below, and the pages group on them — never on the
// `[Sar=m]` suffix of a template name.
import type { FormulaFields } from "./tooltip";

/** One derived group's identity, as the backend reports it
 *  (`BlockMeta.derived_group`): the declaration it was split from and the
 *  history length its layers see. The page groups on these fields; the
 *  `[Sar=m]` suffix of a template name is never read. */
export type Declaration = {
  layer_path: number[];
  assignment_index: number | null;
  base_template: string;
};

export type DerivedGroup = {
  base_template: string;
  stream: string;
  axis: string;
  visible: number;
  declaration: Declaration;
  stack_name?: string | null;
  layer_indices: number[];
  count: number;
};

/** `m -> layers` of one declaration. */
export type HistogramEntry = { visible: number; layers: number; layer_indices: number[] };

/** A per-op row per declaration: the backend's sum over its children, which
 *  are indices into the phase's row list. */
export type DeclarationRow = {
  declaration: Declaration & { stream: string; axis: string };
  path: string;
  op: string;
  children: number[];
  histogram: HistogramEntry[];
  layers: number;
  flops_per_rank: number;
  access_bytes_per_rank: number;
  t_compute_s: number;
  t_memory_s: number;
  t_comm_s: number;
  t_sol_s: number;
  t_op_s: number;
  count_per_step: number;
  /** The summed FLOPs over the summed access bytes (backend). */
  arithmetic_intensity?: number | null;
  arithmetic_intensity_formula?: FormulaFields | null;
  /** The parent's measured time as the backend's one measured formula
   *  states it (`Folded.subtotal`: a share only when every term under it is
   *  measured). Never a sum of the children made here. */
  t_measured_s: number | null;
  shape_in?: unknown;
  shape_out?: unknown;
};

/** The key a declaration groups under. */
export function declarationKey(d: Declaration): string {
  return `${d.layer_path.join(".")}:${d.assignment_index ?? ""}:${d.base_template}`;
}

/** `1–8` — the history lengths a histogram spans. */
export function visibleSpan(h: { visible: number }[]): string {
  if (!h.length) return "";
  const lo = h[0].visible;
  const hi = h[h.length - 1].visible;
  return lo === hi ? `${lo}` : `${lo}–${hi}`;
}
