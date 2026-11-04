// The per-op table's fold over the backend's module rows (`flops_by_module`):
// which rows a reader sees, by depth, folding and cost — no row is chosen by
// its op name.
import type { ModuleFoldRow } from "./pages/ResultSummary";

/** A table row's identity as the backend names it: the block invocation a
 *  template header stands for, every other row's aggregation_id. */
export function foldKey(row: ModuleFoldRow): string {
  return row.op === "<template>" ? `T::${row.template_id}` : row.aggregation_id ?? "";
}

export function foldDepth(row: ModuleFoldRow): number {
  return row.depth || 0;
}

export function hasFoldChildren(rows: ModuleFoldRow[], i: number): boolean {
  return i + 1 < rows.length && foldDepth(rows[i + 1]) > foldDepth(rows[i]);
}

export function visibleFold(
  rows: ModuleFoldRow[],
  folded: Set<string>,
): { row: ModuleFoldRow; i: number; kids: boolean }[] {
  const out: { row: ModuleFoldRow; i: number; kids: boolean }[] = [];
  let skipDeeperThan: number | null = null;
  // A row that costs nothing — no FLOPs, no traffic, no bytes on the wire,
  // no time (its subtree rolled up, the backend's `*_agg` columns) — is
  // pure layout plumbing (the views around MLA head splits and expert
  // routing) and is hidden at any depth. A row that costs something stays,
  // whatever its op: a concat or permute that moves bytes takes time.
  const costless = (row: ModuleFoldRow) =>
    !((row.flops_per_rank || 0) > 0)
    && !(((row.access_bytes_agg ?? row.access_bytes_per_rank) || 0) > 0)
    && !(((row.comm_bytes_agg ?? row.comm_bytes_per_rank) || 0) > 0)
    && !(((row.t_op_agg ?? row.t_op_s) || 0) > 0);
  rows.forEach((row, i) => {
    const d = foldDepth(row);
    if (skipDeeperThan != null && d > skipDeeperThan) return;
    skipDeeperThan = null;
    const kids = hasFoldChildren(rows, i);
    if (costless(row)) return;
    out.push({ row, i, kids });
    if (kids && folded.has(foldKey(row))) skipDeeperThan = d;
  });
  return out;
}
