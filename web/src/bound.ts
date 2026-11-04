// The term that set an op's time — compute, memory, comm or the launch
// floor (a kernel's shortest time, however little it does) — drawn one way
// everywhere it shows: the per-op table's bound column, the formula
// popover's verdict, the compare page's decode split.
import type { RowBound } from "./pages/ResultSummary";

export const BOUND_PILL: Record<RowBound, string> = {
  compute: "text-violet-700 bg-violet-50 border border-violet-200",
  memory: "text-emerald-700 bg-emerald-50 border border-emerald-200",
  comm: "text-sky-700 bg-sky-50 border border-sky-200",
  launch: "text-amber-700 bg-amber-50 border border-amber-200",
};

/** The same palette as text alone: a column of one term (FLOPs and
 *  t_compute are compute's) names itself in that term's colour. */
export const BOUND_TEXT: Record<RowBound, string> = {
  compute: "text-violet-700",
  memory: "text-emerald-700",
  comm: "text-sky-700",
  launch: "text-amber-700",
};

export function isBound(s: string | null | undefined): s is RowBound {
  return !!s && s in BOUND_PILL;
}
