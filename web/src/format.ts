/** vanilla util/format.js `escapeHtml`. */
export function escapeHtml(s: unknown): string {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] || c),
  );
}

/** The decimal (SI) prefixes FLOP counts and bytes in motion are scaled by.
 *  A byte count reads in one of two units by what it is: an amount held —
 *  HBM, weights, KV, a checkpoint on disk — in binary (`formatGiB`), so a
 *  card's "180 GiB" and the weights subtracted from it share a unit; bytes
 *  moved per step (traffic, collectives) in decimal (`formatBytes`), the
 *  unit of the GB/s they are divided by. */
const SI_PREFIXES: readonly (readonly [number, string])[] = [
  [1e15, "P"], [1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "K"],
];

/** The largest SI prefix `magnitude` reaches (1 and "" below a thousand). */
export function siPrefix(magnitude: number): { scale: number; prefix: string } {
  const a = Math.abs(magnitude);
  for (const [scale, prefix] of SI_PREFIXES) if (a >= scale) return { scale, prefix };
  return { scale: 1, prefix: "" };
}

/** `v` in its own SI prefix: `formatSI(2.13e12, "FLOPs")` → "2.13 TFLOPs". */
export function formatSI(v: number, unit: string, digits = 2): string {
  if (!Number.isFinite(v)) return "—";
  const { scale, prefix } = siPrefix(v);
  return `${(v / scale).toFixed(digits)} ${prefix}${unit}`;
}

/** Tick labels for a value axis: ONE prefix for the whole axis, picked from
 *  its largest value, so ticks read 0 / 1 / 2 / 3 TFLOPs instead of mixing
 *  units. The prefixes are powers of ten, so the axis's round ticks stay round
 *  numbers in the chosen unit. */
export function siAxisLabel(maxAbs: number, unit: string): (v: number) => string {
  const { scale, prefix } = siPrefix(maxAbs);
  return (v: number) => (v === 0 ? "0" : `${Number((v / scale).toPrecision(6))} ${prefix}${unit}`);
}

export function formatFlops(flops: number | null | undefined): string {
  if (flops == null || !Number.isFinite(flops)) return "—";
  if (Math.abs(flops) < 1e6) return flops.toFixed(0);
  return formatSI(flops, "");
}

/** A count (parameters) in its own SI prefix — the prefix an axis of counts
 *  uses (`siAxisLabel(max, "")`), so a value and its axis agree: 187.11 M,
 *  2.77 T. Below a thousand, the count itself. */
export function formatCount(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const { scale, prefix } = siPrefix(n);
  return scale === 1 ? String(Math.round(n * 100) / 100) : `${(n / scale).toFixed(2)} ${prefix}`;
}

/** A token count, exact, with thousands separators: 4,096. */
export function formatTokens(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US");
}

/** A dimension's bound value as the spec writes it: exact, unseparated
 *  (d_model=7168). A dim is a symbol's value, not a quantity to round. */
export function formatDim(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return String(n);
}

export function formatNumber(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(2) + "K";
  return String(Math.round(n * 100) / 100);
}

// --- Chart numbers -------------------------------------------------------
// One reading for every chart's axes, end labels and tooltips: at most three
// significant digits, trailing zeros dropped, the unit stepped up instead of
// a long mantissa — 100B, 1T, 10T; 1K, 128K, 1M; 50 ms, 700 s.

function sig3(x: number): string {
  return String(Number(x.toPrecision(3)));
}

/** A count on a chart (parameters, tokens/s): 539M, 30.5B, 10T. */
export function fmtCompact(n: number): string {
  if (!Number.isFinite(n)) return "";
  const a = Math.abs(n);
  for (const [scale, unit] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]] as const) {
    if (a >= scale) return `${sig3(n / scale)}${unit}`;
  }
  return sig3(n);
}

/** A context length on a chart: a whole number of 1024-token blocks reads
 *  binary as the sweep grids are written (1K, 128K, 1M); any other value (a
 *  zoomed axis's tick) reads compact (32.4K). */
export function fmtCtx(n: number): string {
  if (!Number.isFinite(n)) return "";
  if (n >= 1024 && n % 1024 === 0) return fmtSeqLabel(n);
  return fmtCompact(n);
}

/** A duration on a chart: 3.2 µs, 14 ms, 1.1 s, 700 s. */
export function fmtDuration(s: number): string {
  if (!Number.isFinite(s)) return "";
  if (s === 0) return "0";
  const a = Math.abs(s);
  if (a < 1e-3) return `${sig3(s * 1e6)} µs`;
  if (a < 1) return `${sig3(s * 1e3)} ms`;
  return `${sig3(s)} s`;
}

/** Same labels as vanilla `InferFmt.fmtSeqLabel`. */
export function fmtSeqLabel(seq: number): string {
  if (seq >= 1048576) {
    const m = seq / 1048576;
    return m === Math.floor(m) ? `${m}M` : `${m.toFixed(1)}M`;
  }
  if (seq >= 1024) return `${Math.round(seq / 1024)}K`;
  return String(seq);
}

/** Bytes moved — traffic, collectives — in decimal units. */
export function formatBytes(bytes: number): string {
  return formatSI(bytes, "B");
}

/** vanilla `formatShape` — empty / N/A stay N/A. */
export function formatShape(shapeStr: string | null | undefined): string {
  if (!shapeStr || shapeStr === "N/A") return "N/A";
  return shapeStr;
}

/** The largest binary prefix `magnitude` reaches (1 and "" below 1 KiB). */
function binaryPrefix(magnitude: number): { scale: number; prefix: string } {
  const a = Math.abs(magnitude);
  for (const [scale, prefix] of [[1024 ** 4, "Ti"], [1024 ** 3, "Gi"], [1024 ** 2, "Mi"], [1024, "Ki"]] as const) {
    if (a >= scale) return { scale, prefix };
  }
  return { scale: 1, prefix: "" };
}

/** Bytes held — HBM, weights, KV, checkpoint — in binary units. */
export function formatGiB(bytes: number, digits = 2): string {
  if (!Number.isFinite(bytes)) return "—";
  const { scale, prefix } = binaryPrefix(bytes);
  if (scale === 1 && Number.isInteger(bytes)) return `${bytes} B`;
  return `${(bytes / scale).toFixed(digits)} ${prefix}B`;
}

/** The binary unit an axis of bytes held is drawn in: the prefix of the
 *  axis max. A chart plots `bytes / scale`, so the axis picks round ticks in
 *  that unit (8 GiB, 16 GiB) — ticks chosen in raw bytes and relabelled come
 *  out as 18.6265 GiB. */
export function binaryUnit(maxAbs: number): { scale: number; unit: string } {
  const { scale, prefix } = binaryPrefix(maxAbs);
  return { scale, unit: `${prefix}B` };
}

export function fmtTime(s: number | null | undefined): string {
  if (s == null || !(s > 0)) return "—";
  if (s < 1e-3) return `${(s * 1e6).toFixed(1)} µs`;
  if (s < 1) return `${(s * 1e3).toFixed(1)} ms`;
  return `${s.toFixed(2)} s`;
}

/** `{ symbol: meaning }` from GET /dim-glossary — vanilla util/format.js. */
let _dimGlossary: Record<string, string> = {};

export function setDimGlossary(items: Record<string, string> | null | undefined) {
  _dimGlossary = items || {};
}

export function dimMeaning(symbol: string | null | undefined): string {
  if (!symbol) return "";
  const raw = _dimGlossary[symbol] || "";
  return raw.replace(/\*\*/g, "").replace(/`/g, "");
}

/** Consecutive indices to "1-5, 7". */
export function compressRanges(indices: number[]): string {
  if (!indices || indices.length === 0) return "";
  const sorted = [...indices].sort((a, b) => a - b);
  const ranges: string[] = [];
  let start = sorted[0];
  let prev = sorted[0];
  for (let i = 1; i < sorted.length; i++) {
    const cur = sorted[i];
    if (cur === prev + 1) {
      prev = cur;
      continue;
    }
    ranges.push(start === prev ? `${start}` : `${start}-${prev}`);
    start = cur;
    prev = cur;
  }
  ranges.push(start === prev ? `${start}` : `${start}-${prev}`);
  return ranges.join(", ");
}
