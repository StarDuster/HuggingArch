/** Greedy label placement for ECharts scatter / line-end labels.
 *
 *  ECharts re-runs labelLayout on legend click / resize. A new pass is
 *  the first time a (seriesIndex, dataIndex) pair repeats — NOT "the
 *  composite key decreased". Series are not visited in seriesIndex
 *  order, so a monotonic key cleared `placed` mid-pass and families
 *  stacked on top of each other.
 *  Slots stay inside optional [top, bottom] so they never leave the plot.
 */

type Box = { x1: number; y1: number; x2: number; y2: number };
type Host = { x: number; y: number; width: number; height: number };
type Align = "left" | "right" | "center";
type VAlign = "top" | "middle" | "bottom";
type Cand = { x: number; y: number; align: Align; vAlign: VAlign };

export type LabelLayoutParams = {
  rect: Host;
  labelRect: { x: number; y: number; width: number; height: number };
  dataIndex?: number;
  seriesIndex?: number;
  /** Wrapped label text (`\\n`). Height fallback uses the line count. */
  text?: string;
};

/** The plot box labels stay inside, in chart pixels. `left` / `right` are
 *  optional: a chart that knows its width passes them, so a label at the
 *  right edge flips left instead of running off the card. */
export type LabelBounds = { top: number; bottom: number; left?: number; right?: number };

function overlaps(a: Box, b: Box): boolean {
  return !(a.x2 < b.x1 || a.x1 > b.x2 || a.y2 < b.y1 || a.y1 > b.y2);
}

function boxAt(x: number, y: number, w: number, h: number, align: Align, vAlign: VAlign): Box {
  const x1 = align === "right" ? x - w : align === "center" ? x - w / 2 : x;
  const y1 = vAlign === "bottom" ? y - h : vAlign === "top" ? y : y - h / 2;
  return { x1, y1, x2: x1 + w, y2: y1 + h };
}

function makePassGate() {
  const seen = new Set<string>();
  return (params: LabelLayoutParams, placed: Box[]) => {
    const key = `${params.seriesIndex ?? 0}:${params.dataIndex ?? 0}`;
    if (seen.has(key)) {
      placed.length = 0;
      seen.clear();
    }
    seen.add(key);
  };
}

function labelHeight(params: LabelLayoutParams): number {
  const lines = 1 + (params.text ? (params.text.match(/\n/g) || []).length : 0);
  return Math.max(params.labelRect.height, 12 * lines);
}

// Nowhere fits: collapse the label onto its own host — `fontSize: 0` makes the
// text invisible and gives it a zero-size box (overlaps nothing), and the guide
// line scatter re-derives via `_updateLabelLine` (it ignores our
// `labelLinePoints`) degenerates to zero length. Bare `hideOverlap: true` would
// NOT hide it: ECharts runs that pass only among hideOverlap-flagged labels, so
// overlap with a successfully placed label never hides the failed one.
function hideLabel(cx: number, cy: number) {
  return { x: cx, y: cy, fontSize: 0, hideOverlap: true };
}

function clampY(y: number, h: number, bounds?: LabelBounds): number {
  if (!bounds) return y;
  const lo = bounds.top + h / 2;
  const hi = bounds.bottom - h / 2;
  if (hi < lo) return (bounds.top + bounds.bottom) / 2;
  return Math.min(Math.max(y, lo), hi);
}

function inBounds(box: Box, bounds?: LabelBounds): boolean {
  if (!bounds) return true;
  if (bounds.left != null && box.x1 < bounds.left - 0.5) return false;
  if (bounds.right != null && box.x2 > bounds.right + 0.5) return false;
  return box.y1 >= bounds.top - 0.5 && box.y2 <= bounds.bottom + 0.5;
}

/** A bubble's own box: its symbol, as ECharts lays it out. */
function hostBox(host: Host): Box {
  return { x1: host.x, y1: host.y, x2: host.x + host.width, y2: host.y + host.height };
}

/** Labels for a scatter: beside their bubble, clear of every other label and,
 *  where room allows, of every other bubble.
 *
 *  `bubbles` outlives one layout pass: each pass records every bubble's box in
 *  it, so the next pass knows the bubbles a label meets before their own
 *  labels are laid out. The chart runs one more pass once drawn
 *  (`ScatterChart`); until then a label avoids the bubbles laid out before it.
 *  A label with no spot clear of bubbles takes one clear of labels only — a
 *  name over a neighbour's bubble beats no name. */
export function makeScatterLabelLayout(bounds?: LabelBounds,
                                       bubbles: Map<string, Box> = new Map()) {
  const placed: Box[] = [];
  const gate = makePassGate();
  const GAP = 6;
  return (params: LabelLayoutParams) => {
    gate(params, placed);
    const key = `${params.seriesIndex ?? 0}:${params.dataIndex ?? 0}`;
    const host = params.rect;
    bubbles.set(key, hostBox(host));
    const others = [...bubbles].filter(([k]) => k !== key).map(([, b]) => b);
    const w = Math.max(params.labelRect.width, 8);
    const h = labelHeight(params);
    const cx = host.x + host.width / 2;
    const cy = host.y + host.height / 2;
    const r = host.width / 2;
    const near: Cand[] = [
      { x: cx + r + GAP, y: cy, align: "left", vAlign: "middle" },
      { x: cx - r - GAP, y: cy, align: "right", vAlign: "middle" },
      { x: cx, y: cy - r - GAP, align: "center", vAlign: "bottom" },
      { x: cx, y: cy + r + GAP, align: "center", vAlign: "top" },
      { x: cx + r + GAP, y: cy - h, align: "left", vAlign: "middle" },
      { x: cx + r + GAP, y: cy + h, align: "left", vAlign: "middle" },
      { x: cx - r - GAP, y: cy - h, align: "right", vAlign: "middle" },
      { x: cx - r - GAP, y: cy + h, align: "right", vAlign: "middle" },
    ];
    const step = h + 1;
    const far: Cand[] = [];
    for (let k = 1; k <= 24; k++) {
      far.push(
        { x: cx + r + GAP, y: cy - k * step, align: "left", vAlign: "middle" },
        { x: cx + r + GAP, y: cy + k * step, align: "left", vAlign: "middle" },
        { x: cx - r - GAP, y: cy - k * step, align: "right", vAlign: "middle" },
        { x: cx - r - GAP, y: cy + k * step, align: "right", vAlign: "middle" },
      );
    }
    const fits = (c: Cand, avoid: Box[]): Box | null => {
      const y = clampY(c.y, h, bounds);
      const b = boxAt(c.x, y, w, h, c.align, c.vAlign);
      return !avoid.some((p) => overlaps(p, b)) && inBounds(b, bounds) ? b : null;
    };
    let chosen: Cand | null = null;
    let box: Box | null = null;
    // Clear of labels and bubbles, near then far; else clear of labels only,
    // beside the bubble.
    for (const [cands, avoid] of [
      [near, [...placed, ...others]], [far, [...placed, ...others]], [near, placed],
    ] as const) {
      for (const c of cands) {
        const b = fits(c, avoid as Box[]);
        if (b) {
          chosen = { ...c, y: clampY(c.y, h, bounds) };
          box = b;
          break;
        }
      }
      if (box) break;
    }
    if (!chosen || !box) return hideLabel(cx, cy);
    placed.push(box);
    const guide = Math.hypot(chosen.x - cx, chosen.y - cy) > 18;
    return {
      x: chosen.x,
      y: chosen.y,
      align: chosen.align,
      verticalAlign: chosen.vAlign,
      hideOverlap: false,
      ...(guide
        ? { labelLinePoints: [[cx, cy], [(cx + chosen.x) / 2, (cy + chosen.y) / 2], [chosen.x, chosen.y]] }
        : {}),
    };
  };
}

/** A line's name beside its last point, inside the plot: to the right of
 *  it where the plot has room there (most lines end before the axis does,
 *  and past a line's end no line of its own runs), else to the left.
 *
 *  A name that would overlap one already placed is hidden, not moved: a
 *  moved name needs a guide line back to its point, and guide lines across
 *  the plot read as data. The legend and the tooltip still name every line.
 *  The right side is tried only when `bounds.right` says where the plot
 *  ends. */
export function makeLineEndLabelLayout(bounds?: LabelBounds) {
  const placed: Box[] = [];
  const gate = makePassGate();
  return (params: LabelLayoutParams) => {
    gate(params, placed);
    const host = params.rect;
    const w = Math.max(params.labelRect.width, 8);
    const h = labelHeight(params);
    const cx = host.x + host.width / 2;
    const cy = host.y + host.height / 2;
    const sides: [number, Align][] = [
      ...(bounds?.right != null ? [[cx + host.width / 2 + 6, "left"] as [number, Align]] : []),
      [cx - host.width / 2 - 6, "right"],
    ];
    // At the point, or nudged by up to one line — still beside its own
    // line end, so no guide line is needed. Further than that, hidden.
    for (const [x, align] of sides) {
      for (const dy of [0, -h / 2, h / 2, -h, h]) {
        const y = clampY(cy + dy, h, bounds);
        const b = boxAt(x, y, w, h, align, "middle");
        if (!placed.some((p) => overlaps(p, b)) && inBounds(b, bounds)) {
          placed.push(b);
          return { x, y, align, verticalAlign: "middle" as const, hideOverlap: false };
        }
      }
    }
    return hideLabel(cx, cy);
  };
}
