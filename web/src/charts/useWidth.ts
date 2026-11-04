import { useCallback, useRef, useState } from "react";

/** The rendered width of the element the returned callback ref is attached
 *  to, kept current on resize — what a chart needs to size its legend rows or
 *  know where its plot ends before ECharts lays it out. 0 until measured; an
 *  element that mounts later (a chart that had nothing to draw at first) is
 *  measured when it appears. */
export function useWidth<T extends HTMLElement>() {
  const [width, setWidth] = useState(0);
  const ro = useRef<ResizeObserver | null>(null);
  const ref = useCallback((el: T | null) => {
    ro.current?.disconnect();
    ro.current = null;
    if (!el) return;
    setWidth(el.clientWidth);
    ro.current = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.current.observe(el);
  }, []);
  return [ref, width] as const;
}

/** Rows a plain ECharts legend wraps into at `width` px: each item is its
 *  icon, its text (≈ `charPx` per character) and the gap. An estimate a
 *  chart reserves room by, so the legend sits below the axis name instead of
 *  over it. */
export function legendRows(names: string[], width: number, opts?: {
  iconPx?: number; gapPx?: number; charPx?: number; usable?: number;
}): number {
  if (!names.length) return 0;
  const icon = opts?.iconPx ?? 16;
  const gap = opts?.gapPx ?? 14;
  const char = opts?.charPx ?? 6.2;
  const avail = Math.max(120, (width || 600) * (opts?.usable ?? 0.92));
  let rows = 1;
  let used = 0;
  for (const n of names) {
    const w = icon + 5 + n.length * char + gap;
    if (used > 0 && used + w > avail) {
      rows += 1;
      used = 0;
    }
    used += w;
  }
  return rows;
}
