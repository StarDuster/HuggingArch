import { lazy, Suspense } from "react";
import type { EChartsOption, EChartsType } from "echarts";
import { ErrorBoundary } from "../ErrorBoundary";
import { useNearViewport } from "./useNearViewport";

export { AXIS, chartFileName, interactiveBits } from "./options";

const Impl = lazy(() => import("./HaChartImpl"));

/** Lazy so the 1 MB ECharts chunk is not on the first-paint critical path.
 *  The page chrome (nav, form) renders; this slot shows a one-line pending. */
export function HaChart({
  option,
  height,
  onEvents,
  onInstance,
  merge,
}: {
  option: EChartsOption;
  height: number;
  /** Merge an update into the chart rather than replace it: for a chart
   *  that restyles the same series in place (a pinned curve), so an element
   *  under the mouse is not swapped out from under it. Series are matched
   *  by `id`; one the update no longer has is removed. */
  merge?: boolean;
  onEvents?: Record<string, (params: unknown) => void>;
  /** The live instance, once (kept across option updates) — for zr-level
    interaction (click-anywhere, cursor) the series events cannot express. */
  onInstance?: (chart: EChartsType) => void;
}) {
  const [ref, near] = useNearViewport();
  if (!near) return <div ref={ref} style={{ height }} />;
  return (
    <ErrorBoundary compact height={height}>
    <Suspense
      fallback={
        <div className="text-xs text-gray-400 text-center py-8" style={{ height }}>
          Drawing…
        </div>
      }
    >
      <Impl option={option} height={height} onEvents={onEvents} onInstance={onInstance} merge={merge} />
    </Suspense>
    </ErrorBoundary>
  );
}
