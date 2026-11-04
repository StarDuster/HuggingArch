import type { EChartsOption } from "echarts";

/** Shared axis chrome — matches the old SVG charts' gray ticks. */
export const AXIS = {
  axisLine: { lineStyle: { color: "#cbd5e1" } },
  axisTick: { show: true, lineStyle: { color: "#cbd5e1" } },
  axisLabel: { color: "#6b7280", fontSize: 10 },
  splitLine: { lineStyle: { color: "#f3f4f6" } },
  nameTextStyle: { color: "#9ca3af", fontSize: 11 },
};

/** Toolbox (zoom / reset / save) + inside wheel-zoom. Restore is double-click
 *  equivalent. Brush-zoom is the toolbox dataZoom, not a homemade drag layer. */
/** A saved image's file name from the chart's title: "Decode TPOT — batch
 *  size = 1" → "huggingarch-decode-tpot-batch-size-1". */
export function chartFileName(title: string): string {
  const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return slug ? `huggingarch-${slug}` : "huggingarch-chart";
}

export function interactiveBits(opts?: {
  slider?: boolean;
  /** The chart's title; the saved image is named after it. */
  name?: string;
}): Pick<EChartsOption, "toolbox" | "dataZoom"> {
  const zoom: NonNullable<EChartsOption["dataZoom"]> = [
    { type: "inside", filterMode: "none" },
  ];
  if (opts?.slider) zoom.push({ type: "slider", height: 18, bottom: 4 });
  return {
    toolbox: {
      right: 8,
      top: 0,
      itemSize: 14,
      feature: {
        dataZoom: { title: { zoom: "Zoom", back: "Back" } },
        restore: { title: "Reset" },
        saveAsImage: { title: "Save", pixelRatio: 2, name: chartFileName(opts?.name ?? "") },
      },
    },
    dataZoom: zoom,
  };
}

/** Every chart's tooltip stays inside the chart's box, so on a phone it is
 *  never cut off at the viewport edge. A chart may still say otherwise. */
export function confineTooltip(option: EChartsOption): EChartsOption {
  const tip = option.tooltip;
  if (!tip || Array.isArray(tip)) return option;
  return { ...option, tooltip: { confine: true, ...tip } };
}
