import ReactECharts from "echarts-for-react";
import type { EChartsOption, EChartsType } from "echarts";
import { confineTooltip } from "./options";

export default function HaChartImpl({
  option,
  height,
  onEvents,
  onInstance,
  merge,
}: {
  option: EChartsOption;
  height: number;
  merge?: boolean;
  onEvents?: Record<string, (params: unknown) => void>;
  onInstance?: (chart: EChartsType) => void;
}) {
  return (
    <ReactECharts
      option={confineTooltip(option)}
      style={{ height, width: "100%" }}
      notMerge={!merge}
      // Merging still drops series the update no longer has (by id).
      replaceMerge={merge ? ["series"] : undefined}
      lazyUpdate
      onEvents={onEvents}
      onChartReady={onInstance}
    />
  );
}
