import type { ReactNode } from "react";
import { tbl } from "../../DataTable";
import { binaryUnit, escapeHtml, siPrefix } from "../../format";
import { tipHandlers } from "../../Tip";
import { formulaHTML, type FormulaFields } from "../../tooltip";

export type MetricKind = "bytes" | "flops" | "count" | "time";
export type MetricGroup = "stored" | "prefill" | "decode";

export type MetricCol<Row> = {
  key: string;
  label: string;
  kind: MetricKind;
  group: MetricGroup;
  get: (row: Row) => number | null | undefined;
  /** Small print under the value (weight precision / source). */
  note?: (row: Row) => string | null | undefined;
  /** Backend-owned badge next to the value (decode KV "dominant"). */
  mark?: (row: Row) => string | null | undefined;
  /** Formula popover for the value, in the shared formulaHTML shape. */
  formula?: (row: Row) => FormulaFields | null | undefined;
};

export type SummaryRow = {
  key: string;
  label: string;
  color: string;
  marks?: string;
  aside?: ReactNode;
};

const GROUP_LABEL: Record<MetricGroup, string> = {
  stored: "Stored",
  prefill: "Prefill / step",
  decode: "Decode / step",
};

/** One scale for an entire column. The unit lives in the header and every
 *  value keeps the same decimal width, so magnitudes scan vertically. */
type ColumnFormat = {
  unit: string;
  fmt: (v: number | null | undefined) => string;
};

const EMPTY_COLUMN: ColumnFormat = {
  unit: "",
  fmt: (v) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(2)),
};

function columnFormat(
  kind: MetricKind,
  vals: (number | null | undefined)[],
): ColumnFormat {
  const finite = vals.filter((v): v is number => v != null && Number.isFinite(v));
  if (finite.length === 0) return EMPTY_COLUMN;
  const maxAbs = Math.max(...finite.map(Math.abs));

  let scale = 1;
  let unit = "";
  if (kind === "time") {
    if (maxAbs >= 1) unit = "s";
    else if (maxAbs >= 1e-3) { scale = 1e-3; unit = "ms"; }
    else { scale = 1e-6; unit = "µs"; }
  } else if (kind === "bytes") {
    // Bytes in binary units, as every other page states memory (f879d8564):
    // the KV table below this one says 224.00 MiB for what read 234.88 MB here.
    ({ scale, unit } = binaryUnit(maxAbs));
  } else {
    const si = siPrefix(maxAbs);
    scale = si.scale;
    unit = kind === "flops" ? `${si.prefix}FLOPs` : si.prefix;
  }

  return {
    unit,
    fmt: (v) => (v == null || !Number.isFinite(v) ? "—" : (v / scale).toFixed(2)),
  };
}

function CellInner<Row>({
  col, src, text,
}: {
  col: MetricCol<Row>; src: Row; text: string;
}) {
  const note = col.note?.(src);
  const mark = col.mark?.(src);
  const f = col.formula?.(src);
  // The number and its mark never split. The note wraps only where the cell
  // lets it: desktop table cells are nowrap (the table scrolls), the stacked
  // phone rows give the value a bounded column.
  return (
    <span className="break-words">
      <span className="whitespace-nowrap">
        {f ? (
          <span
            className="underline decoration-dotted"
            {...tipHandlers(formulaHTML(f))}
          >
            {text}
          </span>
        ) : text}
        {mark ? (
          <span className="pg-dominant ml-1 text-10 text-amber-700 font-sans font-medium">
            ({mark})
          </span>
        ) : null}
      </span>
      {note ? (
        <span
          className="ml-1 text-10 text-gray-400 font-sans font-normal"
          {...tipHandlers(`<div style="white-space:pre-wrap">${escapeHtml(note)}</div>`)}
        >
          ({note})
        </span>
      ) : null}
    </span>
  );
}

/** One row per compared entry, columns grouped stored / prefill / decode.
 *  Every numeric cell carries `data-value` = the backend number so tests can
 *  match the payload without parsing the display string. The desktop table is
 *  max-content wide and scrolls instead of squeezing columns; below 640 px each
 *  entry folds into three stacked key-value groups. The desktop table stays
 *  hand-built on the shared `tbl` tokens: tests pin `data-col`/`data-unit` on
 *  the column `<th>`s, which DataTable's header API cannot carry. */
export function SummaryTable<Row extends SummaryRow>({
  rows, cols, of, footnotes,
}: {
  rows: Row[];
  cols: MetricCol<Row>[];
  of?: (row: Row) => Row;
  footnotes?: Partial<Record<MetricGroup, string>>;
}) {
  const groups = (["stored", "prefill", "decode"] as MetricGroup[])
    .map((g) => ({ g, cols: cols.filter((c) => c.group === g) }))
    .filter((x) => x.cols.length);
  const srcOf = (row: Row) => (of ? of(row) : row);
  const formats: Record<string, ColumnFormat> = {};
  for (const c of cols) {
    formats[c.key] = columnFormat(c.kind, rows.map((row) => c.get(srcOf(row))));
  }
  const groupEdge = (groupIndex: number) =>
    groupIndex === 0 ? "" : "border-l border-gray-200";
  return (
    <div>
      <div className={`hidden sm:block ${tbl.wrapBare}`}>
        <table className={`${tbl.table} min-w-max`}>
          <thead>
            <tr className="text-left text-gray-500">
              <th className="px-1 py-1" />
              {groups.map(({ g, cols: gc }, gi) => (
                <th key={g} className={`px-1 py-1 font-medium ${groupEdge(gi)}`} colSpan={gc.length}>
                  {GROUP_LABEL[g]}
                </th>
              ))}
            </tr>
            <tr className="text-left text-gray-400">
              <th className="px-1 py-1">entry</th>
              {groups.flatMap(({ cols: gc }, gi) => gc.map((c, ci) => (
                <th
                  key={c.key}
                  data-col={c.key}
                  data-unit={formats[c.key].unit || undefined}
                  className={`px-1 py-1 font-normal text-right ${ci === 0 ? groupEdge(gi) : ""}`}
                >
                  <span className="block whitespace-nowrap">{c.label}</span>
                  <span className="block font-mono text-10 text-gray-400 whitespace-nowrap">
                    {formats[c.key].unit || "\u00A0"}
                  </span>
                </th>
              )))}
            </tr>
          </thead>
          <tbody className="font-mono text-gray-800">
            {rows.map((row) => {
              const src = srcOf(row);
              return (
                <tr key={row.key} data-entry={row.key}>
                  <td className="px-1 py-1 font-sans whitespace-nowrap">
                    <span className="inline-block w-2 h-2 rounded-sm mr-1 align-middle" style={{ background: row.color }} />
                    <span data-entry-label>{row.label}</span>
                    {row.marks ? <span className="text-gray-400"> {row.marks}</span> : null}
                    {row.aside ? <span className="ml-1 font-sans font-normal text-11 text-gray-500 whitespace-nowrap">({row.aside})</span> : null}
                  </td>
                  {groups.flatMap(({ cols: gc }, gi) => gc.map((c, ci) => {
                    const v = c.get(src);
                    return (
                      <td
                        key={c.key}
                        className={`px-1 py-1 text-right whitespace-nowrap tabular-nums ${ci === 0 ? groupEdge(gi) : ""}`}
                        data-col={c.key}
                        data-value={v == null ? "" : String(v)}
                      >
                        <CellInner col={c} src={src} text={formats[c.key].fmt(v)} />
                      </td>
                    );
                  }))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="sm:hidden space-y-3">
        {rows.map((row) => {
          const src = srcOf(row);
          return (
            <div key={row.key} data-entry={row.key} className="rounded-lg border border-gray-200 p-3">
              <div className="font-sans text-xs mb-1">
                <span className="inline-block w-2 h-2 rounded-sm mr-1 align-middle" style={{ background: row.color }} />
                <span data-entry-label className="font-mono text-gray-800">{row.label}</span>
                {row.marks ? <span className="text-gray-400"> {row.marks}</span> : null}
                {row.aside ? <span className="ml-1 font-sans font-normal text-11 text-gray-500 break-words">({row.aside})</span> : null}
              </div>
              {groups.map(({ g, cols: gc }) => (
                <div key={g} className="mt-2">
                  <div className="text-11 text-gray-500 font-medium mb-1">{GROUP_LABEL[g]}</div>
                  {gc.map((c) => {
                    const v = c.get(src);
                    return (
                      <div
                        key={c.key}
                        className="flex items-baseline justify-between gap-3 py-0.5 text-xs"
                        data-col={c.key}
                        data-value={v == null ? "" : String(v)}
                      >
                        <span className="text-gray-500 shrink-0">
                          {c.label}
                          {formats[c.key].unit ? (
                            <span className="ml-1 font-mono text-10">
                              ({formats[c.key].unit})
                            </span>
                          ) : null}
                        </span>
                        <span className="font-mono text-gray-800 text-right min-w-0 tabular-nums">
                          <CellInner col={c} src={src} text={formats[c.key].fmt(v)} />
                        </span>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          );
        })}
      </div>
      {footnotes != null && groups.some(({ g }) => footnotes[g]) && (
        <div className="mt-2 space-y-0.5 text-11 text-gray-500">
          {groups.map(({ g }) => (
            footnotes[g] ? (
              <div key={g} data-footnote={g}>
                {GROUP_LABEL[g]}: {footnotes[g]}
              </div>
            ) : null
          ))}
        </div>
      )}
    </div>
  );
}
