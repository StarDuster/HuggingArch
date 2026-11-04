import { useEffect, useLayoutEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";

/** Shared table style tokens. Complex tables that cannot go through
 *  `DataTable` (form grids, editor matrices) import these directly so every
 *  table on the site still shares one visual source. */
export const tbl = {
  // A table taller than the screen scrolls in its own frame, so its header
  // can stay frozen at the frame's top (a header cannot stick to the page
  // from inside a frame that scrolls sideways).
  wrap: "overflow-auto max-h-[80vh] rounded-xl border border-gray-200 bg-white shadow-sm",
  wrapBare: "overflow-auto max-h-[80vh]",
  table: "w-full text-xs text-gray-700",
  // Headers are often identifiers (dim symbols, op and component names):
  // keep their case — `d_kv_lora` uppercased is not a dim anyone declared.
  // The site's warm accent (the amber of its tabs and metric cards), frozen
  // (sticky) and opaque so rows scroll under it, not through it. The text
  // colour is the thead's and cells inherit it, so a column that names
  // itself in a colour (`headerClassName`) never fights a default for it.
  thead: "sticky top-0 z-20 bg-gradient-to-b from-[#fff8ee] to-orange-100 text-11 text-orange-900",
  // A group's label sits centred over its columns on a rule that spans
  // them; the groups are told apart by the column rules (`<colgroup>`).
  groupHeadCell: "px-2 pt-2.5 text-11 font-semibold tracking-wide text-orange-800",
  // The rule under the header is drawn, not a border: a collapsed table's
  // borders belong to its grid and scroll away from a frozen header.
  headCell: "relative shadow-[inset_0_-2px_0_var(--color-orange-400)] px-3 py-2 font-semibold whitespace-nowrap",
  row: "border-b border-gray-100 last:border-0 hover:bg-gray-50/60",
  // Cells hold one line: a short value (a model id, `RTX-4090`, `0% faster`)
  // must not break because a long text column wants the width. A cell of
  // prose or a long joined list wraps instead (`wrapCell`, whole — adding
  // `whitespace-normal` on top of `cell` loses to its nowrap in the CSS).
  cell: "px-3 py-1.5 align-top whitespace-nowrap",
  wrapCell: "px-3 py-1.5 align-top whitespace-normal min-w-[14rem]",
  // A cell whose content marks its own break points (a shape list breaks
  // between tensors, each tensor held whole): no minimum, no white-space rule.
  flowCell: "px-3 py-1.5 align-top",
  // Every column is bounded, not only the ones marked `wrap`: a cell's
  // content is as wide as itself up to 22rem, and wraps past that. A short
  // value never breaks (its width is its own, so a long neighbour cannot
  // squeeze it), and a long one — a layer list, a joined set of names —
  // cannot push the table out of its frame because nobody marked it.
  capped: "inline-block w-max max-w-[22rem] whitespace-normal align-top [overflow-wrap:anywhere]",
  footRow: "border-t-2 border-gray-300 bg-gray-50 font-semibold text-gray-900",
  empty: "px-3 py-6 text-center text-gray-400",
} as const;

export type TableAlign = "left" | "right" | "center";

const alignClass: Record<TableAlign, string> = {
  left: "text-left",
  right: "text-right",
  center: "text-center",
};

export type ColumnDef<Row> = {
  key: string;
  header?: ReactNode;
  align?: TableAlign;
  /** Extra classes on every body cell; right-aligned columns also get
   *  `tabular-nums` automatically. */
  className?: string | ((row: Row, i: number) => string | undefined);
  headerClassName?: string;
  /** Prose or a long joined list: the column wraps (every other cell holds
   *  one line). */
  wrap?: boolean;
  /** Cell content; defaults to `row[key]`. */
  render?: (row: Row, i: number) => ReactNode;
  /** Merge this cell across N columns (the next N-1 columns are skipped). */
  colSpan?: (row: Row, i: number) => number;
  /** Extra attributes (e.g. `data-*` test hooks) on each body cell. */
  cellProps?: (row: Row, i: number) => Record<string, unknown>;
  /** What the column sorts by — the row's raw value, not its formatted
   *  text (`1.2 GB` must sort as bytes); its header becomes a sort toggle.
   *  Numbers compare as numbers, anything else as text; a missing value
   *  (null, undefined, "" or NaN) sorts last either way. `null`: the column
   *  has no order to sort by (a checkbox, an action button), or the table's
   *  row order is its meaning (a tree, a fixed metric layout). Required, so
   *  a new table says which of the two each column is. */
  sortValue: ((row: Row) => number | string | null | undefined) | null;
};

/** The column a table is sorted by, and which way. */
export type SortState = { key: string; dir: "asc" | "desc" } | null;

const missing = (v: unknown) =>
  v == null || v === "" || (typeof v === "number" && Number.isNaN(v));

/** `rows` in `sort`'s order (stable; missing values last). A table that
 *  pages sorts its whole list with this before slicing a page out.
 *
 *  Three properties, each pinned in `DataTable.test.ts` because each is one
 *  ordinary refactor from gone:
 *  - **missing values last in BOTH directions** — the `missing` branch is
 *    deliberately not multiplied by `sign`;
 *  - **ties keep the input order** — `|| a.i - b.i` is redundant as code
 *    (ES2019 requires `Array.prototype.sort` to be stable) and kept as the
 *    statement of intent, because the shape that breaks the property is
 *    factoring the sign out — `(c || a.i - b.i) * sign` reverses ties on a
 *    descending sort, and reads like simplification;
 *  - **the caller's array is not reordered** — the index wrapper copies. */
export function sortRows<Row>(rows: Row[], columns: ColumnDef<Row>[], sort: SortState): Row[] {
  // `sortValue` is checked ONCE. `find` used to require it as well, so the
  // same property had two gates and each was the other's spare: a mutation
  // of either alone changed nothing, and the contract test could only see
  // the pair. The `!sort` half of the guard stays — it is not a duplicate,
  // it is what narrows `sort` for `sort.dir` below, and `tsc` says so.
  const col = sort ? columns.find((c) => c.key === sort.key) : undefined;
  if (!sort || !col?.sortValue) return rows;
  const value = col.sortValue;
  const sign = sort.dir === "asc" ? 1 : -1;
  return rows
    .map((row, i) => ({ row, i, v: value(row) }))
    .sort((a, b) => {
      if (missing(a.v) || missing(b.v)) return Number(missing(a.v)) - Number(missing(b.v)) || a.i - b.i;
      const c = typeof a.v === "number" && typeof b.v === "number"
        ? a.v - b.v
        : String(a.v).localeCompare(String(b.v), undefined, { numeric: true });
      return c * sign || a.i - b.i;
    })
    .map((x) => x.row);
}

// A header click: a new column starts high-to-low when it holds numbers
// (right-aligned), A-to-Z otherwise; a second click reverses; a third clears.
// Exported for `DataTable.test.ts`: the three-click cycle is a contract every
// sortable table on the site shows, and the header button is its only other
// entry point.
export function nextSort<Row>(col: ColumnDef<Row>, cur: SortState): SortState {
  const first = col.align === "right" ? "desc" : "asc";
  if (cur?.key !== col.key) return { key: col.key, dir: first };
  return cur.dir === first ? { key: col.key, dir: first === "asc" ? "desc" : "asc" } : null;
}

export type HeaderGroup = {
  label: ReactNode;
  span: number;
  align?: TableAlign;
  className?: string;
};

type DataTableProps<Row> = {
  columns: ColumnDef<Row>[];
  rows: Row[];
  rowKey: (row: Row, i: number) => string | number;
  /** Optional super-header row above the column headers (grouped tables). */
  headerGroups?: HeaderGroup[];
  /** Custom row(s) appended at the end of the body — totals, summaries. */
  footer?: ReactNode;
  /** Replace the whole body (tree/fold tables keep their own row components). */
  renderBody?: () => ReactNode;
  /** Replace the `<tr>` for one row. */
  renderRow?: (row: Row, i: number) => ReactNode;
  empty?: ReactNode;
  loading?: boolean;
  hideHeader?: boolean;
  framed?: boolean;
  /** Row hover affordance — enable for rows that are clickable. */
  hoverable?: boolean;
  rowClassName?: (row: Row, i: number) => string | undefined;
  onRowClick?: (row: Row, i: number) => void;
  rowProps?: (row: Row, i: number) => Record<string, unknown>;
  className?: string;
  tableClassName?: string;
  /** Extra attributes (id, data-*) on the `<table>` element. */
  tableProps?: Record<string, unknown>;
  /** The sort, when the caller owns it (a paged table sorts its whole list
   *  with `sortRows` and passes the page). Without `onSort` the table keeps
   *  its own sort and orders `rows` itself. */
  sort?: SortState;
  onSort?: (sort: SortState) => void;
};

export function DataTable<Row>({
  columns,
  rows,
  rowKey,
  headerGroups,
  footer,
  renderBody,
  renderRow,
  empty,
  loading,
  hideHeader,
  framed = true,
  hoverable,
  rowClassName,
  onRowClick,
  rowProps,
  className,
  tableClassName,
  tableProps,
  sort: sortProp,
  onSort,
}: DataTableProps<Row>) {
  const [ownSort, setOwnSort] = useState<SortState>(null);
  const sort = onSort ? sortProp ?? null : ownSort;
  const setSort = onSort ?? setOwnSort;
  if (!onSort) rows = sortRows(rows, columns, ownSort);
  const colCount = columns.length || 1;

  // A column never narrows while the same table is on screen. Widths follow
  // content, so a new sort or page — other rows on screen — would otherwise
  // reflow every column and shake the table; each header keeps the widest it
  // has been as its minimum. A wrapping column is left out: it is the slack
  // that keeps the table inside its frame as the others hold their widths.
  // Only rows count: a loading or empty table is
  // one stretched placeholder cell, not column widths. A different set of
  // columns starts afresh.
  const tableRef = useRef<HTMLTableElement>(null);
  // Whether columns sit past the frame's right edge, not yet scrolled to.
  const frameRef = useRef<HTMLDivElement>(null);
  const [moreRight, setMoreRight] = useState(false);
  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;
    // A few pixels past the edge (borders, rounding) hide nothing worth a cue.
    const update = () => setMoreRight(
      frame.scrollWidth - frame.clientWidth - frame.scrollLeft > 12);
    update();
    frame.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(frame);
    if (tableRef.current) ro.observe(tableRef.current);
    return () => { frame.removeEventListener("scroll", update); ro.disconnect(); };
  }, []);
  const widest = useRef<{ columns: string; px: number[] }>({ columns: "", px: [] });
  const columnSet = columns.map((c) => c.key).join("\u0001");
  const showsRows = !loading && (rows.length > 0 || !!renderBody);
  // Widths the reader dragged a column edge to. The first drag freezes every
  // column at its laid-out width (fixed layout from then on, so a column can
  // also narrow — its content wraps rather than spill); a double-click on an
  // edge, or a new set of columns, hands the layout back to the content.
  const [dragged, setDragged] = useState<{ columns: string; px: number[] } | null>(null);
  const widths = dragged?.columns === columnSet ? dragged.px : null;
  const headCells = () =>
    Array.from(tableRef.current?.querySelectorAll<HTMLTableCellElement>(":scope > thead > tr:last-child > th") ?? []);
  const startResize = (i: number, e: ReactPointerEvent<HTMLSpanElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const start = widths ?? headCells().map((th) => th.getBoundingClientRect().width);
    const x0 = e.clientX;
    // The table never ends short of its frame: what a narrowed column gives
    // up goes to a wrapping column (the slack), else to the last one — never
    // back to the column being dragged.
    const frame = tableRef.current?.parentElement?.clientWidth ?? 0;
    const slack = columns.findIndex((c, k) => c.wrap && k !== i);
    const fill = slack >= 0 ? slack : columns.length - 1 === i ? columns.length - 2 : columns.length - 1;
    const move = (ev: PointerEvent) => {
      const px = [...start];
      px[i] = Math.max(48, start[i] + ev.clientX - x0);
      const short = frame - px.reduce((a, b) => a + b, 0);
      if (short > 0 && fill >= 0) px[fill] += short;
      setDragged({ columns: columnSet, px });
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.style.cursor = "";
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    document.body.style.cursor = "col-resize";
  };
  const resetWidths = () => {
    widest.current = { columns: "", px: [] };
    headCells().forEach((th) => { th.style.minWidth = ""; });
    setDragged(null);
  };
  // Pinned columns stay at the frame's left edge while the table scrolls
  // sideways; each keeps its place in the order (header groups stay whole)
  // and stops after the pinned columns before it. Applied to the drawn
  // cells, so a table that draws its own rows (`renderRow`, `renderBody`)
  // pins the same way. A cell merged across columns does not pin.
  const [pinned, setPinned] = useState<ReadonlySet<string>>(new Set());
  const togglePin = (key: string) =>
    setPinned((cur) => {
      const next = new Set(cur);
      if (!next.delete(key)) next.add(key);
      return next;
    });
  useLayoutEffect(() => {
    const table = tableRef.current;
    if (!table) return;
    const apply = () => {
      const ths = Array.from(table.querySelectorAll<HTMLTableCellElement>(":scope > thead > tr:last-child > th"));
      const lefts: (number | null)[] = [];
      let left = 0;
      let edge = -1;
      columns.forEach((c, i) => {
        if (!pinned.has(c.key)) { lefts.push(null); return; }
        lefts.push(left);
        left += ths[i]?.getBoundingClientRect().width ?? 0;
        edge = i;
      });
      const pin = (cell: HTMLTableCellElement, at: number) => {
        const l = cell.colSpan === 1 ? lefts[at] ?? null : null;
        if (l == null) {
          cell.removeAttribute("data-pinned");
          cell.style.position = "";
          cell.style.left = "";
        } else {
          cell.setAttribute("data-pinned", at === edge ? "edge" : "");
          // Inline: a cell's own `relative` would otherwise win.
          cell.style.position = "sticky";
          cell.style.left = `${l}px`;
        }
      };
      ths.forEach((th, i) => pin(th, i));
      table.querySelectorAll<HTMLTableRowElement>(":scope > tbody > tr, :scope > tfoot > tr").forEach((tr) => {
        let at = 0;
        for (const cell of Array.from(tr.cells)) {
          pin(cell, at);
          at += cell.colSpan;
        }
      });
    };
    apply();
    if (!pinned.size) return;
    // Widths follow content and the window: re-measure when the table does.
    const ro = new ResizeObserver(apply);
    ro.observe(table);
    return () => ro.disconnect();
  });
  useLayoutEffect(() => {
    const ths = tableRef.current?.querySelectorAll<HTMLTableCellElement>(":scope > thead > tr:last-child > th");
    if (!ths || !showsRows) return;
    if (widths) {
      ths.forEach((th) => { th.style.minWidth = ""; });
      return;
    }
    if (widest.current.columns !== columnSet) widest.current = { columns: columnSet, px: [] };
    const px = widest.current.px;
    ths.forEach((th, i) => {
      if (columns[i]?.wrap) return;
      px[i] = Math.max(px[i] ?? 0, th.getBoundingClientRect().width);
      th.style.minWidth = `${px[i]}px`;
    });
  });
  // A resized window lays the table out afresh: kept widths would otherwise
  // hold it wider than a narrowed page.
  useEffect(() => {
    const reset = () => {
      widest.current = { columns: "", px: [] };
      tableRef.current?.querySelectorAll<HTMLTableCellElement>(":scope > thead > tr > th")
        .forEach((th) => { th.style.minWidth = ""; });
    };
    window.addEventListener("resize", reset);
    return () => window.removeEventListener("resize", reset);
  }, []);
  let body: ReactNode;
  if (renderBody) {
    body = (
      <>
        {renderBody()}
        {footer}
      </>
    );
  } else if (loading) {
    body = (
      <tr>
        <td colSpan={colCount} className={tbl.empty}>
          {empty ?? "Loading…"}
        </td>
      </tr>
    );
  } else if (!rows.length) {
    body = (
      <tr>
        <td colSpan={colCount} className={tbl.empty}>
          {empty ?? "No data"}
        </td>
      </tr>
    );
  } else {
    body = (
      <>
        {rows.map((row, i) => {
          if (renderRow) return renderRow(row, i);
          const cls = [
            tbl.row,
            hoverable || onRowClick ? "hover:bg-gray-50" : "",
            onRowClick ? "cursor-pointer" : "",
            rowClassName?.(row, i) ?? "",
          ]
            .filter(Boolean)
            .join(" ");
          const cells: ReactNode[] = [];
          for (let c = 0; c < columns.length; c++) {
            const col = columns[c];
            const span = col.colSpan?.(row, i) ?? 1;
            const extra =
              typeof col.className === "function"
                ? col.className(row, i)
                : col.className;
            cells.push(
              <td
                key={col.key}
                colSpan={span > 1 ? span : undefined}
                className={[
                  col.wrap ? tbl.wrapCell : tbl.cell,
                  alignClass[col.align ?? "left"],
                  col.align === "right" ? "tabular-nums" : "",
                  extra ?? "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                {...col.cellProps?.(row, i)}
              >
                {(() => {
                  const content = col.render
                    ? col.render(row, i)
                    : ((row as Record<string, ReactNode>)[col.key] ?? null);
                  return col.wrap || content == null || content === "" ? content : <span className={tbl.capped} data-cap="">{content}</span>;
                })()}
              </td>,
            );
            c += span - 1;
          }
          return (
            <tr
              key={rowKey(row, i)}
              className={cls}
              onClick={onRowClick ? () => onRowClick(row, i) : undefined}
              {...rowProps?.(row, i)}
            >
              {cells}
            </tr>
          );
        })}
        {footer}
      </>
    );
  }
  return (
    <div className={["relative", className ?? ""].filter(Boolean).join(" ")}>
    <div ref={frameRef} className={framed ? tbl.wrap : tbl.wrapBare}>
      <table
        ref={tableRef}
        className={[tbl.table, tableClassName ?? ""].filter(Boolean).join(" ")}
        data-resized={widths ? "" : undefined}
        style={widths ? { tableLayout: "fixed", width: widths.reduce((a, b) => a + b, 0) } : undefined}
        {...tableProps}
      >
        {/* One <colgroup> per header group, ruled off from the one before:
         *  the rule runs down every row, the body's own included. */}
        {(headerGroups?.length ? headerGroups : [{ span: columns.length }]).reduce<{ at: number; out: ReactNode[] }>(
          (acc, g, gi) => {
            const cols = columns.slice(acc.at, acc.at + g.span).map((c, k) => (
              <col key={c.key} style={widths ? { width: widths[acc.at + k] } : undefined} />
            ));
            acc.out.push(<colgroup key={gi} className={gi > 0 ? "border-l border-gray-200" : undefined}>{cols}</colgroup>);
            acc.at += g.span;
            return acc;
          },
          { at: 0, out: [] },
        ).out}
        {hideHeader ? null : (
          <thead className={tbl.thead}>
            {headerGroups?.length ? (
              <tr>
                {headerGroups.map((g, gi) => (
                  <th
                    key={gi}
                    colSpan={g.span}
                    className={[
                      tbl.groupHeadCell,
                      alignClass[g.align ?? (g.span > 1 ? "center" : "left")],
                      g.className ?? "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <div className={`pb-1 ${g.label ? "border-b border-orange-300" : ""}`}>{g.label}</div>
                  </th>
                ))}
              </tr>
            ) : null}
            <tr>
              {columns.map((col, ci) => {
                const active = sort?.key === col.key ? sort.dir : null;
                const isPinned = pinned.has(col.key);
                return (
                  <th
                    key={col.key}
                    aria-sort={active ? (active === "asc" ? "ascending" : "descending") : undefined}
                    className={[
                      tbl.headCell,
                      "group/th",
                      alignClass[col.align ?? "left"],
                      col.headerClassName ?? "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    {col.sortValue ? (
                      <button
                        type="button"
                        data-sort={col.key}
                        onClick={() => setSort(nextSort(col, sort))}
                        className={`group inline-flex items-center gap-1 hover:text-gray-800 ${active ? "text-gray-800" : ""} ${col.align === "right" ? "flex-row-reverse" : ""}`}
                      >
                        {col.header ?? col.key}
                        {/* The arrow is drawn, not text: the header still reads as its
                            name. A fixed width, so marking a column does not reflow it. */}
                        <span aria-hidden className={`inline-block w-2.5 text-center ${
                          active === "asc" ? "text-amber-600 after:content-['▲']"
                            : active === "desc" ? "text-amber-600 after:content-['▼']"
                            : "text-gray-300 opacity-0 group-hover:opacity-100 after:content-['↕']"}`} />
                      </button>
                    ) : (col.header ?? col.key)}
                    {/* Pin: keep this column in view while scrolling sideways. */}
                    <button
                      type="button"
                      data-pin={col.key}
                      aria-pressed={isPinned}
                      title={isPinned ? "Unpin column" : "Pin column to the left"}
                      onClick={() => togglePin(col.key)}
                      className={`ml-1 inline-flex align-middle ${isPinned
                        ? "text-amber-600"
                        : "text-gray-400 opacity-0 hover:text-gray-700 group-hover/th:opacity-100 focus-visible:opacity-100"}`}
                    >
                      <svg aria-hidden viewBox="0 0 16 16" className="h-3 w-3" fill={isPinned ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
                        <path d="M6 2h4l-.5 4 2.5 2.5v1H4v-1L6.5 6z" />
                        <path d="M8 9.5V14" strokeLinecap="round" />
                      </svg>
                    </button>
                    {/* Drag to resize, double-click to fit the content again.
                        No text: a header reads as its name. */}
                    <span
                      aria-hidden
                      data-col-resize=""
                      onPointerDown={(e) => startResize(ci, e)}
                      onDoubleClick={resetWidths}
                      className="absolute -right-1 top-0 z-10 h-full w-2 cursor-col-resize select-none after:absolute after:left-1/2 after:top-1/4 after:h-1/2 after:w-px after:bg-gray-300 after:opacity-0 hover:after:opacity-100 hover:after:bg-amber-500"
                    />
                  </th>
                );
              })}
            </tr>
          </thead>
        )}
        <tbody>{body}</tbody>
      </table>
    </div>
      {/* Columns past the frame's right edge: a fade says the table goes on,
       *  where an overlay scrollbar (macOS, touch) shows nothing until used. */}
      {moreRight ? (
        <div aria-hidden data-more-right=""
          className="pointer-events-none absolute inset-y-0 right-0 w-10 rounded-r-xl bg-gradient-to-l from-gray-400/25 to-transparent" />
      ) : null}
    </div>
  );
}
