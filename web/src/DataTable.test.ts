// What every sortable table on the site relies on and no page test can see:
// the order `sortRows` puts rows in, and the direction a header click picks.
import { describe, expect, test } from "vitest";

import { nextSort, sortRows, type ColumnDef, type SortState } from "./DataTable";

type Row = { id: string; n?: number | null | string; s?: string };

const byN: ColumnDef<Row> = { key: "n", align: "right", sortValue: (r) => r.n };
const byS: ColumnDef<Row> = { key: "s", sortValue: (r) => r.s };
const unsortable: ColumnDef<Row> = { key: "pick", sortValue: null };
const columns = [byN, byS, unsortable];

const ids = (rows: Row[]) => rows.map((r) => r.id).join(",");
const asc = (key: string): SortState => ({ key, dir: "asc" });
const desc = (key: string): SortState => ({ key, dir: "desc" });

describe("sortRows", () => {
  test("ties keep the input order in both directions", () => {
    const rows: Row[] = [
      { id: "a", n: 5 }, { id: "b", n: 1 }, { id: "c", n: 5 },
      { id: "d", n: 1 }, { id: "e", n: 5 },
    ];
    expect(ids(sortRows(rows, columns, asc("n")))).toBe("b,d,a,c,e");
    // Descending reverses the columns' values, never the tied rows among
    // themselves: `a,c,e` stay in the order the caller gave them.
    expect(ids(sortRows(rows, columns, desc("n")))).toBe("a,c,e,b,d");
  });

  test("missing values sort last in both directions", () => {
    const rows: Row[] = [
      { id: "null", n: null }, { id: "three", n: 3 }, { id: "empty", n: "" },
      { id: "one", n: 1 }, { id: "nan", n: NaN }, { id: "undef", n: undefined },
      { id: "two", n: 2 },
    ];
    // null, "", NaN and undefined are all "no value", and they go after the
    // values in ascending order AND in descending order — a reader sorting a
    // column to find its largest row must not get a screenful of blanks.
    expect(ids(sortRows(rows, columns, asc("n"))))
      .toBe("one,two,three,null,empty,nan,undef");
    expect(ids(sortRows(rows, columns, desc("n"))))
      .toBe("three,two,one,null,empty,nan,undef");
  });

  test("numbers compare as numbers, not as their text", () => {
    const rows: Row[] = [{ id: "a", n: -2 }, { id: "b", n: -10 }, { id: "c", n: 9 }, { id: "d", n: 10 }];
    // Text order (even with numeric collation) would put -2 before -10.
    expect(ids(sortRows(rows, columns, asc("n")))).toBe("b,a,c,d");
  });

  test("text compares with numeric collation, so x2 precedes x10", () => {
    const rows: Row[] = [{ id: "a", s: "x10" }, { id: "b", s: "x2" }, { id: "c", s: "x1" }];
    expect(ids(sortRows(rows, columns, asc("s")))).toBe("c,b,a");
    expect(ids(sortRows(rows, columns, desc("s")))).toBe("a,b,c");
  });

  test("a column with no sortValue, or an unknown key, leaves rows as they are", () => {
    const rows: Row[] = [{ id: "a", n: 5 }, { id: "b", n: 1 }];
    expect(sortRows(rows, columns, asc("pick"))).toBe(rows);
    expect(sortRows(rows, columns, asc("nope"))).toBe(rows);
    expect(sortRows(rows, columns, null)).toBe(rows);
  });

  test("the caller's array is not reordered in place", () => {
    const rows: Row[] = [{ id: "a", n: 5 }, { id: "b", n: 1 }];
    sortRows(rows, columns, asc("n"));
    // Pages hold their row list across renders and slice pages out of it.
    expect(ids(rows)).toBe("a,b");
  });
});

describe("nextSort", () => {
  test("a numeric (right-aligned) column opens high-to-low, text opens A-to-Z", () => {
    expect(nextSort(byN, null)).toEqual({ key: "n", dir: "desc" });
    expect(nextSort(byS, null)).toEqual({ key: "s", dir: "asc" });
  });

  test("three clicks: open, reverse, clear", () => {
    const one = nextSort(byN, null);
    expect(one).toEqual({ key: "n", dir: "desc" });
    const two = nextSort(byN, one);
    expect(two).toEqual({ key: "n", dir: "asc" });
    expect(nextSort(byN, two)).toBeNull();
    // And the same cycle on a text column, whose first click is the other way.
    const a = nextSort(byS, null);
    expect(a).toEqual({ key: "s", dir: "asc" });
    const b = nextSort(byS, a);
    expect(b).toEqual({ key: "s", dir: "desc" });
    expect(nextSort(byS, b)).toBeNull();
  });

  test("clicking another column opens that column's own first direction", () => {
    expect(nextSort(byS, desc("n"))).toEqual({ key: "s", dir: "asc" });
    expect(nextSort(byN, asc("s"))).toEqual({ key: "n", dir: "desc" });
  });
});
