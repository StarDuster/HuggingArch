"""Rows of an implementation whose own bench code changed.

A row prices the launch its provider built at the library version on its
identity. When the provider's OWN code changes — its `build`, its
`convert_inputs`, the inputs it is handed — nothing on the row says so, and
every old row of that implementation becomes doubtful whatever its outcome:
a measurement timed the old launch, and a failure failed on it (and counts
as tried for `--skip-existing` / `lookup.covered`, so leaving it blocks the
re-measurement the fix was for). `kernel_bench/ARCHITECTURE.md` calls for
purging them in the same merge, through the existing revocation channels:
the results CSVs locally, with the removed rows kept as a
`bench_purge_<date>_<reason>.csv` record, and `store.delete_rows` by id in
the lake.

Which rows go is named on the command line, never kept here: the
implementations by `--impl`, and where the change is narrower, the cells by
`--cell` (a Python expression over the row's own dims) and the outcomes by
`--outcome`.

    python -m tools.purge_impl_rows --impl torch_gather_sparse_gqa
    python -m tools.purge_impl_rows --impl torch_dense_mqa_logits \\
        --cell 'B * s_query * s_kv * (n_indexer_heads * 6 + 4) > 4 << 30'
    python -m tools.purge_impl_rows --lake --impl torch_qsa_sparse_gated
    ... --apply --reason 'the timed einsum now broadcasts the KV head'

Dry run unless `--apply`, and `--apply` needs `--reason`: a gap in the
corpus nobody can read a year out is indistinguishable from a defect.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.unserved_rows import (FRESH_SECONDS, RECORDS_DIR, _json_column,
                                 _record, _rows, append_record, default_trees,
                                 result_files)


def _selected(row: Dict[str, Any], impls: List[str], outcomes: List[str],
              version: Optional[str], cell: Optional[str]) -> bool:
    """Whether this row is one the named change revokes."""
    # A row is keyed by its implementation's `impl_id`: the kernel_impl, then
    # `@` and the deployment parameters it folds in (`Provider.impl_id`). The
    # separator was introduced after rows had already been published without
    # it, so the recorded name is classified by the registry rather than cut
    # at a character — otherwise this revokes nothing for a kernel whose rows
    # carry the older spelling.
    from kernel_bench.providers import catalog_kernel
    if catalog_kernel(row.get("kernel_impl") or "") not in impls:
        return False
    if outcomes and (row.get("outcome") or "measured") not in outcomes:
        return False
    if version and (row.get("framework_version") or "") != version:
        return False
    if cell is None:
        return True
    dims = _json_column(row.get("shape_json"))
    try:
        # The cell's own axes as names, so the threshold reads as the
        # provider's own arithmetic does.
        return bool(eval(cell, {"__builtins__": {}}, dict(dims)))
    except NameError:
        return False       # an axis this cell does not bind: not that cell
    except Exception as error:
        raise SystemExit(f"--cell {cell!r} on dims {dims}: {error}") from error


def _report(counts: Counter, total: int, scanned: int, what: str) -> None:
    print(f"{what}: {scanned} rows scanned, {total} selected")
    for label, count in counts.most_common():
        print(f"  {count:>7}  {label}")


def _label(row: Dict[str, Any]) -> str:
    return (f"{row.get('framework', '')}:{row.get('kernel_impl', '')} "
            f"{row.get('framework_version', '')} "
            f"[{row.get('outcome') or 'measured'}]")


def _purge_lake(args, record: str) -> Tuple[int, int]:
    from tools import lake_rows
    doomed: List[Tuple[str, Dict[str, Any]]] = []
    counts: Counter = Counter()
    scanned = foreign = 0
    # Off a read-only copy of the catalog (`tools.lake_rows`), never by
    # paging the service: over the whole lake that ran it out of memory.
    # The impls are asked of SQL as well as of `_selected`, so a lake of
    # millions of rows does not cross into Python to be rejected there.
    for row in lake_rows.rows(
            "split_part(m.kernel_impl, '@', 1) IN (SELECT unnest(?::VARCHAR[]))",
            [list(args.impl)]):
        scanned += 1
        flat = {k: ("" if v is None else v) for k, v in row.items()}
        if not _selected(flat, args.impl, args.outcome, args.framework_version,
                         args.cell):
            continue
        if flat.pop("owner") != lake_rows.OWNER:
            foreign += 1           # the service deletes only its owner's
            continue
        counts[_label(flat)] += 1
        doomed.append((flat["id"], flat))
    _report(counts, len(doomed), scanned,
            "lake" + (f" ({foreign} more another owner holds, not deletable "
                      f"here)" if foreign else ""))
    if not (args.apply and doomed):
        return len(doomed), 0
    append_record(record, [
        {**row, "source_csv": "lake", "row_id": rid,
         "purge_reason": args.reason} for rid, row in doomed])
    print(f"recorded what was removed in {record}")
    deleted = lake_rows.delete([rid for rid, _row in doomed])
    print(f"  deleted {deleted}")
    return len(doomed), deleted


def _purge_files(args, record: str) -> int:
    files = result_files(args.dir or default_trees())
    if not files:
        print("no result CSVs found", file=sys.stderr)
        return 1
    counts: Counter = Counter()
    scanned = 0
    planned: List[Tuple[str, List[str], List[Dict[str, str]],
                        List[Tuple[int, Dict[str, str]]]]] = []
    for path in files:
        fields, rows = _rows(path)
        if not rows or "kernel_impl" not in (fields or []):
            continue
        keep: List[Dict[str, str]] = []
        removed: List[Tuple[int, Dict[str, str]]] = []
        for line, row in enumerate(rows, start=2):
            scanned += 1
            if _selected(row, args.impl, args.outcome,
                         args.framework_version, args.cell):
                counts[_label(row)] += 1
                removed.append((line, row))
            else:
                keep.append(row)
        if removed:
            planned.append((path, fields, keep, removed))
    _report(counts, sum(counts.values()), scanned, f"{len(files)} files")
    # FAIL CLOSED: the harness appends to the CSV it is measuring into, and
    # a rewrite moved over it loses those rows.
    busy = [path for path, *_ in planned
            if time.time() - os.path.getmtime(path) < FRESH_SECONDS]
    if busy:
        print(f"ABORT: {len(busy)} affected file(s) written in the last "
              f"{FRESH_SECONDS}s (the harness is appending); nothing "
              f"rewritten:", file=sys.stderr)
        for path in busy:
            print(f"    {path}", file=sys.stderr)
        return 3
    if not args.apply:
        return 0
    records: set = set()
    for path, fields, keep, removed in planned:
        # In RECORDS_DIR, named for THIS purge: not beside the tree, where
        # the next tool to glob `<tree>/*.csv` reads it back as results, and
        # not under the shared `unserved` name `_record` defaults to, which
        # belongs to the other trigger.
        records.add(_record(path, fields, removed, args.reason, record))
        # Written beside and moved into place: a reader (the upload loop)
        # sees the old file or the new one, never half of one.
        temporary = f"{path}.rewriting"
        with open(temporary, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(keep)
        os.replace(temporary, path)
    for where in sorted(records):
        print(f"recorded what was removed in {where}")
    print(f"rewrote {len(planned)} files")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--impl", action="append", default=[], required=True,
                        help="a `kernel_impl` whose bench code changed "
                             "(repeatable)")
    parser.add_argument("--outcome", action="append", default=[],
                        help="only these outcomes (`measured`, `oom`, "
                             "`error`, …); default every one, because a "
                             "failure of the old build is as doubtful as a "
                             "measurement of it and counts as tried")
    parser.add_argument("--framework-version", default=None,
                        help="only rows measured at this library version")
    parser.add_argument("--cell", default=None,
                        help="a Python expression over the row's dims "
                             "(`s_query * n_topk_keys > 2**26`): only cells "
                             "it selects, for a change that only alters the "
                             "launch past a threshold")
    parser.add_argument("--dir", action="append", default=[],
                        help="a results tree (repeatable); default: every "
                             "tree in `unserved_rows.DEFAULT_TREES`")
    parser.add_argument("--lake", action="store_true",
                        help="the bench lake instead of files: rows read off "
                             "a read-only catalog copy, deletes through the "
                             "local service (tools.lake_rows)")
    parser.add_argument("--reason", default="",
                        help="what goes in the record's `purge_reason` on "
                             "every removed row: WHAT changed in the build "
                             "and the evidence (a commit, a measured "
                             "number). Required with --apply")
    parser.add_argument("--record", default=None,
                        help="where to append the purge record (default: "
                             "<RECORDS_DIR>/bench_purge_<date>_<reason>.csv, "
                             "outside every results tree)")
    parser.add_argument("--apply", action="store_true",
                        help="delete the rows listed")
    args = parser.parse_args(argv)
    if args.apply and not args.reason:
        parser.error(
            "--apply needs --reason: the record is the only thing that "
            "tells a reader a year out that this gap is a cleanup and not "
            "a defect, and it has to say WHICH change revoked the rows")
    slug = "".join(ch if ch.isalnum() else "_"
                   for ch in (args.reason or "build_changed"))[:40].strip("_")
    record = args.record or os.path.join(
        RECORDS_DIR, f"bench_purge_{time.strftime('%Y-%m-%d')}_{slug}.csv")
    if args.lake:
        found, deleted = _purge_lake(args, record)
        if not args.apply:
            print("\n(dry run; --apply to delete)")
            return 0
        if deleted != found:
            print(f"FAILED: the lake deleted {deleted} of the {found} rows "
                  f"listed", file=sys.stderr)
            return 5
        return 0
    code = _purge_files(args, record)
    if not args.apply and code == 0:
        print("\n(dry run; --apply to delete)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
