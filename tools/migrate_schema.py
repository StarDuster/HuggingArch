"""Migrate benchmark CSVs forward to the current schema, filling what the
older shape could not record.

Every column a version adds is a fact the rows before it never carried, and
none of them is recoverable after the fact:

  v3  `cross_check` — whether the latency was checked against the fp32
      reference. The oracle OOMs at the biggest cells and the harness measured
      those anyway, correctly but silently, so any given v2 row may or may not
      have been vetted.
  v4  `cross_check_rel_err` / `cross_check_cosine` — the numbers behind that
      verdict, computed and discarded by v3.
  v5  `kernel_symbol` — what the device actually ran. Requires a GPU and a
      profiler attached at measurement time; a finished CSV cannot be asked.
  v6  grouped-GEMM `n_tokens` — old jobs measured every padded expert row, so
      their physical row count is recoverable exactly as E × s_query.
  v7  `run_id` / `box` / `measured_load_factor` — which sweep produced the
      row, which box profile measured it, and the routing load it saw. All
      three are observations taken at measurement time; a finished CSV holds
      none of them.

So a migrated row is stamped `unknown` and left with EMPTY values, never
`passed` and never a fabricated 0 or plausible symbol. `unknown` and empty both
say "this corpus predates the record"; anything else would be a claim about an
observation that never happened, on exactly the rows where it is least likely
to have (the big attention and GEMM cells). Rows re-measured under the current
schema get real values.

Idempotent: a file already carrying every column is left alone.

    python tools/migrate_schema.py --dry-run <csv files>
    python tools/migrate_schema.py --apply   /root/results_all/*.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference import benchmarks as _benchmarks
# The engine lives with the schema so the harness's append path runs the same
# migration (`kernel_bench.emit.RowWriter`); re-exported for this CLI and its
# tests.
from backend.inference.benchmark_migration import FILL, FILLED, migrate_text

__all__ = ["FILL", "FILLED", "main", "migrate_text"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    paths: List[str] = []
    for pat in args.csvs:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])

    changed = already = failed = 0
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
            out = migrate_text(text)
        except Exception as e:
            print(f"  FAIL {p}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if out == text:
            already += 1
            continue
        changed += 1
        n = out.count("\n") - 1
        print(f"  {'would migrate' if args.dry_run else 'migrated'} "
              f"{os.path.basename(p)}: {n:,} rows")
        if args.apply:
            with open(p, "w", encoding="utf-8") as f:
                f.write(out)

    print(f"\n{changed} file(s) migrated to schema v"
          f"{_benchmarks.KNOWN_SCHEMA_VERSION}, {already} already there"
          + (f", {failed} failed" if failed else ""))
    if changed and FILLED:
        filled = ", ".join(f"{c}={FILL[c]!r}" for c in sorted(FILLED))
        print(f"  filled {filled} — each column's own word for what a row "
              f"that predates it truthfully says (see FILL); a finished CSV "
              f"cannot be asked for an observation it never recorded")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
