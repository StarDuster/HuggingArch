"""Which spans the launch gate has never looked at.

`kernel_bench.launch_trace` reads the ops a timed call dispatches and refuses
a row whose launch is not the cell's — the door that catches "this row priced
one thing and measured another". It can only see what the dispatcher sees, so
a kernel behind a pybind entry point into compiled code dispatches nothing
observable and the door has nothing to compare. That is not a failure; it is a
BLIND SPOT, and until now it was indistinguishable from a span the door
guards successfully, because "the trace agreed" and "there was no trace" took
the same branch.

This enumerates it from the rows: per (op, framework), how many rows carry a
`launch_signature` and how many do not. A span where every framework row
carries none is a span the gate has never guarded.

Only rows MEASURED since the recorder landed are judged. An older row has the
column empty because nothing was recording, and the schema version cannot
tell that apart from "observed nothing" — the migration relabels an old row
to the current version and fills the new column empty, so on a migrated
corpus every row claims to be current. The date is the only honest filter.

    python -m tools.launch_blind_spots                       # every local mirror
    python -m tools.launch_blind_spots --dir <results dir>
    python -m tools.launch_blind_spots --op gated_delta_rule

It matters where the next campaign operates. The gate's blind spot is exactly
"a framework moved work across a launch boundary inside compiled code", and
that is where the quantise/dequantise boundary lives: vLLM launches its fp8
activation quantise separately while its silu-and-mul variant folds it into
the epilogue, and `gated_delta_rule` is the same shape — vLLM moved the q/k
l2norm into an earlier launch while sglang keeps it inside, and no row can
disagree because neither kernel dispatches anything the recorder can read.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.unserved_rows import default_trees, result_files


#: The day the recorder landed (`8bbe7d0f3`, kernel_bench/launch_trace.py).
#: Rows measured ON that day are mixed and are NOT judged: a sweep was
#: running while the code changed, so the same provider on the same box has
#: traced and untraced rows that day for a reason that is about the clock
#: rather than the kernel (torch SDPA on 10-21: 49,477 untraced against
#: 9,953 traced; on 10-22: 128 against 13,844).
#: A row measured before it has the column empty because nothing was
#: recording, not because nothing was observable — and the SCHEMA VERSION
#: cannot tell those apart, because `tools/migrate_schema.py` relabels an old
#: row to the current version and fills the new column empty. So the date the
#: row was measured is the only honest filter, and the number below is a
#: statement about this repo's history rather than about any row.
_RECORDER_LANDED = "2026-10-21"

#: A library whose rows are the harness's own reference implementations. They
#: dispatch eager ops, so they are always traced — which is why a span can
#: look covered while every FRAMEWORK row in it is unguarded.
_BASELINES = ("torch",)




def tally(files, op: str = "") -> Tuple[Dict[Tuple[str, str], List[int]], int]:
    """{(op, framework): [rows, traced]} and how many rows predate the column."""
    seen: Dict[Tuple[str, str], List[int]] = collections.defaultdict(
        lambda: [0, 0])
    too_old = 0
    for path in files:
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if "launch_signature" not in row:
                        break                  # a file older than the column
                    if op and row.get("op") != op:
                        continue
                    if (row.get("date") or "") <= _RECORDER_LANDED:
                        too_old += 1
                        continue
                    key = (row.get("op") or "", row.get("framework") or "")
                    seen[key][0] += 1
                    if (row.get("launch_signature") or "").strip():
                        seen[key][1] += 1
        except OSError as error:
            print(f"  {path}: {error}", file=sys.stderr)
    return seen, too_old


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", action="append", default=[])
    parser.add_argument("--op", default="", help="one op only")
    parser.add_argument("--min-rows", type=int, default=20,
                        help="ignore a span with fewer rows than this")
    args = parser.parse_args(argv)

    files = result_files(args.dir or default_trees())
    if not files:
        print("no result CSVs found", file=sys.stderr)
        return 1
    seen, too_old = tally(files, args.op)
    if not seen:
        print("no rows carry the launch_signature column yet")
        return 0

    # An op is BLIND when no framework row in it was ever traced. Baselines
    # are counted apart: they are always traced and would mask the answer.
    by_op: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    for (op, framework), (rows, traced) in seen.items():
        slot = by_op[op]
        if framework in _BASELINES:
            slot[2] += rows
            slot[3] += traced
        else:
            slot[0] += rows
            slot[1] += traced

    blind = {op: s for op, s in by_op.items()
             if s[0] >= args.min_rows and s[1] == 0}
    print(f"{len(files)} files | {sum(v[0] for v in seen.values()):,} rows "
          f"judged | {len(by_op)} ops | {too_old:,} measured on or before "
          f"the day the recorder landed ({_RECORDER_LANDED}), not judged")
    print(f"\nBLIND — every framework row untraced ({len(blind)} ops):")
    print(f"  {'op':<34}{'framework rows':>15}{'baseline traced':>17}")
    for op, s in sorted(blind.items(), key=lambda kv: -kv[1][0]):
        print(f"  {op:<34}{s[0]:>15,}{s[3]:>17,}")
    partial = {op: s for op, s in by_op.items()
               if s[0] >= args.min_rows and 0 < s[1] < s[0]}
    print(f"\nPARTLY traced ({len(partial)} ops): a span the door sees for "
          f"some implementations and not others")
    print(f"  {'op':<34}{'framework rows':>15}{'traced':>9}")
    for op, s in sorted(partial.items(), key=lambda kv: -(kv[1][0] - kv[1][1]))[:15]:
        print(f"  {op:<34}{s[0]:>15,}{s[1]:>9,}")
    print("\nA blind span is not a wrong one — it is one where a kernel that "
          "moved work across\na launch boundary could not be contradicted by "
          "its own row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
