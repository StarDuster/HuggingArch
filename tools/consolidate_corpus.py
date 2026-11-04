"""Merge measured CSVs into one corpus file, deduped on the row identity.

Measurement runs accumulate across directories and passes, and the same
(model, shape, dtype, gpu, phase, parallel, framework, version, kernel_impl)
gets re-measured — a later pass re-walking a shape an earlier one already
covered, or the same file written twice. The identity is what
`op_calibration` indexes on, so duplicates are not additional evidence; they
are the same cell repeated, and whichever copy `best_of` happens to see wins.

Dedup keeps the FASTEST p50 per identity, matching how the corpus is consumed:
the reference for an op is the best a library achieves at that shape, so a copy
inflated by a noisy neighbour or a cold cache should not displace a clean one.
Rows whose p50 can't be read keep the first occurrence.

    python tools/consolidate_corpus.py --dry-run /root/results*/*.csv
    python tools/consolidate_corpus.py -o /root/corpus.csv /root/results*/*.csv
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference import benchmarks as _benchmarks

# The row identity `op_calibration` keys on. Kept explicit (not derived from
# ALL_COLUMNS minus measurements) so adding a measurement column can't silently
# widen the key and stop dedup from firing.
IDENTITY = (
    "model", "spec_hash", "phase", "parallel", "framework", "framework_version",
    "gpu", "op", "node_spec_id", "shape_json", "dtype_env_json", "kernel_impl",
)


def _p50(row: Dict[str, str]) -> float:
    try:
        return float(row.get("p50_us") or "inf")
    except (TypeError, ValueError):
        return float("inf")


# Relative p50 spread above which a repeated measurement is a flag, not
# noise. Within-run sampling noise sits at a few percent; a fifth apart is
# one of slow-side contamination, a fast side that skipped the work, or a
# real box difference — the three things a report must name rather than
# let min-dedup
# bury.
SPREAD_FLAG_THRESHOLD = 0.2


def dedupe(rows: List[Dict[str, str]],
           *, spread_threshold: float = SPREAD_FLAG_THRESHOLD
           ) -> Tuple[List[Dict[str, str]], int, List[Dict]]:
    """(kept, n_duplicate_rows, spread_flags).

    Keeps the FASTEST p50 per identity — the min envelope, the right answer
    to one-sided latency noise (contention and throttling only ever make a
    kernel slower). But a large spread is not noise, and min must not
    silently bury it: every identity whose p50 spread (max−min)/min exceeds
    `spread_threshold` comes back as a flag naming the spread and the
    run_ids on each side."""
    best: Dict[Tuple, Dict[str, str]] = {}
    by_key: Dict[Tuple, List[Dict[str, str]]] = collections.defaultdict(list)
    for r in rows:
        k = tuple(r.get(c, "") for c in IDENTITY)
        by_key[k].append(r)
        if k not in best or _p50(r) < _p50(best[k]):
            best[k] = r

    dupes = sum(len(v) - 1 for v in by_key.values() if len(v) > 1)
    flags: List[Dict] = []
    for k, grp in by_key.items():
        if len(grp) < 2:
            continue
        p50s = [_p50(r) for r in grp]
        lo, hi = min(p50s), max(p50s)
        if lo <= 0 or lo == float("inf"):
            continue
        spread = (hi - lo) / lo
        if spread <= spread_threshold:
            continue
        flags.append({
            "op": k[7], "node_spec_id": k[8], "gpu": k[6],
            "kernel_impl": k[11], "shape_json": k[9],
            "min_us": lo, "max_us": hi, "spread": spread,
            "run_ids": sorted({r.get("run_id", "") for r in grp} - {""}),
        })
    flags.sort(key=lambda f: -f["spread"])
    return list(best.values()), dupes, flags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("-o", "--output", default=None, help="merged corpus path")
    ap.add_argument("--dry-run", action="store_true", help="report only")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.output:
        ap.error("--output is required unless --dry-run")

    paths: List[str] = []
    for pat in args.csvs:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])

    rows: List[Dict[str, str]] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                rows.extend(_benchmarks.parse_rows(f.read())[1])
        except Exception as e:
            print(f"  SKIP {p}: {type(e).__name__}: {e}")

    kept, dupes, flags = dedupe(rows)
    print(f"  read {len(rows):,} rows from {len(paths)} files")
    print(f"  duplicate rows dropped : {dupes:,}")
    print(f"  unique identities kept : {len(kept):,}")
    if flags:
        # The envelope took the min of these — correctly — but a spread this
        # wide is contamination, a run that skipped the work, or a real
        # box difference, and the
        # report has to say so rather than let min bury it.
        print(f"\n  SPREAD FLAGS: {len(flags)} identitie(s) measured apart "
              f"by more than {SPREAD_FLAG_THRESHOLD:.0%} in p50:")
        for f in flags[:20]:
            runs = ",".join(f["run_ids"]) or "?"
            print(f"    {f['op']} {f['node_spec_id']} {f['kernel_impl']} "
                  f"{f['gpu']}: {f['min_us']:.1f}–{f['max_us']:.1f}us "
                  f"(+{f['spread']:.0%}) runs={runs}")
        if len(flags) > 20:
            print(f"    … and {len(flags) - 20} more")
    if args.dry_run:
        return 0

    text = _benchmarks.emit_csv(list(_benchmarks.ALL_COLUMNS), kept)
    _benchmarks.validate_text(text, source=args.output)   # refuse to write junk
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nwrote {len(kept):,} rows → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
