"""Drop rows that re-measure a cell another model already covers.

Two model names can describe the same architecture — DeepSeek-V3.2 and
V3.2-Exp, Kimi-K2-Base and K2-Instruct, GLM-5.1 and GLM-5.2. Their specs
fingerprint identically, so a kernel measured under one name and under the
other is the SAME physical measurement recorded twice: same shape, same dtype,
same GPU, same kernel. Only the `model` column and the run-to-run noise differ.

The corpus loader treats that as a hard error — two wide keys canonicalizing to
one slim key is an ambiguity it refuses to resolve — so a corpus carrying both
copies does not load at all, which silently disables `--skip-existing` and makes
every later sweep re-measure everything.

Deduped by slim key, keeping the FASTEST copy: between two measurements of
one cell the faster one is the one less disturbed by whatever else touched
the card. The surviving row
keeps its own `model` value, so provenance is not invented; lookups for the
dropped model resolve through the slim key, which is what the slim key is for.

    python tools/dedup_isomorphic_rows.py --dry-run /root/results_all/*.csv
    python tools/dedup_isomorphic_rows.py --apply   /root/results_all/*.csv
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
from backend.inference.op_calibration import BenchmarkKey


def _p50(row: Dict[str, str]) -> float:
    try:
        return float(row.get("p50_us") or "nan")
    except (TypeError, ValueError):
        return float("nan")


def _slim(row: Dict[str, str]):
    """The row's canonical physical identity, or None if it cannot be built —
    an unparseable row is left alone rather than guessed at."""
    try:
        return BenchmarkKey.from_csv_row(row).to_slim()
    except Exception:
        return None


def winners(rows: List[Dict[str, str]]) -> Dict[object, Tuple[Tuple[int, float], str, int]]:
    """slim key -> (p50, source path, index) of the copy that survives.

    Ties break on (path, index) so a re-run of this tool over the same corpus
    drops the same rows — a dedup that depends on dict ordering would keep
    shuffling which model owns a cell.
    """
    best: Dict[object, Tuple[float, str, int]] = {}
    for path, idx, r in rows:
        s = _slim(r)
        if s is None:
            continue
        v = _p50(r)
        # An unusable p50 must not remove the identity from `best`. `partition`
        # keeps exactly the row `best` names, so an identity absent from it has
        # EVERY copy dropped — deleting coverage instead of deduplicating it.
        # Sorting NaN/≤0 to the back keeps a real measurement when one exists
        # and still leaves one row standing when none does.
        rank = (0, v) if (v == v and v > 0) else (1, 0.0)
        cur = best.get(s)
        if cur is None or (rank, path, idx) < cur:
            best[s] = (rank, path, idx)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="where dropped rows go (default: <csv dir>/.isomorphic_rows)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    paths: List[str] = []
    for pat in args.csvs:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])

    per_file: Dict[str, List[Dict[str, str]]] = {}
    flat: List[Tuple[str, int, Dict[str, str]]] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                rows = _benchmarks.parse_rows(f.read())[1]
        except Exception as e:
            print(f"  SKIP {p}: {type(e).__name__}: {e}")
            continue
        per_file[p] = rows
        flat.extend((p, i, r) for i, r in enumerate(rows))

    best = winners(flat)

    tot_keep = tot_drop = 0
    dropped_for: collections.Counter = collections.Counter()
    for p, rows in per_file.items():
        kept, dup = [], []
        for i, r in enumerate(rows):
            s = _slim(r)
            if s is not None and best.get(s, (None, None, None))[1:] != (p, i):
                dup.append(r)
                keeper = best[s]
                dropped_for[(r.get("model", "?"), os.path.basename(keeper[1]))] += 1
            else:
                kept.append(r)
        assert len(kept) + len(dup) == len(rows)
        tot_keep += len(kept)
        tot_drop += len(dup)
        if not dup:
            continue
        print(f"  DEDUP {os.path.basename(p)}: {len(dup)}/{len(rows)}")
        if not args.apply:
            continue
        out_dir = args.out_dir or os.path.join(os.path.dirname(p) or ".",
                                               ".isomorphic_rows")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, os.path.basename(p)), "w", encoding="utf-8") as f:
            f.write(_benchmarks.emit_csv(list(_benchmarks.ALL_COLUMNS), dup))
        with open(p, "w", encoding="utf-8") as f:
            f.write(_benchmarks.emit_csv(list(_benchmarks.ALL_COLUMNS), kept))

    verb = "would drop" if args.dry_run else "dropped"
    print(f"\n{verb} {tot_drop:,} rows already covered under another model name "
          f"| kept {tot_keep:,}")
    if dropped_for:
        print("  dropped model → surviving file:")
        for (m, keeper), n in dropped_for.most_common(10):
            print(f"    {m.split('/')[-1]:<28} → {keeper:<34} {n:>6,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
