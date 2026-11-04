"""Measured rows whose recorded peak is not the card's peak any more.

`peak_tflops_at_test` and `peak_bw_at_test` are what the GPU table said
when a row was measured, and they are what every utilization the row shows
is divided by (`mfu`, `mbu`, the SOL gate). They are provenance, not
measurement — the latency is the measurement — so when the table is
CORRECTED, as against merely edited, the rows that carry the old number are
showing a utilization of a peak the hardware never had.

That is what happened to 1,572 RTX-4090 rows: the bf16 column was the
vendor sheet's 165.2, computed at a 2.52 GHz boost clock this card clears
(it holds 2.73 GHz under a sustained GEMM, measured), so the column is now
178.9 and every row stamped with the old one reads 8.3% high — up to 105%
of a peak, which is the signal "this measurement is impossible" firing on
an arithmetic slip instead.

Leaving them is not neutral either: `op_calibration` compensates for a peak
that moved by scaling the measured LATENCY (`peak_at_test / current_peak`),
and the silicon did not get 8% faster, so until the rows are restamped
every estimate reading them is wrong in the other direction.

The new numbers come from the same two functions the emitter used
(`Env.peak_tflops` for the row's dtype, `Env.peak_bw_gbps`), so this
rewrites nothing by hand and nothing per SKU:

    python -m tools.restamp_peaks                       # every local mirror
    python -m tools.restamp_peaks --apply
    python -m tools.restamp_peaks --dir <results dir> --apply
    python -m tools.restamp_peaks --lake [--apply]

Order matters where the same rows are in several places: the BOX first,
then the mirror, then the lake — the mirror pulls with `rsync -a`, which
restores whatever the box still has.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.unserved_rows import default_trees, result_files


#: A file the harness touched this recently is being appended to, and
#: rewriting it in place would leave the appender writing into an orphaned
#: inode — its measurements lost. Skipped and counted, not raced.
_SETTLED_S = 600

#: A difference below this is the format's own rounding (the columns carry
#: four decimals), not a table that moved.
_SAME = 1e-3


def _peaks(gpu: str, dtype_env: Any) -> Tuple[Optional[float],
                                              Optional[float]]:
    """What the row's two peak columns would be written as today.

    `dtype_env` arrives as JSON text from a CSV and as a dict from the
    service, and the difference is not cosmetic: parsed as text a dict
    reprs with single quotes, `json.loads` refuses it, and an environment
    read as empty falls back to one dtype for every row — which is how
    18,955 H100 fp32 rows came to carry the bf16 peak and 5,965 fp8 rows
    the same, in a pass that was only supposed to correct the 4090's bf16
    column. So the shape is asked, never assumed.
    """
    from backend.inference.gpu_specs import (get_gpu, hbm_bytes,  # noqa: F401
                                             peak_flops_for_dtype)
    spec = get_gpu(gpu)
    if spec is None:
        return None, None
    if isinstance(dtype_env, Mapping):
        env = dict(dtype_env)
    else:
        try:
            env = json.loads(dtype_env or "{}")
        except json.JSONDecodeError as error:
            # Not a fallback: a row whose environment cannot be read has no
            # peak this can recompute, and guessing one is what broke the
            # first pass.
            raise SystemExit(
                f"cannot read a row's dtype_env ({dtype_env!r}): {error}"
            ) from error
    dtype = env.get("activation") or env.get("weight")
    if not dtype:
        raise SystemExit(f"a row carries no activation or weight dtype: {env!r}")
    try:
        flops = peak_flops_for_dtype(spec, str(dtype))
    except Exception:
        flops = None
    return (flops / 1e12 if flops else None,
            float(spec.mem_bw_gbps) if spec.mem_bw_gbps else None)


def _restamped(row: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """The row's corrected pair, or None where nothing moved."""
    tflops, bw = _peaks(row.get("gpu") or "", row.get("dtype_env_json") or "")
    out = []
    for column, value in (("peak_tflops_at_test", tflops),
                          ("peak_bw_at_test", bw)):
        was = row.get(column) or ""
        if value is None:
            out.append(was)
            continue
        want = f"{value:.4f}"
        if not was:
            out.append(was)                 # never stamped; not ours to invent
        elif abs(float(was) - value) <= _SAME:
            out.append(was)
        else:
            out.append(want)
    if out == [row.get("peak_tflops_at_test") or "",
               row.get("peak_bw_at_test") or ""]:
        return None
    return out[0], out[1]




def _lake(method: str, path: str, body=None):
    import urllib.request
    from pathlib import Path
    secret = Path("/tmp/ha_bench/bench_secret").read_text().strip()
    user = os.environ.get("HA_BENCH_USER",
                          "07a6e2a7-54b4-4e52-ba0d-21ce44f06b09")
    request = urllib.request.Request(
        "http://127.0.0.1:8100" + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"X-HA-Internal": secret, "X-HA-User": user,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode())


def _walk_lake(apply: bool) -> int:
    """Every row the lake holds, through the same correction."""
    patch: List[Dict[str, object]] = []
    moved: Counter = Counter()
    for upload in _lake("GET", "/api/bench/list")["uploads"]:
        ref = (f"run_id={upload['run_id']}" if upload.get("run_id")
               else f"upload_id={upload['upload_id']}")
        offset, total = 0, None
        while True:
            page = _lake("GET", f"/api/bench/rows?{ref}&limit=1000"
                                f"&offset={offset}")
            got = page.get("rows") or []
            total = int(page.get("total") or 0) if total is None else total
            for row in got:
                # The dict the service returns, NOT stringified: `_peaks`
                # reads the environment out of it.
                flat = {k: ("" if v is None else v if k == "dtype_env_json"
                            else str(v))
                        for k, v in row.items()}
                fix = _restamped(flat)
                if fix is None:
                    continue
                moved[(row.get("gpu"), flat.get("peak_tflops_at_test"),
                       fix[0])] += 1
                patch.append({"id": row["id"],
                              "peak_tflops_at_test": float(fix[0]),
                              "peak_bw_at_test": float(fix[1])})
            offset += len(got)
            if not got or offset >= (total or 0):
                break
    print(f"lake: {len(patch)} rows carry a peak the table has corrected")
    for (gpu, was, now), count in moved.most_common():
        print(f"  {count:>7}  {gpu}: {was} -> {now}")
    if apply and patch:
        done = 0
        for start in range(0, len(patch), 500):
            done += int(_lake("POST", "/api/bench/rows/restamp",
                              {"rows": patch[start:start + 500]}
                              ).get("updated") or 0)
        print(f"  restamped {done}")
    return len(patch)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--lake", action="store_true")
    args = parser.parse_args(argv)

    if args.lake:
        _walk_lake(args.apply)
        if not args.apply:
            print("\n(dry run; --apply to restamp)")
        return 0

    files = result_files(args.dir or default_trees())
    if not files:
        print("no result CSVs found", file=sys.stderr)
        return 1
    moved: Counter = Counter()
    touched = skipped = 0
    for path in files:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = list(reader.fieldnames or []), list(reader)
        if not rows or "peak_tflops_at_test" not in fields:
            continue
        if time.time() - os.path.getmtime(path) < _SETTLED_S:
            skipped += 1
            continue
        changed = 0
        for row in rows:
            fix = _restamped(row)
            if fix is None:
                continue
            moved[(row.get("gpu"), row.get("peak_tflops_at_test"), fix[0])] += 1
            row["peak_tflops_at_test"], row["peak_bw_at_test"] = fix
            changed += 1
        if changed and args.apply:
            temporary = f"{path}.rewriting"
            with open(temporary, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, path)
            touched += 1
    print(f"{len(files)} files, {sum(moved.values())} rows carry a peak the "
          f"table has corrected:")
    for (gpu, was, now), count in moved.most_common(12):
        print(f"  {count:>8}  {gpu}: {was} -> {now}")
    if skipped:
        print(f"\n{skipped} files skipped: the harness wrote them in the "
              f"last {_SETTLED_S // 60} minutes")
    if args.apply:
        print(f"rewrote {touched} files")
    else:
        print("(dry run; --apply to restamp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
