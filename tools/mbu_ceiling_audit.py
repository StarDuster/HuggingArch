#!/usr/bin/env python3
"""Why does a measured row claim more bandwidth than its card has?

`achieved_bw_gbps = access_bytes_per_rank / p50` — the numerator from the
walk, the denominator from one timed launch. When the quotient exceeds the
card's own peak, one of the two is not describing the same movement, and
this tool says which, per (op, kernel_impl) class, with numbers.

It reports three things, none of them a second traffic model:

1. THE REFUSED BAND. The write gate (`emit.above_sol`) refuses a row above
   peak; run logs recovered by `--drops` name what it refused. Rows that
   predate the gate can still be audited here.

2. DOSE-RESPONSE ON A DIM. For each class, median MBU per value of each
   shape dim. A launch that ignores a dim of the cell it serves makes the
   numerator grow with that dim while the latency does not: the median MBU
   of `sglang_causal_conv1d_update` is 0.147 / 0.199 / 0.348 / 0.404 at
   s_query 1 / 2 / 5 / 6 because the build passes `x[:, :, -1]`, one token,
   whatever the cell's window. That trend is the fingerprint of a
   cell/launch mismatch and needs no GPU to see.

3. THE WALK'S OWN DECOMPOSITION, VERIFIED BEFORE IT IS USED. For a cell
   whose activation arrives wide and is consumed narrow, the walk charges
   the conversion too — read wide, write narrow, write its scales — because
   somebody does pay those bytes (`Traffic.convert_bytes`,
   `backend/arch/eval/traffic.py`; on the serving graph a
   `quantise_activation` node of its own, or a member of the fused
   producer that writes it). The harness's FP8 providers
   quantize in their BUILD, not in the timed region, so the timed launch
   moves the GEMM's bytes and not the conversion's. This tool splits the
   two — and it is allowed to report the split only when its reconstruction
   of the numerator comes back to `access_bytes_per_rank` within
   `RESIDUAL_TOLERANCE` (0.1%), with the residual printed. A wider gap
   prints MISMATCH and no ratio: the tool cannot both be wrong about the
   engine's model and be trusted about the remainder. The tolerance is not
   slack for a cost model — it is two roundings this tool does not restate,
   a merged column's weight scales being tiled PER PORT (224 bytes on a
   `d_reduce=7168` cell) and a routed cell's rows being padded per expert
   (`launch_rows` vs `launch_rows_exact`, ≤3e-4 of the total). A 2.6x
   numerator error cannot hide in 0.1%. `gemm`, `per_head_gemm` and
   `grouped_gemm` are the three shapes it knows.

Not a gate. A row's p50 is a real measurement of a real launch; what is
wrong in these classes is which movement the numerator describes, and that
is fixed where the cell is planned (`kernel_bench/plan.py`) or where the
provider declares the geometry it runs (`serves=KernelSignature`), never by
recomputing traffic in the harness — that second model existed, drifted
from the walk on 6 of 11 signatures, and was removed (CLAUDE.md §1.2).

Usage:
  mbu_ceiling_audit.py                       # classes over peak, ranked
  mbu_ceiling_audit.py --class per_head_linear[:kernel_impl]
  mbu_ceiling_audit.py --dims                # dose-response per class
  mbu_ceiling_audit.py --drops               # what the write gate refused
  mbu_ceiling_audit.py --reconcile           # join jobs, split the numerator
Options: --results DIR (repeatable), --jobs DIR (repeatable), --all (every
row, not only over-peak), --limit N.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import itertools
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.unserved_rows import is_record

DEFAULT_RESULTS = (".cache/h100_results", ".cache/bench4090")
DEFAULT_JOBS = (".cache/h100_results/*/jobs", ".cache/bench4090/jobs")
#: `emit.UTILIZATION_CEILING` — kept as a number to compare against, not as a
#: second definition.
CEILING = 1.0

# Bytes per element, for the dtype tokens a `dtype_env` uses. Narrow-format
# metadata is fp32 in every scheme the corpus carries.
WIDTH = {"bf16": 2, "fp16": 2, "float16": 2, "bfloat16": 2, "fp32": 4,
         "float32": 4, "fp8_e4m3": 1, "fp8_e5m2": 1, "fp8": 1, "int8": 1,
         "fp4": 0.5, "nvfp4": 0.5, "mxfp4": 0.5, "int4": 0.5}
_SCALE_BLOCK = 128
#: How far the reconstruction may miss the walk's own number and still be
#: reported — see the module docstring: two roundings, not slack.
RESIDUAL_TOLERANCE = 1e-3


def _rows(results_dirs):
    """Every measured row under the result dirs, with its MBU."""
    out = []
    for d in results_dirs:
        for path in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
            if is_record(path):
                continue    # a record's rows are deleted, not corpus
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        got = float(row.get("achieved_bw_gbps") or 0)
                        peak = float(row.get("peak_bw_at_test") or 0)
                    except (TypeError, ValueError):
                        continue
                    if not peak or not got:
                        continue
                    row["mbu"] = got / peak
                    row["_file"] = path
                    out.append(row)
    return out


def _classes(rows, *, over_only=True):
    by = collections.defaultdict(list)
    for row in rows:
        if over_only and row["mbu"] <= 1.0:
            continue
        by[(row["op"], row["kernel_impl"], row["gpu"])].append(row)
    return by


def report_classes(rows, over_only=True):
    by = _classes(rows, over_only=over_only)
    print(f"{'rows':>5}  {'min':>6} {'med':>6} {'max':>6}   op / kernel_impl / gpu")
    for key, got in sorted(by.items(), key=lambda kv: -len(kv[1])):
        mbu = [r["mbu"] for r in got]
        print(f"{len(got):5d}  {min(mbu):6.3f} {statistics.median(mbu):6.3f} "
              f"{max(mbu):6.3f}   {key[0]} / {key[1]} / {key[2]}")
    print(f"\nthe write gate refuses above {CEILING:.2f}x peak, so every row "
          f"here is the bottom of its class's real overstatement; "
          f"--drops says what was refused.")


def report_dims(rows, want=None):
    """Median MBU per value of each dim: a dim the launch ignores trends."""
    by = _classes(rows, over_only=False)
    ranked = sorted(by.items(), key=lambda kv: -sum(r["mbu"] > 1 for r in kv[1]))
    for key, got in ranked:
        if want and want not in (key[0], f"{key[0]}:{key[1]}"):
            continue
        over = sum(r["mbu"] > 1 for r in got)
        if not over and want is None:
            continue
        print(f"\n== {key[0]} / {key[1]} / {key[2]}  "
              f"({len(got)} rows, {over} over peak)")
        dims = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in got:
            try:
                shape = json.loads(row["shape_json"])
            except (TypeError, ValueError):
                continue
            for name, value in shape.items():
                dims[name][value].append(row["mbu"])
        for name, values in sorted(dims.items()):
            if len(values) < 2:
                continue
            cells = " ".join(
                f"{v}:{statistics.median(m):.3f}({len(m)})"
                for v, m in sorted(values.items())[:12])
            trend = _monotone(values)
            print(f"   {name:20s} {cells}{'   <-- MONOTONE' if trend else ''}")


def _monotone(values):
    """Does median MBU rise with the dim over at least 3 distinct values?"""
    points = [(v, statistics.median(m)) for v, m in sorted(values.items())
              if isinstance(v, (int, float))]
    if len(points) < 3:
        return False
    ups = sum(1 for a, b in itertools.pairwise(points)
              if b[1] > a[1] * 1.15)
    return ups >= len(points) - 1


_DROP = re.compile(r"DROP (\S+) (\S+) (\{.*?\}): above SOL — "
                   r"claimed (compute|bandwidth) is ([\d.]+)%")


def report_drops(log_globs):
    """What the write gate refused, per class, from the run logs."""
    counts = collections.Counter()
    pcts = collections.defaultdict(list)
    files = [p for g in log_globs for p in glob.glob(g, recursive=True)]
    for path in files:
        with open(path, errors="replace") as fh:
            for line in fh:
                hit = _DROP.search(line)
                if not hit:
                    continue
                key = (hit.group(2), hit.group(1), hit.group(4))
                counts[key] += 1
                pcts[key].append(float(hit.group(5)))
    print(f"{len(files)} log files, {sum(counts.values())} refused rows "
          f"(historical: a class fixed since still appears here)")
    print(f"{'rows':>6}  {'min':>7} {'med':>7} {'max':>9}   op / kernel / side")
    for key, got in counts.most_common():
        val = sorted(pcts[key])
        print(f"{got:6d}  {val[0]:6.0f}% {val[len(val) // 2]:6.0f}% "
              f"{val[-1]:8.0f}%   {key[0]} / {key[1]} / {key[2]}")


def _job_index(jobs_globs, models):
    """(model, op, shape_json, dtype_json, phase, parallel) -> job."""
    want = {m.replace("/", "__") + ".jsonl" for m in models}
    index = {}
    for pattern in jobs_globs:
        for directory in glob.glob(pattern):
            for name in os.listdir(directory):
                if name not in want:
                    continue
                with open(os.path.join(directory, name)) as fh:
                    for line in fh:
                        job = json.loads(line)
                        key = (job.get("model", ""), job["op"],
                               json.dumps(job["dims"], separators=(",", ":"),
                                          sort_keys=True),
                               json.dumps(job["dtype_env"],
                                          separators=(",", ":"), sort_keys=True),
                               job["phase"], job.get("parallel") or "tp=1")
                        index.setdefault(key, job)
    return index


def _elements(shape):
    if shape is None:
        return None
    if shape and isinstance(shape[0], list):
        return sum(_elements(s) for s in shape)
    return math.prod(shape)


def _rows_and_k(signature, shape_in):
    """(rows, K) of the activation the GEMM contracts, or None.

    One place per signature, read off the shapes the walk published — the
    same positional read `refs` makes (`bb301d5a0`), not a reconstruction
    from `dims`.
    """
    if shape_in and isinstance(shape_in[0], list):
        return None
    if signature in ("gemm", "per_head_gemm", "grouped_gemm"):
        return math.prod(shape_in[:-1]), shape_in[-1]
    return None


def split_numerator(job):
    """(gemm_bytes, convert_bytes, reconstructed_total) or None.

    `convert` is the wide read + narrow write + scale write the walk charges
    for an operand that arrives wide and is consumed narrow
    (`Traffic.convert_bytes`); `gemm` is the rest. Returns None when the
    cell carries no conversion, and the caller must check that
    `reconstructed_total == job["access_bytes_per_rank"]` before believing
    either half.
    """
    env = job.get("dtype_env") or {}
    act, out = env.get("activation"), env.get("output")
    weight = env.get("weight")
    if act not in ("fp8_e4m3", "fp8_e5m2", "fp8") or out == act:
        return None
    got = _rows_and_k(job.get("signature"), job.get("shape_in"))
    if got is None:
        return None
    rows, k_dim = got
    wide, narrow = WIDTH.get(out, 2), WIDTH.get(act, 1)
    blocks = -(-k_dim // _SCALE_BLOCK)
    act_scale = rows * blocks * WIDTH["fp32"]
    convert = rows * k_dim * wide + rows * k_dim * narrow + act_scale
    # The weight shape when the job carries one (`shape_weight`), and the
    # identity's scalars when it does not — a merged column's weight is two
    # ports of one tensor and a routed bank's is one expert's, so the walk
    # has no single shape to carry there. Reading `dims` is the lossy road
    # (CLAUDE.md §2.7) and is why the caller must check the reconstruction
    # against `access_bytes_per_rank` before believing any of this.
    w_shape = job.get("shape_weight")
    if w_shape is None:
        dims = job.get("dims") or {}
        if not {"d_out", "d_reduce"} <= dims.keys():
            return None
        w_shape = [dims["d_out"], dims["d_reduce"]]
        if "n_experts" in dims:
            w_shape = [dims["n_experts"], *w_shape]
    w_elements = _elements(w_shape)
    o_elements = _elements(job.get("shape_out"))
    if w_elements is None or o_elements is None:
        return None
    w_scale = 0
    if weight in ("fp8_e4m3", "fp8_e5m2", "fp8") and len(w_shape) >= 2:
        tiles = math.prod(-(-d // _SCALE_BLOCK) for d in w_shape[-2:])
        w_scale = tiles * math.prod(w_shape[:-2] or [1]) * WIDTH["fp32"]
    gemm = (rows * k_dim * narrow + act_scale
            + w_elements * WIDTH.get(weight, 2) + w_scale
            + o_elements * WIDTH.get(out, 2))
    return gemm, convert, gemm + convert


def report_reconcile(rows, jobs_globs, limit=None, want=None):
    over = [r for r in rows if r["mbu"] > 1.0]
    if want:
        over = [r for r in over
                if want in (r["op"], f"{r['op']}:{r['kernel_impl']}")]
    models = {r["model"] for r in over}
    print(f"joining {len(over)} over-peak rows in {len(models)} models "
          f"to their planned jobs...", file=sys.stderr)
    index = _job_index(jobs_globs, models)
    seen = collections.defaultdict(list)
    unjoined = 0
    for row in over:
        key = (row["model"], row["op"], row["shape_json"],
               row["dtype_env_json"], row["phase"], row["parallel"])
        job = index.get(key)
        if job is None:
            unjoined += 1
            continue
        seen[(row["op"], row["kernel_impl"], row["gpu"])].append((row, job))
    print(f"{unjoined} rows have no job on this box (its plan was not pulled)\n")
    for key, pairs in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        print(f"== {key[0]} / {key[1]} / {key[2]}  ({len(pairs)} joined)")
        for row, job in pairs[:limit or 3]:
            total = float(job["access_bytes_per_rank"])
            split = split_numerator(job)
            print(f"   {row['model']} {row['phase']} {row['parallel']}")
            print(f"     dims      {row['shape_json']}")
            print(f"     shape_in  {job['shape_in']}  -> out {job['shape_out']}"
                  f"  w {job['shape_weight']}")
            print(f"     engine    {total / 1e6:10.2f} MB   p50 {row['p50_us']} us"
                  f"   MBU {row['mbu']:.3f}")
            if split is None:
                print("     (no conversion term in this cell: the numerator is "
                      "the launch's own operands, or an unknown signature)")
                continue
            gemm, convert, rebuilt = split
            residual = (rebuilt - total) / total
            if abs(residual) > RESIDUAL_TOLERANCE:
                print(f"     MISMATCH  reconstructed {rebuilt / 1e6:.2f} MB vs "
                      f"{total / 1e6:.2f} MB ({residual:+.2%}) — no split")
                continue
            ratio = total / gemm
            print(f"     convert   {convert / 1e6:10.2f} MB  (wide read + narrow "
                  f"write + scales — a launch the build does untimed)")
            print(f"     gemm      {gemm / 1e6:10.2f} MB  (what the timed kernel "
                  f"reads and writes; reconstruction residual {residual:+.4%})")
            print(f"     ratio     {ratio:10.4f}x   MBU if the cell were the "
                  f"gemm launch alone: {row['mbu'] / ratio:.3f}")
        print()
    _class_summary(seen)


def _class_summary(seen):
    print("corrected band per class (over-peak rows only, exact "
          "reconstructions only):")
    for key, pairs in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        fixed = []
        for row, job in pairs:
            split = split_numerator(job)
            if split is None:
                continue
            gemm, _convert, rebuilt = split
            total = float(job["access_bytes_per_rank"])
            if abs(rebuilt - total) / total > RESIDUAL_TOLERANCE:
                continue
            fixed.append(row["mbu"] * gemm / rebuilt)
        if not fixed:
            continue
        print(f"   {key[0]:32s} {key[1]:34s} n={len(fixed):3d}  "
              f"corrected MBU {min(fixed):.3f}..{max(fixed):.3f}"
              f"{'   ALL UNDER PEAK' if max(fixed) <= 1.0 else '   STILL OVER'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", action="append", default=[])
    ap.add_argument("--jobs", action="append", default=[])
    ap.add_argument("--logs", action="append", default=[])
    ap.add_argument("--class", dest="klass", default=None,
                    help="op or op:kernel_impl to restrict to")
    ap.add_argument("--dims", action="store_true")
    ap.add_argument("--drops", action="store_true")
    ap.add_argument("--reconcile", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="every row, not only the ones over peak")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    results = args.results or list(DEFAULT_RESULTS)
    rows = _rows(results)
    print(f"{len(rows)} measured rows with both an achieved and a peak "
          f"bandwidth, {sum(r['mbu'] > 1 for r in rows)} of them over peak\n")
    if args.drops:
        report_drops(args.logs or [".cache/**/queue.out",
                                   ".cache/**/run-all.log", "results/**/*.log"])
        return 0
    if args.dims:
        report_dims(rows, args.klass)
        return 0
    if args.reconcile:
        report_reconcile(rows, args.jobs or list(DEFAULT_JOBS),
                         limit=args.limit, want=args.klass)
        return 0
    report_classes(rows, over_only=not args.all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
