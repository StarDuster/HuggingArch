#!/usr/bin/env python3
"""Does the harness build the operands the WALK says this cell moves?

The gap this closes, in one example. `moe_permute`'s signature declared its
input with the OUTPUT's row count, so the harness built an equal-length
permutation while the row's numerator charged a dispatch — 1.50x the traffic
on a DeepSeek-V3.2 prefill cell, 0.63x on a gpt-oss one. Nothing caught it
for weeks: `cell_check` compares the built tensors against the DECLARATION
(so a wrong declaration is invisible to it), the cross-check's oracle for a
movement op is the provider's own expression (self-confirming), and the SOL
ceiling only fires on the high side — under `ep < topk` the error is low and
silent.

The missing comparison is between two facts that both already exist: the
elements the signature's declared operands hold, and the elements the walk
publishes on the job (`shape_in`). This is NOT a second traffic model (which
`plan.py` tried and removed): no bytes, no dtypes, no padding law — just
whether the two sides count the same elements.

A report, not a gate: some providers legitimately build operands the engine
does not carry (a paged cache's block table, a fused span's scratch). Each
such signature has to be looked at and declared before this can refuse
anything — CLAUDE.md §4.

Baseline over the H100 plans (519,051 cells with carried shapes), so the
next person knows what is already understood and what is not. EXACT on
every cell, and therefore ready to be declared and enforced:

    gemm 159,647   silu_and_mul 30,336   embedding 25,676   rmsnorm 20,829
    all_reduce 13,920   per_head_gemm 11,119   scale_const 6,423
    all_to_all 4,992   silu_and_mul_with_clamp 4,876   sigmoid 4,820
    moe_gate 4,711   fused_add_rmsnorm 4,308   topk 3,641
    causal_conv1d 3,610   rmsnorm_unscaled 3,338   moe_topk_softmax 2,580
    dsa_topk_mask 2,058   sum_reduce 1,862   softplus 264   neg_exp 16

Structurally not comparable, with the reason:

  * `all_gather` 2-512x, `allreduce_fused_rmsnorm` exactly 2x — a
    collective's declaration is the whole message, the walk carries one
    rank's buffer;
  * `mla_sparse_attention` 0.004-0.999x, `indexer_logits` 0.004-1.007x —
    the harness allocates what the kernel addresses, the walk charges what
    it streams, and a sparse read is most of that difference;
  * `fused_recurrent_gated_delta_rule` 8-77x — a recurrent state against a
    token stream.

Mixed, and each one a lead rather than a verdict:

  * `moe_permute` 0.016-892x — THE BUG this tool exists for;
  * `gqa_attention` 67% exact (0.8-1.999x), `mla_attention` 25% (0.53-1x),
    `moe_combine` 33% (0.78-1x), `mul` 53% (1-2x), `add` 91% (1-2x),
    `grouped_gemm` 96% (1-1920x), `rmsnorm_gated` 90% (0.004-1.97x);
  * `rope` / `rope_partial` — this tool cannot read their declaration yet
    (`_shape` raises); a gap in the tool, not a finding.

Two of those numbers were the tool's OWN bugs, found by looking at the
outliers rather than trusting them: keying the stored-weight exclusion on
the operand's role reported 1,904 gemm cells at 0.04x (an external weight
IS an operand of `shape_in`), and keying it on `shape_weight` reported some
at 55,297x (a fused span's stored weight has no single shape, so that field
is None there too). Hence the operand-count rule in `_declared_elements`,
after which gemm is exact on all 159,647 of its cells.

Turning it into a gate means declaring the expected relation per signature
(beside the signature, not in a list here) and then refusing what
disagrees. Until then it is read, not enforced.

Exit code 1 when a signature whose projection is `"contract"` cannot name
what the walk carried (`_cannot_zip_carried`) — a deficient DECLARATION,
which is the structural cause of the whole class. The element-count
comparison beside it stays a report: a ratio is not a verdict until the
signature's expected relation is stated, and stating it per signature by
hand is what two reviews rejected (four collectives with identical
declarations would have needed three different hand-written verdicts).

Usage: cell_parity_report.py PLAN_DIR_OR_JSONL [--all]
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Agreement within this factor is not worth a line: a declared operand may
#: round a routed rows axis up to a rectangle where the walk's own extent is
#: the exact mean (`launch_rows` vs `launch_rows_exact`).
TOLERANCE = 1.10


def _walk_elements(shape_in) -> int:
    """Elements the walk says this cell reads, over all of its operands."""
    operands = shape_in
    if operands and not isinstance(operands[0], (list, tuple)):
        operands = [operands]
    total = 0
    for operand in operands or ():
        total += math.prod(int(d) for d in operand)
    return total


def _cannot_zip_carried(inv) -> bool:
    """Whether this cell's operands came from the cell's LOSSY projection
    because the signature's engine layouts could not name what the walk
    carried.

    Derived, not declared, and that is the point. `moe_permute` declares
    `engine_outputs` and no `engine_inputs`, so `refs._carried_axes` zips
    nothing, `refs._dims` falls back to `inv.dims` — "the lossy corpus-key
    projection", by its own docstring — and the harness then built an
    equal-length permutation against a dispatch's numerator. Asking the
    declarations that already exist catches that, defaults to a FINDING for
    a new signature rather than to silence, and needs no second vocabulary
    of per-signature verdicts (four collectives with identical declarations
    would have taken three different hand-written ones).

    A projection other than `"contract"` is excluded because it says so
    itself: the collective's published payload, the DSA indexer's selection
    mask, the packed-qkv rope's folded width are declared NOT to be layout
    permutations of the carried shapes.
    """
    from backend.inference.op_signatures import get_signature
    from kernel_bench.refs import _carried_axes

    sig = get_signature(inv.signature)
    if sig is None or sig.shape_projection != "contract":
        return False
    if inv.shape_in is None:
        return False
    return not _carried_axes(inv)


def _declared_elements(inv) -> int:
    """Elements the signature's declared ACTIVATION operands hold — what the
    harness allocates (`refs._shape` reads the same declaration)."""
    from backend.inference.op_signatures import get_signature
    from kernel_bench.refs import _dims, _shape

    sig = get_signature(inv.signature)
    if sig is None:
        return 0
    layouts = list(sig.layouts_for(_dims(inv)))
    # Which declared operands belong to the side the walk carries. A STORED
    # weight does not: it is a parameter, and `shape_in` is activations only.
    # An EXTERNAL one does — `linear_external_weight`'s weight is computed by
    # an earlier node, so the walk lists it as an operand and the harness
    # builds it like any input. Neither the role nor `shape_weight` tells the
    # two apart (a fused span's stored weight has no single shape either, so
    # `shape_weight` is None there too); the walk's own OPERAND COUNT does.
    # Keying on the role alone reported 1,904 gemm cells at 0.04x and keying
    # on `shape_weight` reported some at 55,297x — both were this tool's
    # findings about itself.
    operands = inv.shape_in
    carried = (len(operands) if operands and isinstance(operands[0], (list, tuple))
               else 1)
    if len(layouts) != carried:
        layouts = [lay for lay in layouts if lay.role != "weight"]
    return sum(math.prod(int(d) for d in _shape(inv, lay.name))
               for lay in layouts)


def main(argv) -> int:
    from kernel_bench.invocation import Invocation

    target = next((a for a in argv if not a.startswith("-")), "")
    show_all = "--all" in argv
    paths = ([target] if target.endswith(".jsonl")
             else sorted(glob.glob(os.path.join(target, "*.jsonl"))))
    if not paths:
        print(f"no plans at {target!r}")
        return 2

    seen = set()
    off = collections.Counter()
    # The STRUCTURAL finding, derived rather than declared — see
    # `_cannot_zip_carried`.
    unzipped: collections.Counter = collections.Counter()
    worst: dict = {}
    cells = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        for line in lines:
            job = json.loads(line)
            key = json.dumps(job.get("physical_key"))
            if key in seen:
                continue
            seen.add(key)
            if job.get("shape_in") is None:
                continue
            try:
                inv = Invocation.from_job(job)
                declared = _declared_elements(inv)
            except Exception as exc:          # a cell this tool cannot read
                off[(job["signature"], f"unreadable: {type(exc).__name__}")] += 1
                continue
            walk = _walk_elements(job["shape_in"])
            cells += 1
            if not walk or not declared:
                continue
            ratio = declared / walk
            agrees = 1 / TOLERANCE <= ratio <= TOLERANCE
            if _cannot_zip_carried(inv):
                unzipped[job["signature"]] += 1
            if show_all or not agrees:
                off[(job["signature"], "")] += 1
                prev = worst.get(job["signature"])
                if prev is None or abs(math.log(ratio)) > abs(math.log(prev[0])):
                    worst[job["signature"]] = (
                        ratio, job["op"], job["phase"], job["dims"],
                        job["shape_in"], declared, walk)

    print(f"{cells} cells with carried shapes, "
          f"{sum(off.values())} outside {TOLERANCE:g}x\n")
    if unzipped:
        print("DEFICIENT DECLARATIONS — the walk carries operands these "
              "signatures' engine layouts cannot name, so the harness built "
              "its own from the cell's lossy projection:")
        for sig, n in unzipped.most_common():
            print(f"  {sig:26s} {n} cell(s)")
        print()
    print(f"{'signature':26s} {'cells':>7s}  worst ratio  declared / walk elements")
    for (sig, note), n in off.most_common():
        if note:
            print(f"{sig:26s} {n:7d}  {note}")
            continue
        ratio, op, phase, dims, shape_in, declared, walk = worst[sig]
        print(f"{sig:26s} {n:7d}  {ratio:9.3f}x  {declared:,} / {walk:,}")
        print(f"{'':26s} {'':7s}  {op} {phase} {json.dumps(dims)[:60]}")
        print(f"{'':26s} {'':7s}  walk shape_in {json.dumps(shape_in)[:70]}")
    return 1 if unzipped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
