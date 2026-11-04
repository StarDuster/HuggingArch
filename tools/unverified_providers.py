#!/usr/bin/env python3
"""Which registered kernels have never actually run?

A provider's `serves` is a PROMISE to consumers: the coverage audit counts a
cell as served when some provider claims it, and the API publishes the claim
(`capabilities`). A claim made from reading source and never executed is
indistinguishable, in every one of those places, from one measured a
thousand times — and it is not the same thing.

The case this exists for: `deepgemm_fp8_blockwise` was widened to the 32-wide
scale tile on the strength of its kernel template, which does compile that
granularity; no HOST path on sm90 reaches it, so every one of the 1,410 cells
it claimed would have failed at build. Reading the template is not running
the kernel, and nothing said so until someone ran it.

Joins the registry against the rows this machine has: the box mirrors plus
the local 4090 results. A kernel with no rows is not necessarily broken —
it may be waiting for a card (`min_sm`), or for an env, or simply not have
come up in the queue yet, or be right about a launch no checkpoint we serve
asks for (`unreached_reason`: marlin's weight-only FP8, the fp16 twin of a
norm every corpus model ships in bf16) — so this prints WHY it might be,
and leaves the judgement to a human. What that leaves is the set worth
reading: "no reason declared" beside a kernel that accepts none of the
planned cells (`tools/provider_gaps.py`) is a declaration bug or a gap in
our own plans, not an honest absence.

Usage: unverified_providers.py [--all]
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: Every local record of a row having been measured. The box mirrors are
#: what a measuring run wrote; `bench_lake/corpora` is what the service
#: exports of the lake, which is where a row measured outside the queue
#: (a hand probe, another contributor's upload) is visible.
ROOTS = (os.path.join(ROOT, ".cache/h100_results/*/results/*/*.csv"),
         os.path.join(ROOT, ".cache/bench4090/results/*/*.csv"),
         os.path.join(ROOT, ".cache/bench_lake/corpora/*.csv"))


def measured_impls() -> collections.Counter:
    """Registered kernel → rows measured, over every recorded spelling.

    A row records `Provider.impl_id`, and the separator between the name and
    its resolved parameters changed once (`flashmla_decode_b64` before,
    `flashmla_decode@_b64` after), so the recorded name is classified by the
    registry rather than by splitting on a character — `catalog_kernel` is
    the one classifier, and asking it here is what keeps a kernel measured
    under the older spelling from reading as never run."""
    from kernel_bench.providers import catalog_kernel
    seen: collections.Counter = collections.Counter()
    for pattern in ROOTS:
        for path in glob.glob(pattern):
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    for row in csv.DictReader(fh):
                        impl = catalog_kernel(row.get("kernel_impl") or "")
                        if impl:
                            seen[impl] += 1
            except OSError:
                continue
    return seen


def why(provider) -> str:
    """Why this kernel might have no rows, from its own declaration.

    Three different facts, each declared beside the kernel and each wanting
    a different action: it must not be measured (`disabled_reason`), the
    card is wrong for it (`min_sm`), or it is right and nothing in this
    corpus asks for it (`unreached_reason` — marlin's weight-only FP8
    against a corpus whose FP8 models quantise the activation too). What is
    left — "no reason declared" — is the set worth a human's time, and it
    is the only reading that should ever be a to-do.

    Read off the declaration, never inferred from the name or the rows: a
    kernel's own registration is where its author said this, and a second
    list here would be a second thing to keep in step.
    """
    reasons = []
    if provider.disabled_reason:
        reasons.append("disabled")
    if provider.min_sm:
        reasons.append(f"needs sm_{provider.min_sm}")
    if provider.unreached_reason:
        reasons.append(f"no cell in this corpus: {provider.unreached_reason}")
    return ", ".join(reasons) or "no reason declared"


def main(argv) -> int:
    from kernel_bench.providers import all_providers

    rows = measured_impls()
    unverified, verified = [], 0
    for p in all_providers():
        if rows.get(p.kernel_impl):
            verified += 1
            continue
        unverified.append((p.library, p.kernel_impl, p.signature, why(p)))

    print(f"{verified} of {verified + len(unverified)} registered kernels have "
          f"rows on this machine; {len(unverified)} have none\n")
    print(f"{'library':12s} {'kernel_impl':44s} {'signature':22s} why it might be")
    for library, impl, sig, reason in sorted(unverified):
        print(f"{library:12s} {impl:44s} {sig:22s} {reason}")
    if "--all" in argv:
        print("\nmeasured, by rows:")
        for impl, n in rows.most_common():
            print(f"  {n:8d} {impl}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
