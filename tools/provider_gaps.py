#!/usr/bin/env python3
"""Which registered kernels can never be handed a cell, and which clause says no.

`tools/unverified_providers.py` asks whether a kernel has ever RUN. This asks
the prior question: could it ever be OFFERED anything? A provider's `serves`
is a promise — the coverage audit counts a cell as served when some provider
claims it, and the API publishes the claim — so a declaration that refuses
every cell the plans contain is a promise nobody can honour, and in both of
those places it is indistinguishable from a measurement.

The join is one line of logic: for every planned cell of a provider's
signature, ask `Provider.accepts`, and where it says no, keep the FIRST
clause of `Provider.refusals` (`KernelSignature.refusals` is a generator and
`accepts` reads its first item, so there is one gate here, not a predicate
beside a diagnostic). Counting those clauses is what separates the two
reasons a kernel has no rows, which want opposite actions:

  * a `max_dims` ceiling or an `activation` list the corpus always misses,
    because the corpus genuinely has no such cell — honest, and the
    registration should say so (`Provider.unreached_reason`), or
  * a declaration that disagrees with what the walk states about cells the
    corpus really does have — ours to fix, in the walk or in the
    declaration, and it must keep reading as a gap until it is.

This tool cannot tell those apart; it reports the counts and prints each
kernel's declared reason beside them, so what is left — "no reason declared"
on a kernel that accepts nothing — is exactly the set worth a human's time.

The plans are the mirrors a measuring run wrote (`jobs/*.jsonl`, one line per
planned cell). They are a moving target: a live pull rewrites them as models
are re-planned, so the COUNTS here are as of a run, while the clause is a
fact of the declaration.

Usage:
  provider_gaps.py                      # every provider that accepts nothing
  provider_gaps.py --all                # every provider, accepted / planned
  provider_gaps.py --impl marlin_fp8_w8a16        # one kernel, verbosely
  provider_gaps.py --signature gemm               # one signature
  provider_gaps.py --json out.json                # the whole join, machine-read
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: Every local record of a cell having been PLANNED, which is what a provider
#: could be handed. The box mirrors are what `bench plan` wrote for each
#: checkpoint of the corpus; a local plan directory is a hand-run plan.
PLANS = (os.path.join(ROOT, ".cache/h100_results/*/jobs/*.jsonl"),
         os.path.join(ROOT, ".cache/bench4090/jobs/*.jsonl"))


def planned_cells(patterns=PLANS, signature=None):
    """`signature -> [Invocation]` for every planned cell on this machine.

    Built through `Invocation.from_job`, the same parse the driver makes, so
    a cell is read here exactly as the run would read it — including the
    `kernel_params` a kernel may require and the formats its cache slots
    name. A job that does not parse is counted, not guessed at.
    """
    from kernel_bench.invocation import Invocation
    by_signature = collections.defaultdict(list)
    files = sorted(set(path for pattern in patterns
                       for path in glob.glob(pattern)))
    unparseable = 0
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    inv = Invocation.from_job(json.loads(line))
                except Exception:
                    unparseable += 1
                    continue
                if signature is None or inv.signature == signature:
                    by_signature[inv.signature].append(inv)
    return by_signature, files, unparseable


def join(providers, by_signature, env):
    """`(provider, planned, accepted, Counter(first refusing clause))`."""
    out = []
    for provider in providers:
        cells = by_signature.get(provider.signature, ())
        accepted, clauses = 0, collections.Counter()
        for inv in cells:
            try:
                first = next(provider.refusals(inv, env), None)
            except Exception as exc:          # a clause that cannot be read
                first = f"<raised:{type(exc).__name__}>"
            if first is None:
                accepted += 1
            else:
                clauses[first] += 1
        out.append((provider, len(cells), accepted, clauses))
    return out


def _env(providers):
    """An env in which every kernel is INSTALLED and the card is this box's.

    The question here is scope, not availability: a kernel whose library is
    absent from this checkout still has a declaration, and reading
    `detect_env` would make the answer depend on which env the tool happens
    to run in. `min_sm` is left to the real card — a Blackwell kernel on an
    H100 is unavailable, which `unverified_providers` already prints.
    """
    from kernel_bench.env import Env
    return Env(torch_version=None, cuda_available=True, device_name=None,
               gpu_sku="H100_SXM", peak_bw_gbps=None, sm="sm_90",
               libs={p.import_name: "0" for p in providers})


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="every provider, not only those that accept nothing")
    ap.add_argument("--impl", default=None, help="one kernel_impl, verbosely")
    ap.add_argument("--signature", default=None, help="one signature only")
    ap.add_argument("--json", default=None, help="write the whole join here")
    args = ap.parse_args(argv)

    from kernel_bench.providers import all_providers
    from tools.unverified_providers import why

    providers = [p for p in all_providers()
                 if (args.impl is None or p.kernel_impl == args.impl)
                 and (args.signature is None or p.signature == args.signature)]
    if not providers:
        print("no provider matches", file=sys.stderr)
        return 2
    signature = (providers[0].signature
                 if args.signature or (args.impl and len(providers) == 1)
                 else None)
    by_signature, files, unparseable = planned_cells(signature=signature)
    total = sum(len(v) for v in by_signature.values())
    print(f"{len(files)} plan files, {total} planned cells"
          + (f", {unparseable} unparseable" if unparseable else ""))
    rows = join(providers, by_signature, _env(providers))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([{"library": p.library, "kernel_impl": p.kernel_impl,
                        "signature": p.signature, "planned": n,
                        "accepted": a, "declared_reason": why(p),
                        "refusals": clauses.most_common()}
                       for p, n, a, clauses in rows], fh, indent=1)

    shown = [r for r in rows if args.all or (r[1] and not r[2])]
    print(f"\n{sum(1 for r in rows if r[1] and not r[2])} of {len(rows)} "
          f"providers accept NONE of the cells planned for their signature; "
          f"{sum(1 for r in rows if not r[1])} have no planned cell at all\n")
    for p, n, a, clauses in sorted(shown, key=lambda r: (r[0].signature,
                                                         r[0].kernel_impl)):
        print(f"{p.library:11s} {p.kernel_impl:48s} {p.signature:30s} "
              f"accepts {a:7d} of {n:7d}   {why(p)}")
        for clause, count in clauses.most_common(8 if args.impl else 3):
            print(f"{'':12s}{count:8d}  {clause}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
