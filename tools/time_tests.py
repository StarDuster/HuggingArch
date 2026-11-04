"""Time the same test files in two checkouts, alternating, and compare.

A before/after timing on a shared machine is worthless taken one run at a
time: this box runs several sessions at once, and the same unchanged file
measured twice came out 50 s and 69 s. So the comparison has to be made
against the drift rather than in spite of it — run BEFORE and AFTER back to
back, repeat, and report the best of each, which is the run least disturbed
by whatever else was running.

Usage:

    ./.venv/bin/python -m tools.time_tests \\
        --before ../ha-base \\
        --after  ../ha-change \\
        --rounds 2 \\
        tests/backend/rendering/test_compute_graph_api.py \\
        tests/backend/inference/test_module_rows.py

Each path is relative to both checkouts (make the BEFORE one a worktree of
the base commit). Extra pytest arguments go after `--`.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: pytest's own summary line ("5 failed, 101 passed, 1 skipped in 283.68s").
_OUTCOME = re.compile(r"^\d+ (passed|failed|error)", re.M)
#: the duration pytest appends to it ("... in 283.68s (0:04:43)").
_IN_SECONDS = re.compile(r" in \d[\d.]*s.*$")


def run(checkout: Path, target: str, extra: List[str]) -> Tuple[float, str]:
    """Wall seconds and pytest's outcome summary for one run."""
    started = time.perf_counter()
    done = subprocess.run(
        [str(checkout / ".venv" / "bin" / "python"), "-m", "pytest", target,
         "-q", *extra],
        cwd=checkout, capture_output=True, text=True, check=False)
    seconds = time.perf_counter() - started
    lines = [ln for ln in done.stdout.splitlines() if _OUTCOME.match(ln)]
    if not lines:
        return seconds, f"no summary (rc={done.returncode})"
    # Drop pytest's own duration, so comparing two outcomes compares the
    # counts and not how long the run took.
    return seconds, _IN_SECONDS.sub("", lines[-1]).strip()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True,
                        help="checkout to measure first (the base commit)")
    parser.add_argument("--after", type=Path, required=True,
                        help="checkout with the change")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("targets", nargs="+",
                        help="test paths, relative to both checkouts")
    parser.add_argument("extra", nargs="*", default=[],
                        help="further pytest arguments, after --")
    args = parser.parse_args(argv)

    best: Dict[Tuple[str, str], float] = {}
    outcomes: Dict[Tuple[str, str], str] = {}
    for round_number in range(1, args.rounds + 1):
        for target in args.targets:
            for side, checkout in (("before", args.before),
                                   ("after", args.after)):
                seconds, outcome = run(checkout, target, args.extra)
                key = (target, side)
                best[key] = min(seconds, best.get(key, seconds))
                outcomes[key] = outcome
                print(f"round {round_number} {side:6} {seconds:8.1f}s  "
                      f"{target}  [{outcome}]", flush=True)

    print()
    print("%-56s %9s %9s %8s" % ("file (best of %d)" % args.rounds,
                                 "before", "after", "change"))
    for target in args.targets:
        was, now = best[(target, "before")], best[(target, "after")]
        print("%-56s %9.1f %9.1f %7.0f%%"
              % (target[-56:], was, now, 100 * (now - was) / was))
    print()
    for target in args.targets:
        before, after = outcomes[(target, "before")], outcomes[(target, "after")]
        flag = "" if before == after else "   <-- OUTCOME CHANGED"
        print(f"{target}\n    before: {before}\n    after:  {after}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
