"""Inspect measured rows: MFU, MBU, and whether the cross-check actually ran.

The write path already refuses a row above SOL and never records a kernel
that disagreed with the reference. This is the same physics, applied to a
file after the fact — a resume, an older CSV, a directory `run-all` left
behind — so a sweep can be judged without touching the GPU.

MFU / MBU are `emit.above_sol` (claimed work vs the card's own peaks).
Allclose is the recorded `cross_check` label: a disagreement is never
written, so the question left is which rows were compared (`passed`) and
which were measured without an oracle (`no_reference` / `disabled` / …).
"""
from __future__ import annotations

import glob
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from backend.inference import benchmarks as _benchmarks
from backend.inference.benchmarks import BenchmarkValidationError

from .emit import above_sol, utilization


@dataclass
class CheckReport:
    n_rows: int = 0
    n_files: int = 0
    sol: List[str] = field(default_factory=list)
    allclose: Counter = field(default_factory=Counter)
    # Rows that record an outcome instead of a measurement (`outcome="oom"`:
    # the card could not hold the cell). Counted apart from `allclose`,
    # where they would read as measurements nobody verified.
    refused: int = 0
    mfu: List[float] = field(default_factory=list)
    mbu: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.sol and not self.errors


def check_rows(rows: Iterable[Dict[str, str]]) -> CheckReport:
    """Apply the three checks to already-parsed rows."""
    report = CheckReport()
    for row in rows:
        report.n_rows += 1
        if not _benchmarks.is_measurement(row):
            # A recorded refusal was never checked because nothing ran; in
            # the "unchecked" bucket it reads as a measurement nobody
            # verified, which is a different and worse claim.
            report.refused += 1
            continue
        report.allclose[row.get("cross_check") or ""] += 1
        got = utilization(row)
        if got is not None:
            mfu, mbu = got
            if float(row.get("peak_tflops_at_test") or 0):
                report.mfu.append(mfu)
            if float(row.get("peak_bw_at_test") or 0):
                report.mbu.append(mbu)
        why = above_sol(row)
        if why is not None:
            report.sol.append(
                f"{row.get('op')} {row.get('kernel_impl')}: {why}")
    return report


def check_paths(results: str) -> CheckReport:
    """Load CSV(s) and check them. `results` is a file or a directory."""
    paths = _csv_paths(results)
    report = CheckReport(n_files=len(paths))
    if not paths:
        report.errors.append(f"no CSV at {results}")
        return report
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                _header, rows = _benchmarks.validate_text(
                    handle.read(), source=path)
        except (OSError, BenchmarkValidationError) as exc:
            report.errors.append(str(exc))
            continue
        part = check_rows(rows)
        report.n_rows += part.n_rows
        report.sol.extend(part.sol)
        report.allclose.update(part.allclose)
        report.mfu.extend(part.mfu)
        report.mbu.extend(part.mbu)
    return report


def format_report(report: CheckReport) -> str:
    lines = [
        f"check: {report.n_rows} rows in {report.n_files} file(s)",
        f"  mfu: {_pct_line(report.mfu)}",
        f"  mbu: {_pct_line(report.mbu)}",
        f"  allclose: {_allclose_line(report.allclose)}",
    ]
    if report.sol:
        lines.append(f"  FAIL sol: {len(report.sol)} row(s) claim work "
                     f"the card cannot do")
        lines.extend(f"    {item}" for item in report.sol[:10])
        extra = len(report.sol) - 10
        if extra > 0:
            lines.append(f"    … {extra} more")
    if report.errors:
        lines.append(f"  FAIL: {len(report.errors)} file(s) unreadable")
        lines.extend(f"    {item}" for item in report.errors[:10])
    if report.ok():
        lines.append("  ok")
    return "\n".join(lines)


def _csv_paths(results: str) -> Sequence[str]:
    if os.path.isdir(results):
        return sorted(glob.glob(os.path.join(results, "*.csv")))
    if os.path.isfile(results):
        return [results]
    return []


def _pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ordered = sorted(xs)
    index = min(len(ordered) - 1,
                max(0, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _pct_line(xs: List[float]) -> str:
    if not xs:
        return "n=0 (no recorded peak)"
    p50, p99 = _pct(xs, 50), _pct(xs, 99)
    return (f"p50={p50 * 100:.0f}% p99={p99 * 100:.0f}%  "
            f"(n={len(xs)} with a peak)")


def _allclose_line(counts: Counter) -> str:
    if not counts:
        return "(none)"
    order = ("passed", "no_reference", "not_comparable", "disabled", "unknown")
    parts = [f"{name}={counts[name]}" for name in order if counts[name]]
    extra = sorted(k for k in counts if k and k not in order)
    parts.extend(f"{name}={counts[name]}" for name in extra)
    empty = counts.get("")
    if empty:
        parts.append(f"(blank)={empty}")
    return "  ".join(parts) or "(none)"
