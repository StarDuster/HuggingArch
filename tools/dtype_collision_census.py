"""Read-only census of legacy restamp keys that hid a dtype collision.

The pre-fix ``hbench restamp`` index omitted ``dtype_env_json``.  Two complete
physical cells at one model/node/shape/phase/parallel but different dtypes can
therefore have shared that legacy key.  If their numerators differ, plan order
chose a winner.  This tool answers the operational question without touching
the lake or any result: which current corpus plans still contain such keys?

    python -m tools.dtype_collision_census --plans /path/to/jobs \
        [--out census.json]

``legacy_keys`` counts collision-prone groups under the old key (the review's
“old-key dtype collisions”).  ``dtype_cells`` counts the complete dtype-aware
physical cells in those groups — normally two per legacy key.  Both numbers
are reported because the first describes the old overwrite surface and the
second the cells that a corrected join must price independently.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from backend.inference.benchmark_identity import (
    InvalidIdentityJSON, canonical_identity_json,
)

LegacyKey = Tuple[str, str, str, str, str, str]
PriceIdentity = Tuple[str, Tuple[Optional[float], Optional[float]]]


@dataclass(frozen=True)
class Collision:
    """One legacy key whose dtype-distinct cells carry distinct prices."""

    model: str
    op: str
    legacy_key_count: int = 1
    dtype_cell_count: int = 0

    def as_report(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "op": self.op,
            "legacy_keys": self.legacy_key_count,
            "dtype_cells": self.dtype_cell_count,
        }


def _finite_pair(job: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    out: List[Optional[float]] = []
    for field in ("flops_per_rank", "access_bytes_per_rank"):
        value = job.get(field)
        if value is None:
            out.append(None)
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite {field} in plan job")
        out.append(number)
    return out[0], out[1]


def _carried_cell(job: Dict[str, Any]) -> Tuple[str, str, str, str]:
    carried = job.get("physical_key")
    if not isinstance(carried, (list, tuple)) or len(carried) != 4:
        raise ValueError(
            f"plan job {job.get('op') or '<unknown>'} carries no physical_key")
    op, shape_json, dtype_env_json, phase = carried
    try:
        canonical_shape = canonical_identity_json(shape_json)
        canonical_dtype = canonical_identity_json(dtype_env_json)
    except InvalidIdentityJSON as exc:
        raise ValueError(
            f"plan job {op!r} has an invalid physical_key: {exc}") from exc
    return op, canonical_shape, canonical_dtype, phase


def census_dtype_collisions(jobs_by_plan: Iterable[Tuple[str, List[Dict[str, Any]]]]
                            ) -> Dict[str, Any]:
    """Return collision aggregates from ``(plan name, jobs)`` snapshots.

    Jobs are grouped by the exact legacy restamp key: model, op, node, shape,
    phase, and parallel, with dtype omitted.  A group is reported only when it
    has both more than one dtype and more than one complete ``(dtype, price)``
    identity; two dtypes with the same numerator pair cannot misprice either
    row even though the old key still overwrote the same value.
    """
    # legacy key -> dtype/price identities
    groups: Dict[LegacyKey, set[PriceIdentity]] = defaultdict(set)
    for plan, jobs in jobs_by_plan:
        for job in jobs:
            if "op" not in job or not job.get("physical_key"):
                # Plans also carry structural sidecar records.  They have no
                # numerator and were never restamp index entries.
                continue
            _op, shape_json, dtype_json, phase = _carried_cell(job)
            model = str(job.get("model") or plan)
            legacy = (
                model, str(job["op"]), str(job.get("node_spec_id") or ""),
                shape_json, phase, str(job.get("parallel") or "tp=1"))
            groups[legacy].add((dtype_json, _finite_pair(job)))

    by_model_op: Dict[Tuple[str, str], List[LegacyKey]] = defaultdict(list)
    for legacy, identities in groups.items():
        dtypes = {dtype for dtype, _pair in identities}
        prices = {pair for _dtype, pair in identities}
        if len(dtypes) > 1 and len(prices) > 1:
            by_model_op[(legacy[0], legacy[1])].append(legacy)

    rows = []
    for (model, op), legacies in sorted(by_model_op.items()):
        # A legacy key's dtype set is its complete cells; prices need not be
        # unique within a cell before duplicate plan jobs are folded.
        cells = sum(
            len({dtype for dtype, _pair in groups[legacy]})
            for legacy in legacies)
        rows.append(Collision(model, op, len(legacies), cells))

    return {
        "definition": (
            "legacy restamp key (model, op, node, shape, phase, parallel) "
            "with >1 dtype and >1 numerator pair"),
        "legacy_keys": sum(row.legacy_key_count for row in rows),
        "dtype_cells": sum(row.dtype_cell_count for row in rows),
        "models": [
            {
                "model": model,
                "legacy_keys": sum(r.legacy_key_count for r in model_rows),
                "dtype_cells": sum(r.dtype_cell_count for r in model_rows),
                "ops": [r.as_report() for r in model_rows],
            }
            for model, model_rows in sorted(
                (model, [r for r in rows if r.model == model])
                for model in {r.model for r in rows}
            )
        ],
        "collisions": [row.as_report() for row in rows],
    }


def _plan_snapshots(paths: List[Path]):
    for path in paths:
        jobs: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            jobs.extend(json.loads(line) for line in handle if line.strip())
        yield path.name[:-len(".jsonl")], jobs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="dtype-collision-census")
    parser.add_argument("--plans", required=True,
                        help="one jobs.jsonl or a directory of jobs.jsonl")
    parser.add_argument("--out", help="also write the JSON report here")
    args = parser.parse_args(argv)

    root = Path(args.plans)
    paths = ([root] if root.suffix == ".jsonl" else
             sorted(root.glob("*.jsonl")))
    if not paths:
        parser.error(f"no plans at {args.plans}")
    report = {**census_dtype_collisions(_plan_snapshots(paths)),
              "plans": [str(path) for path in paths]}

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(f"census: {len(paths)} plans, {len(report['models'])} affected models, "
          f"{report['legacy_keys']} legacy keys, "
          f"{report['dtype_cells']} dtype cells")
    print("model                                           op                          legacy_keys dtype_cells")
    for row in report["collisions"]:
        print(f"{row['model']:<47} {row['op']:<27} "
              f"{row['legacy_keys']:>11} {row['dtype_cells']:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
