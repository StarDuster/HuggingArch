#!/usr/bin/env python3
"""Fix gate for the `audit-curves` self-audit (SKILL.md step 2).

A curve-anomaly fix is only allowed to land when it is anchored by deterministic
evidence — otherwise an autonomous loop becomes the "confident error amplifier"
the validation system exists to prevent. This harness enforces the two anchors:

  1. the TARGETED anomaly goes red → green (sweep no longer flags it),
  2. the fix introduces NO new anomalies and keeps every golden/recon test green.

Usage (around an agent's fix):
    # before touching anything:
    python gate.py baseline --include <snaps> --out .audit_baseline.json
    #   ... agent applies the fix to the working tree ...
    python gate.py verify --baseline .audit_baseline.json --target "<key>" [--commit "msg"]

`--target` is a violated-axiom key `model|axiom_id` from the sweep (the
`verify` output also prints the live keys). Exit 0 = PASS (anchored), 1 = FAIL.
On PASS with `--commit`, stages everything and commits; on FAIL it leaves the
tree untouched for the agent to revert + file for human review.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# Accounting goldens + reconciliation anchors. Green here = the fix didn't move
# a reconciled byte / break the KV-read or axiom contracts.
GOLDENS = [
    "tests/backend/inference/test_capacity_golden.py", "tests/backend/arch/validate/test_validator_golden.py",
    "tests/test_kv_hybrid.py", "tests/test_kv_read_cardinality.py",
    "tests/backend/arch/validate/test_axioms.py", "tests/backend/inference/test_inference_invariants.py",
    "tests/backend/arch/validate/test_curve_axioms.py",
]


def _key(f: dict) -> str:
    return "|".join(str(f.get(k, "")) for k in ("model", "axiom_id"))


def _sweep(include: str | None) -> dict:
    cmd = [sys.executable, str(HERE / "sweep.py")]
    if include:
        cmd += ["--include", include, "--models", str(len(include.split(",")))]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    try:
        data = json.loads(out.stdout)
    except Exception:
        print(out.stdout[-2000:], file=sys.stderr)
        raise SystemExit("sweep produced no JSON") from None
    return {_key(f): f for f in data.get("failures", [])}


def _goldens() -> tuple[bool, str]:
    out = subprocess.run([sys.executable, "-m", "pytest", *GOLDENS, "-q"],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    tail = (out.stdout or "").strip().splitlines()[-1:] or [""]
    return out.returncode == 0, tail[0]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline"); b.add_argument("--include", default=None); b.add_argument("--out", default=".audit_baseline.json")
    v = sub.add_parser("verify"); v.add_argument("--baseline", default=".audit_baseline.json")
    v.add_argument("--target", required=True); v.add_argument("--include", default=None)
    v.add_argument("--commit", default=None)
    a = ap.parse_args()

    if a.cmd == "baseline":
        findings = _sweep(a.include)
        g_ok, g_tail = _goldens()
        (ROOT / a.out).write_text(json.dumps(
            {"include": a.include, "findings": sorted(findings), "goldens_ok": g_ok}, indent=2))
        print(f"baseline: {len(findings)} findings, goldens {'GREEN' if g_ok else 'RED — '+g_tail}")
        if not g_ok:
            print("  ⚠ baseline goldens already red — fix on a clean tree.")
        return

    base = json.loads((ROOT / a.baseline).read_text())
    base_keys = set(base.get("findings", []))
    cur = _sweep(a.include or base.get("include"))
    g_ok, g_tail = _goldens()
    target_gone = a.target not in cur
    new = sorted(set(cur) - base_keys)

    print(f"target '{a.target}': {'GONE ✓' if target_gone else 'STILL PRESENT ✗'}")
    print(f"new anomalies introduced: {new or 'none ✓'}")
    print(f"goldens: {'GREEN ✓' if g_ok else 'RED ✗ — '+g_tail}")
    if not target_gone:
        print("  live finding keys:", *sorted(cur), sep="\n    ")

    ok = target_gone and not new and g_ok and base.get("goldens_ok", False)
    if ok:
        print("\nGATE: PASS — fix is anchored (red→green + no regressions + goldens green).")
        if a.commit:
            subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
            subprocess.run(["git", "commit", "-q", "-m", a.commit], cwd=ROOT, check=True)
            print("  committed.")
        sys.exit(0)
    print("\nGATE: FAIL — not anchored. Revert the fix and file for human review.")
    sys.exit(1)


if __name__ == "__main__":
    main()
