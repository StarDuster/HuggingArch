"""Promote an op-proposal artifact into the registry — CLI.

Thin wrapper around `backend.arch.promote_op_proposal.run_promote`.
Library form lives there so the op_workers.review route can splice/pytest/revert
in-process without shelling out.

Usage:
  uv run python scripts/promote_op_proposal.py <op_name> [--apply]

Without `--apply` the script previews artifacts and exits without
writing.
"""
from __future__ import annotations

import argparse
import sys

from backend.arch.promote_op_proposal import preview, run_promote


def _show_preview(artifacts: dict) -> None:
    print("=" * 70)
    print("Proposal preview")
    print("=" * 70)
    print(f"  Section comment:  {artifacts['section_comment']!r}")
    print(f"  New dim symbols:  {artifacts['symbols']}")
    print(f"  Registration:     {len(artifacts['registration'])} chars")
    print(f"  Glossary section: {len(artifacts['glossary_section'])} chars")
    print(f"  Tests source:     {len(artifacts['tests_src'])} chars")
    print()
    if artifacts["registration"].strip():
        print("--- OP_REGISTRATIONS (first 600 chars) ---")
        print(artifacts["registration"][:600])
        print("---")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op_name", help="snake_case op name (proposal dir name)")
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually apply the splices. Without this, preview only.",
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Skip pytest verification after applying (NOT recommended).",
    )
    args = parser.parse_args()

    if not args.apply:
        try:
            artifacts = preview(args.op_name)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        _show_preview(artifacts)
        print()
        print("Preview only. Re-run with --apply to splice into ops.py / "
              "dims.yaml / tests/.")
        return 0

    result = run_promote(args.op_name, skip_tests=args.skip_tests)
    if result.artifacts:
        _show_preview(result.artifacts)
    print()
    if result.pytest_stdout:
        sys.stdout.write(result.pytest_stdout)
    if result.pytest_stderr:
        sys.stderr.write(result.pytest_stderr)

    if not result.ok:
        print(f"[apply] FAILED at {result.stage}: {result.error}", file=sys.stderr)
        return 1
    if result.stage == "test":
        print(f"[test] FAILED — splices reverted. {result.message}", file=sys.stderr)
        return 1
    if result.stage == "splice":
        print(f"[skip-tests] {result.message}")
        return 0
    print(f"[done] {args.op_name} {result.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
