"""CLI entry: `python -m backend.arch.validate <model_id> [--spec NAME]`.

Pretty-prints the validate() result to stdout (JSON) with a structured
diagnostic summary on stderr when validation fails (param mismatches,
unmatched tensors, structural warnings, shape mismatches).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from ._orchestrator import validate


def _print_check_warnings(result):
    """Surface type-contract + module-execution warnings on BOTH pass and
    fail paths — a mislabeled KV topology or a module that doesn't compute is
    worth seeing even while the bytes diff is still open."""
    for key, header in (
        ("type_contract_warnings", "Type-contract warnings"),
        ("module_exec_warnings", "Module-execution warnings"),
        ("axiom_warnings", "Axiom violations (spec drifted from a source-locked fact)"),
        ("reuse_warnings", "Reuse (structure written twice — factor it)"),
        ("model_card_warnings", "Model card (stated parameter counts disagree)"),
    ):
        ws = result.get(key) or []
        if not ws:
            continue
        must_fix = [w for w in ws if not w.get("exempt")]
        info = [w for w in ws if w.get("exempt") or w.get("severity") == "info"]
        print(f"\n⚠ {header} ({len(must_fix)} to address"
              + (f", {len(info)} info/exempt" if info else "") + "):",
              file=sys.stderr)
        for w in ws:
            tmpl = w.get("template", "?")
            tag = ""
            if w.get("exempt"):
                tag = f" [exempt: {w.get('exempt_reason', '')}]"
            elif w.get("severity") == "info":
                tag = " [info]"
            elif w.get("new"):
                tag = " [NEW module]"
            print(f"    [{tmpl}] {w['kind']}{tag}: {w['detail']}", file=sys.stderr)


def main(downstream_gates=()):
    """The CLI. `downstream_gates` are passed to `validate` — the publication
    entry point (`python -m backend.inference.validate`) passes the analysis
    replay; this module's own entry point is the arch-only diagnostic."""
    parser = argparse.ArgumentParser(description="Validate spec against model")
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument("--spec", help="Spec name (auto-match if not given)")
    parser.add_argument("--overlay-dir", dest="overlay_dir", default=None,
                        help="Per-run overlay root (layers <dir>/models + "
                             "<dir>/components/drafts on the shared tree)")
    args = parser.parse_args()

    result = validate(args.model_id, args.spec, overlay_dir=args.overlay_dir,
                      downstream_gates=downstream_gates)
    print(json.dumps(result, indent=2, default=str))

    if not result["valid"]:
        print(f"\n❌ Validation FAILED:", file=sys.stderr)
        for e in result["errors"]:
            print(f"  - {e}", file=sys.stderr)

        # One-line diagnostic summary: if most param_mismatches cluster on a
        # single ratio, it's almost always a dim binding off by that factor.
        mismatches = result.get("param_mismatches") or []
        if mismatches:
            hints = [m.get("ratio_hint") for m in mismatches if m.get("ratio_hint")]
            if hints:
                counts = Counter(hints)
                top_hint, top_count = counts.most_common(1)[0]
                if top_count >= max(2, len(mismatches) // 2):
                    print(
                        f"  (diagnostic: {top_count}/{len(mismatches)} mismatched nodes are {top_hint} "
                        f"— likely one dim binding is off by {top_hint})",
                        file=sys.stderr,
                    )

        unmatched_spec = result.get("unmatched_spec_nodes") or []
        if unmatched_spec:
            print(f"\n  Unmatched SPEC nodes (node has params but no tensor found — "
                  f"typically a naming issue):", file=sys.stderr)
            for u in unmatched_spec:
                print(f"    [{u.get('template','?')}] {u['node_id']} "
                      f"(params={u['params']:,})", file=sys.stderr)

        if result.get("unmatched_hf_tensors"):
            print(f"\n  Unmatched HF tensors (tensor exists but no spec node covers it):", file=sys.stderr)
            for t in result["unmatched_hf_tensors"]:
                print(f"    {t['pattern']}: shape={t['shape']} ×{t['count']} = {t['params']:,} params", file=sys.stderr)

        if mismatches:
            print(f"\n  Per-node param mismatches (ratio = spec/hf):", file=sys.stderr)
            for m in mismatches:
                ratio_str = f" ratio={m.get('ratio')}" if m.get('ratio') is not None else ""
                hint_str = f" [{m['ratio_hint']}]" if m.get('ratio_hint') else ""
                print(f"    {m['spec_node']}: spec={m['spec_params']:,} "
                      f"hf={m['hf_params']:,} diff={m['diff']:+,}{ratio_str}{hint_str}", file=sys.stderr)

        if result.get("blocked_by_engine"):
            print("\n  Every error above is an analysis engine refusal: no spec "
                  "edit resolves it; the spec is blocked by the engine.",
                  file=sys.stderr)
        _print_check_warnings(result)
        sys.exit(1)
    else:
        print(f"\n✅ Validation PASSED: {result['spec_name']}", file=sys.stderr)
        # Surface structural warnings even on PASSED runs — validation
        # passing means "numbers match", but the agent may still have
        # mis-wired the DAG (missing residual / swapped norms). Warnings
        # don't fail the run; they're FYI for human review.
        sw = result.get("structural_warnings") or []
        if sw:
            print(f"\n⚠ Structural warnings ({len(sw)}; spec total matches but"
                  f" topology may diverge from forward):", file=sys.stderr)
            for w in sw:
                tmpl = w.get('template', '?')
                print(f"    [{tmpl}] {w['kind']}: {w['detail']}", file=sys.stderr)
        # Same for geometric (shape-consistency) warnings: numbers match
        # but a multi-input primitive op gets cross-input shapes that
        # don't agree (concat sees mismatched non-axis dims, contract
        # sees mismatched d_reduce, attention_score sees mismatched d_head, etc.).
        # Surfaces MLA-k_pe-style geometry bugs at validate time, not
        # at frontend render time.
        sm = result.get("shape_mismatches") or []
        if sm:
            print(f"\n⚠ Shape mismatches ({len(sm)}; cross-input geometry "
                  f"check on multi-input ops):", file=sys.stderr)
            for m in sm:
                tmpl = m.get('template', '?')
                node = m.get('node', '?')
                op = m.get('op', '?')
                kind = m.get('kind', '?')
                shapes = m.get('shapes', [])
                print(f"    [{tmpl}] {node} ({op}/{kind}): {m['msg']}", file=sys.stderr)
                for j, s in enumerate(shapes):
                    print(f"        input{j} = {s}", file=sys.stderr)
        _print_check_warnings(result)


if __name__ == "__main__":
    main()
