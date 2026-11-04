"""Live registry snapshots for the op-proposal prompt.

Three readers:
  • `build_existing_ops_doc()` — full primitive table including shape_kind,
    shape_in, shape_out, secondary_outputs. Wider than the spec-worker
    table because the agent needs to judge "does an existing op cover
    this?" rigorously, not just pick one off a menu.
  • `read_dim_registry()` — the dim registry rendered as a table so the
    agent knows which short symbols are already taken (and what they
    mean) before proposing new ones.
  • `read_shape_kind_taxonomy()` — the nine shape_kind enum values + a
    one-line discriminator for each, pulled from ops.py docstrings.
"""
from __future__ import annotations

import inspect


def build_existing_ops_doc() -> str:
    """Full primitive table for the op-proposal agent.

    Surfaces shape_kind, shape_in, shape_out, secondary_outputs so the
    agent can judge whether an existing op already covers the signature
    in the gap statement. The spec-worker table omits these because the
    agent there is composing, not extending.
    """
    from backend.arch.ops import OP_REGISTRY

    lines = [
        "| Op | shape_kind | shape_in | shape_out | "
        "FLOPs | Params | Required Dims | Secondary outputs |",
        "|----|------------|----------|-----------|"
        "-------|--------|---------------|--------------------|",
    ]
    for name, spec in OP_REGISTRY.items():
        sk = getattr(spec.shape_kind, "value", spec.shape_kind)
        si = repr(spec.shape_in) if spec.shape_in is not None else "-"
        so = repr(spec.shape_out) if spec.shape_out is not None else "-"
        flops = (spec.flops_expr or "0").replace("|", "\\|")
        params = (spec.params_expr or "0").replace("|", "\\|")
        dims_str = ", ".join(spec.required_dims) if spec.required_dims else "-"
        secondary = (
            ", ".join(spec.secondary_outputs.keys())
            if spec.secondary_outputs else "-"
        )
        lines.append(
            f"| {name} | {sk} | {si} | {so} | "
            f"{flops} | {params} | {dims_str} | {secondary} |"
        )
    return "\n".join(lines)


def read_dim_registry() -> str:
    """Render the dim registry so the agent knows what symbols are taken
    before proposing new ones.

    Rendered from `dims.yaml` rather than read from a prose file: the
    symbols an agent must not collide with are exactly the registered ones,
    and a hand-kept document would answer a slightly older question.
    """
    from backend.arch import dims
    return dims.render_markdown().rstrip()


def read_shape_kind_taxonomy() -> str:
    """Return the live ShapeKind declaration and its discriminators."""
    from backend.arch.ops import ShapeKind

    return inspect.getsource(ShapeKind).rstrip()
