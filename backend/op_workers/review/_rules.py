"""Mechanical rule gate for auto-merge.

The LLM reviewer's `verdict == "auto_merge"` is necessary but not
sufficient. This module imposes additional deterministic checks on
`patch.py` so an LLM saying "looks fine" cannot unilaterally land a
complex op:

  1. `OP_REGISTRATIONS` contains exactly one `register_op(OpSpec(...))`
  2. `shape_kind` keyword is `preserve` or `axis_replace`
     (string literal OR `ShapeKind.PRESERVE` / `ShapeKind.AXIS_REPLACE`)
  3. No `secondary_outputs=` keyword on that OpSpec
  4. New dim symbols are allowed — they are this pipeline's job — but
     `NEW_DIM_SYMBOLS` and the proposal's `dims.yaml` fragment must name
     the same set. The fragment is incremental; it is not the canonical
     `dims.yaml`. Splice uses the same constructor as `dims._load`.

Anything outside this envelope falls through to human review. Multi-output
ops, `explicit`/`contract`/`source` shape kinds, or a new symbol without a
fragment still need a human.

Parsing strategy: ast.parse on the OP_REGISTRATIONS source string.
Regex would tangle on multiline lambdas and nested literals; AST is
robust to formatting.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import ast
import importlib.util
from pathlib import Path

from backend.arch.ops_base import LAYOUT_FAMILIES
from backend.arch.promote_op_proposal import read_artifacts, spliced_files


AUTO_MERGE_SHAPE_KINDS = {"preserve", "axis_replace"}


def _load_patch(patch_py: Path):
    spec = importlib.util.spec_from_file_location("op_proposal_patch_rules", patch_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    """Find all `<name>(...)` calls anywhere in the tree."""
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == name:
                out.append(node)
    return out


def _normalize_shape_kind(node: ast.expr) -> str | None:
    """Extract the shape_kind value from an OpSpec keyword arg.

    Accepts both forms seen in the registry:
      shape_kind="preserve"          → "preserve"
      shape_kind=ShapeKind.PRESERVE  → "preserve"
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute):
        # ShapeKind.PRESERVE → "preserve" (matches ShapeKind enum's lower-case values)
        return node.attr.lower()
    return None


def _new_symbol_fragment_problems(op_dir: Path, new_syms: list[str]) -> list[str]:
    """NEW_DIM_SYMBOLS and the fragment must name the same set."""
    fragment = op_dir / "dims.yaml"
    has_file = fragment.is_file()
    if not new_syms and not has_file:
        return []
    if not has_file:
        return [
            f"NEW_DIM_SYMBOLS {new_syms} need a dims.yaml fragment "
            "(kind, group, meaning per symbol)"
        ]
    import yaml
    try:
        body = load_yaml(fragment.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return [f"dims.yaml fragment does not parse: {e}"]
    if not isinstance(body, dict):
        return ["dims.yaml fragment must be a mapping of symbol -> entry"]
    frag_names = [str(k) for k in body]
    if set(frag_names) != set(new_syms):
        return [
            f"NEW_DIM_SYMBOLS {new_syms} != dims.yaml fragment keys {frag_names}"
        ]
    return []


def _layout_family_problems(kwargs: dict) -> list[str]:
    """A proposed op has no hand-written layout handler, so it declares the
    layout family that places it — the walk, validate and every multi-GPU
    analysis dispatch on it. `OpSpec` itself checks the declaration is
    complete when the registration runs (the splice below)."""
    node = kwargs.get("layout_family")
    family = node.value if isinstance(node, ast.Constant) else None
    if family not in LAYOUT_FAMILIES:
        return [
            "OpSpec declares no layout family: declare "
            "layout_family=\"linear\" for an op applying one [Out, In] "
            "weight (weight_shape=(Out, In), out_key=Out), or "
            "layout_family=\"recurrent\" for a head-wise recurrence "
            "(carries_recurrent_state=True and axis_use tagging two axes "
            "recurrent, one with param=\"state\"); an op of neither kind is "
            "expressed with registered ops"]
    if family == "recurrent" and "axis_use" not in kwargs:
        return ["a recurrent-family OpSpec declares axis_use for every "
                "couples_on axis"]
    return []


def check_auto_merge_eligibility(op_dir: Path | str) -> tuple[bool, list[str]]:
    """Return (eligible, reasons). `reasons` is empty when eligible; lists
    every disqualifier when ineligible (callers may surface them)."""
    op_dir = Path(op_dir)
    patch_py = op_dir / "patch.py"
    if not patch_py.is_file():
        return False, [f"missing {patch_py}"]

    try:
        patch = _load_patch(patch_py)
    except Exception as e:
        return False, [f"failed to load patch.py: {e}"]

    reasons: list[str] = []

    new_syms = [str(s) for s in (getattr(patch, "NEW_DIM_SYMBOLS", []) or [])]
    reasons.extend(_new_symbol_fragment_problems(op_dir, new_syms))

    registration_src = getattr(patch, "OP_REGISTRATIONS", "")
    if not registration_src.strip():
        return False, ["OP_REGISTRATIONS is empty"]

    try:
        tree = ast.parse(registration_src)
    except SyntaxError as e:
        return False, [f"OP_REGISTRATIONS does not parse: {e}"]

    register_calls = _find_calls(tree, "register_op")
    if len(register_calls) != 1:
        reasons.append(
            f"OP_REGISTRATIONS contains {len(register_calls)} register_op "
            f"call(s); auto-merge requires exactly 1"
        )
        # Continue scanning so the user sees all reasons in one report.

    opspec_calls = _find_calls(tree, "OpSpec")
    if not opspec_calls:
        reasons.append("no OpSpec(...) call found in OP_REGISTRATIONS")
        return False, reasons

    for opspec in opspec_calls:
        kwargs = {kw.arg: kw.value for kw in opspec.keywords if kw.arg}

        if "secondary_outputs" in kwargs:
            reasons.append("OpSpec has secondary_outputs (multi-output op)")

        shape_kind_node = kwargs.get("shape_kind")
        if shape_kind_node is None:
            reasons.append(
                "OpSpec omits shape_kind (default is EXPLICIT; "
                "auto-merge requires preserve or axis_replace)"
            )
        else:
            sk = _normalize_shape_kind(shape_kind_node)
            if sk is None:
                reasons.append(
                    "OpSpec.shape_kind is not a recognizable string or "
                    "ShapeKind.<NAME> form"
                )
            elif sk not in AUTO_MERGE_SHAPE_KINDS:
                reasons.append(
                    f"OpSpec.shape_kind={sk!r}; auto-merge requires one of "
                    f"{sorted(AUTO_MERGE_SHAPE_KINDS)}"
                )
        reasons.extend(_layout_family_problems(kwargs))

    # Same splice Open PR runs — `_insert_registry` now uses
    # `dims.validate_fragment` / `_parse_symbol`, i.e. `_load`.
    if not reasons:
        try:
            spliced_files(read_artifacts(op_dir))
        except Exception as e:
            reasons.append(
                f"proposal does not splice into canonical files: {e}"
            )

    return (not reasons), reasons
