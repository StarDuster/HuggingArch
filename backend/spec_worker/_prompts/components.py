"""Read the component library + example specs + live OP_REGISTRY into
prompt-ready strings.

Three independent readers used by the system prompt:
  • `read_all_components()` — manifest of available compound components
    (tier × role × name × description). YAML graphs are NOT inlined —
    the agent reads a file when it needs details. Keeps the system
    prompt compact as the library grows.
  • `read_example_specs()` — full content of 3 representative model
    YAMLs (Qwen2 dense, DSv3 MoE, MiMo-V2 hybrid).
  • `build_primitive_ops_doc()` — markdown table of primitives + FLOPs
    formulas + required dims, generated from the live OP_REGISTRY so it
    can never drift from ops.py.
  • `schema_registry_tables()` — the vocabulary tables `schema.md` quotes
    (roles, cache families and slot kinds, quant schemes, custom-op
    `shape_kind` / `input_roles`), rendered from their registries.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import os
from typing import Dict, List, Tuple

from backend.arch.cache_families import all_families, slot_kinds
from backend.arch.ops import OP_REGISTRY
from backend.arch.ops_base import INPUT_ROLES
from backend.arch.ops_registry import stripped_op_names
from backend.arch.quant_schemes import QUANT_SCHEMES, SCHEME_NONE
from backend.arch.roles import all_roles
from backend.arch.schema.custom_ops import CustomOpSpec

from ..runtime import _COMPONENTS_DIR, _MODELS_DIR


def read_all_components() -> str:
    """Build a manifest of available compound components — name, role,
    one-line description — grouped by tier (library/adapters/drafts) and then by
    role. Full YAML graphs are NOT inlined; the agent should Read the
    file when it needs details.

    Three physical tiers under `backend/arch/components/`:
      • `library/` — distinct topology building blocks. Compose these
        when authoring a new model. May carry a model-family prefix as
        a "first observed in" historical label, but are not scoped to
        that family.
      • `adapters/` — binding + `module`/`inner_modules` shims wrapping a
        library component for a specific model's checkpoint naming. No new
        structure of their own. Reuse only within the same model
        family; if a second family wants the same binding pattern,
        promote it to `library/` and rename.
      • `drafts/` — components already merged to main but not yet promoted.
        They are read-only inputs: reuse one only by listing its name in the
        target spec's top-level `drafts:` list.

    This keeps the spec-gen prompt compact as the library grows. A reachable
    component without a `description:` is reported by the validator's
    non-gating structural warnings so the manifest can be reviewed over time.
    """
    import yaml as _yaml

    def _scan_tier(tier: str) -> Dict[str, List[Tuple[str, str]]]:
        d = os.path.join(_COMPONENTS_DIR, tier)
        by_role: Dict[str, List[Tuple[str, str]]] = {}
        if not os.path.isdir(d):
            return by_role
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(d, fname)
            try:
                with open(path) as f:
                    data = load_yaml(f) or {}
            except (OSError, _yaml.YAMLError) as e:
                # Don't drop a broken component silently — the agent would
                # never know it exists and might reach for it blind. Surface
                # it in the manifest under a loud bucket so the agent sees
                # which file is unusable. (CI gates every component YAML on
                # parse; this covers the local mid-edit case.)
                by_role.setdefault("⚠ FAILED TO LOAD", []).append(
                    (fname, f"could not parse ({type(e).__name__}) — unusable until fixed")
                )
                continue
            role = data.get("role") or "(unroled)"
            name = data.get("name") or fname.removesuffix(".yaml")
            desc = (data.get("description") or "").strip() or "(no description)"
            by_role.setdefault(role, []).append((name, desc))
        return by_role

    role_order = ["block", "attention", "ffn_dense", "ffn_moe", "router",
                  "aux_block"]

    def _render(by_role: Dict[str, List[Tuple[str, str]]]) -> List[str]:
        seen = set(role_order)
        extra = sorted(r for r in by_role if r not in seen)
        lines: List[str] = []
        for role in role_order + extra:
            items = by_role.get(role) or []
            if not items:
                continue
            lines.append(f"#### {role} ({len(items)})")
            for name, desc in sorted(items):
                lines.append(f"- `{name}` — {desc}")
            lines.append("")
        return lines

    sections: List[str] = []
    for tier, blurb in (
        ("library", "Distinct topology building blocks. Compose these "
                    "for new models."),
        ("adapters", "Binding + `module`/`inner_modules` shims for a "
                     "specific model's checkpoint naming. Read for "
                     "reference; only reuse within the same model family."),
        ("drafts", "Merged but not promoted. Read-only; still requires an "
                   "explicit `drafts:` opt-in in the target spec."),
    ):
        by_role = _scan_tier(tier)
        if not by_role:
            continue
        sections.append(f"### {tier}/  — {blurb}")
        sections.extend(_render(by_role))
    return "\n".join(sections).rstrip()


def read_formal_bases() -> str:
    """List formal model-base stems the agent may reference with ``base:``.

    The reference workspace already mirrors ``backend/arch/models``. The list
    makes the namespace contract explicit: an unreviewed run-local model file
    cannot own a shared base name, while any stem shown here is a review-gated
    base the target identity may point at.
    """
    import yaml as _yaml

    lines: List[str] = []
    for fname in sorted(os.listdir(_MODELS_DIR)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        stem = fname.removesuffix(".yaml").removesuffix(".yml")
        path = os.path.join(_MODELS_DIR, fname)
        identity = "no identity claim"
        try:
            with open(path, encoding="utf-8") as f:
                raw = load_yaml(f) or {}
            match = raw.get("match") or {}
            if isinstance(match, dict):
                ids = match.get("model_id") or []
                types = match.get("model_type") or []
                ids = ids if isinstance(ids, list) else [ids]
                types = types if isinstance(types, list) else [types]
                identity = (
                    "representative identity: "
                    + str((ids or types or ["no identity claim"])[0])
                )
        except (OSError, _yaml.YAMLError, AttributeError):
            identity = "⚠ FAILED TO LOAD — unusable as a base"
        lines.append(f"- `{stem}` — {identity}")
    return "\n".join(lines)


def read_example_specs() -> str:
    """Read 2-3 representative model specs as examples."""
    examples = ["qwen2.yaml", "deepseek_v3.yaml", "mimo_v2_flash.yaml"]
    sections = []
    for fname in examples:
        path = os.path.join(_MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                content = f.read().strip()
            sections.append(f"### {fname}\n```yaml\n{content}\n```")
    return "\n\n".join(sections)


def build_primitive_ops_doc() -> str:
    """Generate primitive ops table from the live OP_REGISTRY.

    Pulls `required_dims` directly from each OpSpec — that's the
    authoritative declaration of what the call site must bind, not
    a regex over formula strings. Multi-output primitives surface
    their secondary port names so the agent knows the
    `<node>.<port>` reference syntax is available.

    Semantic `formula` (e.g. "A = Q · Kᵀ / sqrt(d_head)") is surfaced as
    its own column — distinct from `flops_expr` (which is the cost
    expression). `attention_score`'s flops and `matmul`'s flops both
    look like `2·B·n_heads·s_query·s_kv·d_head` and are useless for choosing between
    ops; the formula column tells the agent what each op actually
    computes.
    """
    # Guard-ablation: HA_STRIP_OPS hides model-specific primitives from the
    # agent's prompt too (the runner's in-process registry is full, so filter
    # here at call time to match the agent's import-stripped view). Inert unset.
    _stripped = stripped_op_names()
    lines = [
        "| Op | Computes | FLOPs | Params | Required Dims | Outputs |",
        "|----|----------|-------|--------|---------------|---------|",
    ]
    for name, spec in OP_REGISTRY.items():
        if name in _stripped:
            continue
        formula = (spec.formula or "—").replace("|", "\\|")
        flops = spec.flops_expr or "0"
        params = spec.params_expr or "0"
        dims_str = (", ".join(spec.required_dims)
                    if spec.required_dims else "-")
        # Multi-output ops: list `out + secondary ports` so the agent
        # knows it can reference `<node>.<port>`.
        if spec.secondary_outputs:
            ports = ["out", *list(spec.secondary_outputs.keys())]
            outputs = ", ".join(ports) + " (multi-output: ref as `<node>.<port>`)"
        else:
            outputs = "out"
        lines.append(
            f"| {name} | {formula} | {flops} | {params} | {dims_str} | {outputs} |"
        )
    return "\n".join(lines)


# --- registry tables the schema doc quotes --------------------------------
#
# Each table is rendered from the registry that owns the vocabulary, so the
# prompt cannot list fewer entries than the loader accepts (the hand tables
# in schema.md had drifted to 6 of 20 roles and 9 of 11 quant methods, and
# called the closed cache-family registry an open string).

def _cell(text: str) -> str:
    """One markdown table cell: a single line, pipes escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _codes(names) -> str:
    return ", ".join(f"`{n}`" for n in names)


def _yaml_scalar(value) -> str:
    """A default as the spec would write it (`true`, `null`), not Python's
    repr."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_roles_doc() -> str:
    """Every role in `roles.yaml`: its kind, the role it refines, meaning."""
    lines = ["| role | kind | refines | meaning |", "|---|---|---|---|"]
    for role in all_roles().values():
        refines = f"`{role.refines}`" if role.refines else "—"
        lines.append(f"| `{role.name}` | {role.kind} | {refines} | "
                     f"{_cell(role.meaning)} |")
    return "\n".join(lines)


def build_refining_roles_list() -> str:
    """The roles that refine a broader one, as an inline list."""
    return _codes(r.name for r in all_roles().values() if r.refines)


def build_cache_families_doc() -> str:
    """Every family in `cache_families.yaml` with its kind and meaning."""
    lines = ["| family | kind | meaning |", "|---|---|---|"]
    lines += [f"| `{fam.name}` | {fam.kind} | {_cell(fam.meaning)} |"
              for fam in all_families().values()]
    return "\n".join(lines)


def build_cache_slot_kinds_doc() -> str:
    """The cache slot kinds, one table per storage law: a `node:` entry
    holds a per-token kind, a `state:` entry a state kind."""
    tables = []
    for storage, entry in (("per_token", "node:"), ("state", "state:")):
        lines = [f"| `{entry}` entry kind | holds |", "|---|---|"]
        lines += [f"| `{k.name}` | {_cell(k.meaning)} |"
                  for k in slot_kinds().values() if k.storage == storage]
        tables.append("\n".join(lines))
    return "\n\n".join(tables)


def build_yaml_shape_kinds_list() -> str:
    """The `shape_kind` values a YAML custom op may declare."""
    # A pydantic private attribute: its declared value is the default.
    return _codes(
        CustomOpSpec.__private_attributes__["_YAML_SHAPE_KINDS"].default)


def build_input_roles_list() -> str:
    """The operand roles a custom op's `input_roles` may name."""
    return _codes(r.key for r in INPUT_ROLES)


def build_quant_schemes_doc() -> str:
    """Every registered quant scheme: the fields a `per_role` / override
    entry supplies, their defaults, the `format` / `strategy` values the
    scheme tells apart, and the sidecar tensors it implies per weight."""
    lines = [
        "| method | required fields | optional fields | defaults "
        "| `format` / `strategy` values | sidecars per weight |",
        "|---|---|---|---|---|---|",
        f"| `{SCHEME_NONE.name}` | — | — | — | — | — (stored unquantized) |",
    ]
    for name, scheme in sorted(QUANT_SCHEMES.items()):
        defaults = ", ".join(f"`{k}: {_yaml_scalar(v)}`"
                             for k, v in scheme.default_fields)
        values = []
        if len(scheme.format_variants) > 1:
            # `default` is the variant an entry without `format:` reads.
            values.append("format: " + " / ".join(
                "(omitted)" if v == "default" else f"`{v}`"
                for v in scheme.format_variants))
        if len(scheme.scale_strategy) > 1:
            values.append("strategy: " + " / ".join(
                f"`{v}`" for v in scheme.scale_strategy))
        sidecars = ", ".join(
            f"`{s.suffix}` ({s.kind}" + (f", when {s.when}" if s.when else "")
            + ")" for s in scheme.metadata_siblings)
        lines.append(
            f"| `{name}` | {_codes(scheme.required_fields) or '—'} "
            f"| {_codes(scheme.optional_fields) or '—'} | {defaults or '—'} "
            f"| {'; '.join(values) or '—'} | {_cell(sidecars) or '—'} |")
    return "\n".join(lines)


def schema_registry_tables() -> Dict[str, str]:
    """The placeholders `templates/schema.md` carries, each rendered from
    the registry that declares it."""
    return {
        "{roles_table}": build_roles_doc(),
        "{refining_roles}": build_refining_roles_list(),
        "{cache_families_table}": build_cache_families_doc(),
        "{cache_slot_kinds_table}": build_cache_slot_kinds_doc(),
        "{yaml_shape_kinds}": build_yaml_shape_kinds_list(),
        "{input_roles}": build_input_roles_list(),
        "{quant_schemes_table}": build_quant_schemes_doc(),
    }
