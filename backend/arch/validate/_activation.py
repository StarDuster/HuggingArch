"""Soft-fidelity checks that no numeric anchor can catch.

The bytes / params / coverage gates only see WEIGHTS and SHAPES. Two facts a
spec can get wrong while every hard anchor stays green:

  • the FFN activation function — `silu` vs `gelu` vs `gelu_tanh` vs a custom
    act. Zero params, negligible FLOPs, so it's invisible to coverage / bytes /
    params. `config.hidden_act` is the model author's declared ground truth for
    it, and each activation op declares the config names it computes
    (`OpSpec.config_names`), so the check is a lookup.
  • whether the forward() source was even available to model the block
    topology from. When it's missing (no repo code / unreleased transformers)
    or incomplete (decoder imported from an unfetched sibling), the agent
    modeled the wiring from config + weights alone — the topology may be
    approximate. Better surfaced loudly than silently authoritative.

A third fact the config names outright gates: a component param slot that
declares the config field choosing its op (`ParamSpec.config`, e.g. the GDN
norm gate's `output_gate_type`) must be bound to that op —
`run_param_config_check`. There the config states the choice and the spec
binds it, so a disagreement is a wrong binding, not an approximation.

The first two are WARNINGS, never gating: a wrong activation or an unmodeled
cross-block dataflow doesn't move the cost model meaningfully, and refusing to
publish a byte-correct spec over a soft-fidelity gap is the wrong trade. They
exist so the gap is EXPLICIT — for the refine agent to close and for the UI to
badge — not silent.
"""
from __future__ import annotations

from typing import Any, Dict, List
import backend.arch.roles as roles
from backend.arch.dim_resolution import language_config
from backend.arch.eval.program import Program
from backend.arch.ops import OP_REGISTRY, op_for_config_name

from backend.hf.forward_source import describe_forward_source

from ._finding import Finding

# Gating is a property of the FFN compound (`swiglu_ffn`, `gated_ffn`), not
# of an activation: the op inside it is the pointwise function alone — `silu`,
# `gelu` (erf form), `gelu_tanh` (tanh approximation), `relu_squared`, `situ`.
#: Every op that declares itself an activation. A QUERY over the registry,
#: not a literal: an op registered tomorrow is classified by what it says
#: about itself, and a wrong classification is a wrong predicate — visible,
#: and testable — rather than an absent name, which is neither.
_ACT_OPS = frozenset(n for n, o in OP_REGISTRY.items() if o.is_activation)

# config scoring-func string → the spec op that models it.
_SCORING_TO_OP = {
    "sigmoid": "sigmoid",
    "softmax": "softmax_act",
    "sqrtsoftplus": "softplus",
    "softplus": "softplus",
}


def _config_field(fields: Any, keys) -> Any:
    """The first of `keys` (one field's vendor spellings) the language
    model's config fields set (`dim_resolution.language_config`)."""
    for k in keys:
        v = fields.get(k)
        if v:
            return v
    return None


def _config_scoring_func(fields: Any) -> str | None:
    """The MoE router's scoring nonlinearity, however the vendor names it."""
    return _config_field(fields, ("scoring_func", "moe_router_activation",
                                  "moe_router_activation_func"))


def _config_hidden_act(fields: Any) -> str | None:
    """The FFN activation the language model's config fields name."""
    return _config_field(fields, ("hidden_act", "hidden_activation",
                                  "activation_function"))


def _op_for_config_act(name: str | None) -> str | None:
    """The registered activation a config's `hidden_act` string names — the
    op whose declared `config_names` contain it (`gelu` is the erf form,
    `gelu_pytorch_tanh` the tanh approximation). None when no op declares
    the name (a custom activation the spec can only approximate)."""
    return op_for_config_name(name)


# Component roles that make an enclosed activation an FFN GATE nonlinearity
# (the `silu`/`gelu`/`situ` in a gated-FFN's gate_proj→act→mul). Scoping the
# FFN-gate check to these is what keeps ATTENTION-internal gates (KDA/GLA
# `q_act`/`g_act`, role `attention`) and VISION-tower activations (`mlp.gelu`,
# role `block`) — which are NOT governed by the text `config.hidden_act` — from
# being mis-read as the FFN gate. Without it a faithful situ/linear-attention
# model false-warns just for having a silu somewhere in its attention.


def _spec_activation_ops(resolved) -> tuple[set, set]:
    """Activation ops in the spec's EXPANDED block templates (component-internal
    acts only surface after expansion). Returns `(all_ops, ffn_gate_ops)`:

      • `all_ops` — every activation op anywhere (used for the router-scoring
        check, whose sigmoid/softplus live in the router component).
      • `ffn_gate_ops` — only acts whose ENCLOSING component role is an FFN role
        (`_FFN_GATE_ROLES`), i.e. the actual SwiGLU/GeGLU/SituAndMul gate. This
    is what `config.hidden_act` speaks to."""
    from backend.arch.eval.projection import project_blocks
    from ._module_exec import _S0, _top_templates
    blocks = project_blocks(resolved, input_tokens=_S0)

    all_ops: set = set()
    ffn_ops: set = set()

    def _walk(nodes, in_ffn: bool):
        for n in nodes:
            role = n.role
            here_ffn = in_ffn or (
                n.is_component and role in roles.ffn_gate_roles())
            op = n.op
            if op in _ACT_OPS:
                all_ops.add(op)
                if here_ffn:
                    ffn_ops.add(op)
            kids = n.children
            if kids:
                _walk(kids, here_ffn)

    for tmpl in _top_templates(resolved):
        r = blocks.get(tmpl)
        if r is None:
            continue
        _walk(r.nodes, False)
    return all_ops, ffn_ops


def run_activation_check(resolved, config) -> List[Dict[str, Any]]:
    """Warn when a declared activation disagrees with what the spec models:
    (a) the FFN gate vs `config.hidden_act` (the op whose declared
    `config_names` contain it), and (b) the MoE
    router scoring nonlinearity vs `config.scoring_func` /
    `moe_router_activation` (sigmoid / softmax / sqrtsoftplus). Both are
    param-free and cost-negligible → invisible to every numeric anchor."""
    all_ops, ffn_gate_ops = _spec_activation_ops(resolved)
    fields = language_config(resolved.spec.dims, config)[1]
    out: List[Dict[str, Any]] = []

    # (a) FFN gate — scoped to acts inside FFN-role components, so an
    # attention-internal silu (KDA/GLA gate) or a vision-tower gelu can't be
    # mis-read as the text FFN gate that `config.hidden_act` governs. Only
    # activations a config can name take part: the gate is one of them.
    raw = _config_hidden_act(fields)
    if raw:
        cfg_op = _op_for_config_act(raw)
        gate_ops = {o for o in ffn_gate_ops if OP_REGISTRY[o].config_names}
        if gate_ops and cfg_op not in gate_ops:
            if cfg_op is None:
                detail = (
                    f"config.hidden_act = {raw!r} is a non-standard activation "
                    f"with no matching op — the spec approximates it with "
                    f"{sorted(gate_ops)}. Model the real activation or note it.")
            else:
                detail = (
                    f"config.hidden_act = {raw!r} is op {cfg_op!r} but the "
                    f"spec's FFN gate uses {sorted(gate_ops)}. Bind the FFN's "
                    f"activation to {cfg_op!r} to match the checkpoint.")
            out.append(Finding(
                kind="activation_mismatch", detail=detail,
                extra={"config_hidden_act": raw,
                       "spec_gate_activation": sorted(gate_ops)}))

    # (b) MoE router scoring nonlinearity.
    scoring = _config_scoring_func(fields)
    if scoring:
        expected = _SCORING_TO_OP.get(str(scoring).lower())
        # The ops that CAN be a router's scoring function are exactly the
        # ones `_SCORING_TO_OP` maps a config value onto. Spelled as its own
        # set it was a second copy of the same three names, free to drift
        # from the mapping that decides what `expected` is compared against.
        router_ops = all_ops & frozenset(_SCORING_TO_OP.values())
        if expected and expected not in router_ops:
            if not router_ops:
                detail = (
                    f"config router scoring = {scoring!r} but the spec models NO "
                    f"router nonlinearity (routing abstracted to linear+bias). "
                    f"Add the {expected!r} scoring op to match the checkpoint.")
            else:
                detail = (
                    f"config router scoring = {scoring!r} (expects {expected!r}) "
                    f"but the spec's router uses {sorted(router_ops)}. Swap the "
                    f"router component to match the checkpoint.")
            out.append(Finding(
                kind="router_activation_mismatch", detail=detail,
                extra={"config_scoring_func": scoring,
                       "spec_router_ops": sorted(router_ops)}))

    return out


def run_forward_source_completeness_check(model_id, resolved) -> List[Dict[str, Any]]:
    """Warn when forward() source was missing or incomplete — the block
    topology couldn't be verified against reference code and may be
    approximate. Non-gating: the weight-derived numbers are still trustworthy;
    this flags the SOFT (dataflow/topology) layer as unverified."""
    try:
        d = describe_forward_source(model_id, resolved)
    except Exception as exc:  # network / parse: the source is unverifiable, say so
        return [Finding(
            kind="forward_source_unavailable",
            detail=f"describe_forward_source({model_id!r}) raised "
                   f"{type(exc).__name__}: {exc}")]
    if not d.get("source") or not d.get("files"):
        return [Finding(
            kind="forward_source_missing",
            detail=(
                f"No forward() source available for {model_id} — block topology "
                "(residual/norm placement, gating, cross-block dataflow, "
                "activation) was modeled from config + weights alone and is "
                "APPROXIMATE. Params/bytes/coverage remain trustworthy."),
        )]
    if d.get("incomplete"):
        return [Finding(
            kind="forward_source_incomplete",
            detail=(
                "Forward source is incomplete — the decoder/backbone is imported "
                "from a module that couldn't be fetched (see the source banner). "
                "Block topology may be APPROXIMATE; weight-derived numbers are "
                "still trustworthy."),
        )]
    return []


def run_param_config_check(resolved, config) -> List[Finding]:
    """A component param slot that declares the config fields naming its
    value (`ParamSpec.config`) is bound to the op the config names.

    The GDN norm gate is the case in the library: vLLM and sglang build
    RMSNormGated with `activation=config.output_gate_type` (silu when the
    field is absent), and the components take it as `gate_act`. A config
    saying `sigmoid` with the slot left at its silu default costs nothing a
    byte or FLOP anchor can see, so this compares the two directly. The
    slot's bound op is read off the expanded graph — the children the
    component writes as `op: <slot>` — so a binding passed through an
    enclosing component's own param is read as resolved."""
    program = Program.of(resolved)
    scope, fields = language_config(resolved.spec.dims, config)
    where = f"config.{scope}." if scope else "config."
    out: List[Finding] = []
    seen = set()
    for node in program.nodes():
        if not node.is_component:
            continue
        try:
            graph = program.spec.get_block(node.op)
        except (KeyError, AttributeError):
            continue
        for pname, pspec in (graph.params or {}).items():
            if not pspec.config:
                continue
            raw = _config_field(fields, pspec.config)
            if not raw:
                continue
            want = op_for_config_name(str(raw))
            slots = {n for n, gn in graph.graph.items() if gn.op == f"<{pname}>"}
            bound = sorted({c.op for c in node.children if c.name in slots})
            if want is not None and bound == [want]:
                continue
            key = (node.template, node.display_path, pname)
            if key in seen:
                continue
            seen.add(key)
            field = next(k for k in pspec.config if _config_field(fields, (k,)))
            if want is None:
                detail = (
                    f"[{node.template or 'model'}] {node.display_path} "
                    f"({node.op}): {where}{field} = {raw!r} names no "
                    f"registered op (no op declares it in `config_names`), "
                    f"so `{pname}` cannot be bound to it. Register the "
                    f"activation, or bind `{pname}` to the op that computes "
                    f"{raw!r}.")
            else:
                detail = (
                    f"[{node.template or 'model'}] {node.display_path} "
                    f"({node.op}): {where}{field} = {raw!r} is op {want!r}, "
                    f"but `{pname}` is bound to {bound}. Bind it on the call "
                    f"site: `params: {{ {pname}: {want} }}`.")
            out.append(Finding(
                kind="param_config_mismatch", template=node.template or None,
                node=node.display_path, op=node.op, detail=detail,
                extra={"param": pname, "config_field": field,
                       "config_value": raw, "bound": bound}))
    return out
