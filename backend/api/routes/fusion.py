"""Fusion-pattern registry + per-model fire-status endpoints.

Two read-only endpoints powering the Fusion Patterns page:

  • GET /fusion/registry            — list every fusion plan + every
                                      framework binding (catalog view).
  • GET /fusion/fired?model=...&framework=vllm
                                    — rewrite the model's evaluated graph
                                      resolved spec and report which
                                      plans fired, the anchor leaves,
                                      and the folded sibling spec_ids.

Both are intentionally cheap — the fired endpoint runs the analyzer at
batch=1 and one token, once per language-model phase (prefill, decode),
and on ONE rank: every world runs at degree 1. Matching reads the topology,
the phase of the walk (`when: phase`) and the deployment the query names
(GPU device, activation dtype, server flags). A `when: world_local`
condition depends on the world degrees the endpoint does not take, so the
answer is the single-rank projection: such an entry is listed in
`world_conditioned`, and whether it fires at a given TP is the estimate's
walk to answer.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.arch.schema.fusion import PatternNode

from fastapi import HTTPException

from backend import analyzer
from backend.arch.eval.precision import ResolvedPrecision
from backend.arch.eval.program import Program
from backend.arch.loader import SpecLoader
from backend.arch.ops import is_comm_op
from backend.arch.sharding import DECODE, PREFILL
from backend.inference.framework_bindings import (
    DEFAULT_FRAMEWORK, FRAMEWORK_BINDINGS, ServingDeployment, get_binding)
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.rendering._consts import STYLE_BY_OP

from .._app import app
from ..offload import offload_response
from .._models import precision_from_query


_PLANS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "inference", "fusion_plans",
)
_BINDINGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "inference", "framework_bindings",
)


def _leading_comment_block(yaml_path: str) -> str:
    """Return the leading `# ...` comment block from a yaml file, with the
    `# ` prefix stripped and blank lines preserved. Stops at the first
    non-comment, non-blank line. Returns "" when file missing / no header.
    """
    if not os.path.isfile(yaml_path):
        return ""
    lines: List[str] = []
    with open(yaml_path) as f:
        for raw in f:
            s = raw.rstrip("\n")
            stripped = s.lstrip()
            if stripped.startswith("#"):
                # Drop one leading '#' and an optional single space.
                body = stripped[1:]
                if body.startswith(" "):
                    body = body[1:]
                lines.append(body)
                continue
            if stripped == "":
                # Blank line inside the comment header: keep as a paragraph break.
                if lines:
                    lines.append("")
                continue
            break
    # Trim trailing blanks
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _pattern_style(op: Optional[str], op_choices) -> str:
    """The style class a pattern node draws in, as the compute graph draws
    the same op: a collective as `comm` (its declaration), else the
    renderer's op table (`STYLE_BY_OP`), else `op`. A node that takes one of
    several ops draws in the style they agree on, ignoring the plain `op`
    style; when they disagree it is a plain `op`."""
    def one(name: str) -> str:
        return "comm" if is_comm_op(name) else STYLE_BY_OP.get(name, "op")
    if op:
        return one(op)
    styles = {one(name) for name in (op_choices or ())} - {"op"}
    return styles.pop() if len(styles) == 1 else "op"


def _serialize_pattern_node(pn: "PatternNode") -> Dict[str, Any]:
    return {
        "op": pn.op,
        "op_choices": list(pn.op_choices) if pn.op_choices else None,
        "style": _pattern_style(pn.op, pn.op_choices),
        "inputs": list(pn.inputs),
        "optional_inputs": (list(pn.optional_inputs)
                            if pn.optional_inputs else None),
        "optional": pn.optional,
    }


def _serialize_naive_pattern(pattern: Any) -> Dict[str, Any]:
    """Flatten NaivePattern into a JSON-serializable shape the frontend
    can directly turn into ELK nodes+edges. Two kinds:

      - kind="nodes": explicit DAG, `nodes` is the flattened map.
      - kind="parallel": variable-arity sibling form (rare).

    `edges` is emitted explicitly here — one `{src, dst}` per consumer
    input ref — exactly like the main compute graph's `build_edges`
    (`backend/rendering/_shapes.py`). The connectivity is OUR contract,
    not something the frontend re-derives from each node's `inputs[]`
    (renderDag draws edges strictly from the `edges[]` array). `src` is
    the raw input ref (`external.x` / `producer.port` / `producer`); the
    frontend maps it into its node-id namespace with one shared resolver.
    """
    if pattern.is_parallel():
        pp = pattern.parallel
        return {
            "kind": "parallel",
            "parallel": {
                "op": pp.op,
                "op_choices": list(pp.op_choices) if pp.op_choices else None,
                "style": _pattern_style(pp.op, pp.op_choices),
                "shared_inputs": list(pp.shared_inputs),
                "min_count": pp.min_count,
                "max_count": pp.max_count,
            },
            "edges": [{"src": inp, "dst": "siblings"} for inp in pp.shared_inputs],
            "io_mapping": dict(pattern.io_mapping),
            "external_inputs": pattern.external_input_names(),
        }
    nodes_out: Dict[str, Any] = {}
    edges_out: List[Dict[str, str]] = []
    for name, pn in pattern.expanded_nodes().items():
        nodes_out[name] = _serialize_pattern_node(pn)
        edges_out.extend({"src": inp, "dst": name} for inp in pn.inputs)
    return {
        "kind": "nodes",
        "nodes": nodes_out,
        "edges": edges_out,
        "io_mapping": dict(pattern.io_mapping),
        "external_inputs": pattern.external_input_names(),
    }


def _backends_for_fused_op(fused_op: str) -> Optional[Dict[str, Any]]:
    """Which measured backend kernels can realize this fused op — enumerated
    live from the kernel_bench provider registry (one op-signature, many
    library implementations). Returns None for ops with no signature/providers
    (router, composite FFN, …). Light import: provider discovery pulls no heavy
    libraries (torch/etc. are imported lazily inside each provider's build)."""
    try:
        from kernel_bench.providers import all_providers
        from kernel_bench.signatures import op_to_signature
    except Exception:
        return None
    # A span named after an op realizes as that op's signature; a span the op
    # table does not name (`grouped_gemm_moe`) realizes as the signature of
    # the kernels that declare they absorb it (`Provider.fused_op`).
    sig = op_to_signature(fused_op)
    if not sig:
        claimed = {p.signature for p in all_providers() if p.fused_op == fused_op}
        if len(claimed) > 1:
            raise RuntimeError(
                f"kernels of signatures {sorted(claimed)} all claim fused op "
                f"{fused_op!r}; a fused op realizes as one signature")
        sig = next(iter(claimed), None)
    if not sig:
        return None
    providers = [
        {
            "library": p.library,
            "kernel_impl": p.kernel_impl,
            "provisioning": "source_build" if p.source_build else "pip",
            "pip": " ".join(p.pip),
            # The fusion boundary this kernel absorbs (its span). "" ⇒ a 1-op
            # span (covers exactly `signature`); a wider value (e.g. fused_moe)
            # ⇒ the kernel fuses several canonical ops into one launch.
            "fused_op": p.fused_op or fused_op,
            # What the kernel DECLARES it can serve — dtypes, phases, shape
            # constraints. `None` while a kernel is still on a hand-written
            # gate: an undeclared scope is published as unknown rather than
            # guessed at, so the UI can say "not declared yet" instead of
            # implying the kernel serves everything.
            "capabilities": p.capabilities(),
            # Minimum compute capability, when the kernel has a floor (e.g. 100
            # for a Blackwell-only kernel). None ⇒ no architecture constraint.
            "min_sm": p.min_sm,
        }
        for p in all_providers()
        if p.signature == sig
    ]
    if not providers:
        return None
    providers.sort(key=lambda d: (d["library"], d["kernel_impl"]))
    return {"signature": sig, "providers": providers}


def _no_backend_reasons() -> Dict[str, str]:
    """Why a fused op has no measurable backend kernel — shown when
    `backends` is empty so the UI explains the gap instead of a generic
    "wait for a provider". These are NOT missing coverage: they are
    structural (no single kernel exists) or need a harness kernel_bench
    does not have yet (a multi-GPU collective).

    Keyed by the plan registry: a note whose plan was renamed or removed is
    refused here, at import, rather than sitting unread beside the plan it
    no longer names.
    """
    notes = {
        "gated_act_ffn": (
            "Composite op, not a single kernel — vLLM/sglang run the SwiGLU FFN as "
            "separate gate+up GEMM, silu_and_mul, and down GEMM (each measured under "
            "its own signature). There is no one kernel to benchmark here."
        ),
    }
    unknown = sorted(set(notes) - set(FUSION_SCHEMES))
    if unknown:
        raise ValueError(
            f"no-backend note(s) for {unknown}, which name no fusion plan "
            f"(backend/inference/fusion_plans/)")
    return notes


_NO_BACKEND_REASON: Dict[str, str] = _no_backend_reasons()


def _serialize_plan(fused_op: str, scheme: Any) -> Dict[str, Any]:
    plan_file = os.path.normpath(os.path.join(_PLANS_DIR, f"{fused_op}.yaml"))
    rel = os.path.relpath(
        plan_file,
        os.path.normpath(os.path.join(_PLANS_DIR, "..", "..", "..")),
    )
    return {
        "fused_op": fused_op,
        "plan_file": rel,
        "description": _leading_comment_block(plan_file),
        "naive_pattern": _serialize_naive_pattern(scheme.naive_pattern),
        # Measured backend kernels that realize this op (kernel_bench providers).
        "backends": _backends_for_fused_op(fused_op),
        # When there are no backends, why (structural / harness gap) — else "".
        "backends_note": _NO_BACKEND_REASON.get(fused_op, ""),
    }


def _serialize_binding(binding: Any) -> Dict[str, Any]:
    yaml_name = f"{binding.framework}.yaml"
    plan_file = os.path.normpath(os.path.join(_BINDINGS_DIR, yaml_name))
    rel = os.path.relpath(
        plan_file,
        os.path.normpath(os.path.join(_BINDINGS_DIR, "..", "..", "..")),
    )
    ops_out: List[Dict[str, str]] = []
    for op_name, impl in binding.op_implementations:
        ops_out.append({
            "fused_op": op_name,
            "source_anchor": (None if impl.source_anchor is None else
                              binding.audited_at.cite(impl.source_anchor)),
            # One condition as its mapping; alternatives as a list of them.
            "when": (None if not impl.whens
                     else impl.whens[0].as_dict() if len(impl.whens) == 1
                     else [when.as_dict() for when in impl.whens]),
        })
    return {
        "framework": binding.framework,
        "audited_at": {"repo": binding.audited_at.repo,
                       "tag": binding.audited_at.tag,
                       "commit": binding.audited_at.commit},
        "label": binding.audited_at.label(binding.framework),
        "binding_file": rel,
        "description": _leading_comment_block(plan_file),
        "ops": ops_out,
        # The paged-KV page this framework allocates the cache in by
        # default. Surfaced so a client can show the default beside the
        # `kv_block_size` request field instead of hard-coding 16.
        "kv_block_size": binding.kv_block_size,
    }


@app.get("/fusion/registry")
async def fusion_registry():
    """Return every fusion plan + every framework binding shipped.

    Response shape:
      {
        "plans":    [ {fused_op, plan_file, description, naive_pattern}, ... ],
        "bindings": [ {framework, audited_at, label, binding_file, description,
                       ops, kv_block_size}, ... ],
        "default_framework": "vllm",
        "served_kv_block_sizes": [16, 32, 64, 128, 256]
      }
    Plans + bindings are sorted (plans by fused_op, bindings by framework)
    so the catalog UI ordering is stable.

    `default_framework` is the binding an estimate runs under when the request
    names none, so a client can label the unchosen framework (and the page
    that framework declares) with the one the server would actually pick.
    `served_kv_block_sizes` is the union of the paged-KV pages the registered
    kernel backends declare (`kernel_bench.providers.served_kv_block_sizes`),
    so the page-size control offers what something can measure instead of a
    list copied into the client.
    """
    from kernel_bench.providers import served_kv_block_sizes
    plans = [
        _serialize_plan(fused_op, scheme)
        for fused_op, scheme in sorted(FUSION_SCHEMES.items())
    ]
    bindings_sorted = sorted(
        FRAMEWORK_BINDINGS.values(),
        key=lambda b: b.framework,
    )
    bindings = [_serialize_binding(b) for b in bindings_sorted]
    return {"plans": plans, "bindings": bindings,
            "default_framework": DEFAULT_FRAMEWORK,
            "served_kv_block_sizes": list(served_kv_block_sizes())}


@app.get("/fusion/fired")
async def fusion_fired(
    model: str,
    framework: str = DEFAULT_FRAMEWORK,
    gpu_sku: Optional[str] = None,
    framework_flags: str = "",
    precision: Optional[str] = None,
):
    """Rewrite `model` with the selected framework plans and report which plans
    fired against the framework's binding. A retired `version` query
    parameter is ignored: bindings are not versioned.

    A binding entry that the framework dispatches only under conditions
    (`when:` — a server flag, the activation dtype, the device) is answered
    for the deployment the query describes: `gpu_sku` names the GPU (its
    `device`), the activation is the analysis's 2-byte BF16, and
    `framework_flags` is a comma-separated list of server flags turned on
    (`enable_fused_qk_norm_rope`). Without a `gpu_sku` the device is
    unknown, and a device-gated kernel is not claimed.

    Response shape:
      {
        "model": "...",
        "framework": "...",
        "audited_at": "<release tag>",
        "fired": {
          "<fused_op>": {
            "match_count": int,
            "matches": [
              {"anchor": "<fused_aggregation_id>",
               "matched": ["<agg_id_1>", "<agg_id_2>", ...]}
            ]  // up to 8 samples
          }, ...
        },
        "unfired_bound": [ "<fused_op>", ... ],  // bound by this framework, no match on this model in any phase
        "unbound":       [ "<fused_op>", ... ],  // in the registry, not bound by this framework
        "phases": {
          "prefill" | "decode": {
            "fired":         [ "<fused_op>", ... ],
            "not_in_phase":  [ "<fused_op>", ... ],  // bound, `when: phase` excludes this phase
            "unfired_bound": [ "<fused_op>", ... ]   // bound for this phase, no match
          }
        }
      }
    plus `"projection": "single_rank"` and `"world_conditioned": [...]`, the
    bound entries whose `when: world_local` was answered at degree 1 only.
    Each `fired` entry also carries `phases: {phase: match_count}`; its
    `match_count` and `matches` are from the first phase (prefill, then
    decode) it fired in.
    """
    if not model:
        raise HTTPException(status_code=400, detail="model query param required")
    try:
        profile = precision_from_query(precision)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return await offload_response(_fusion_fired, model, framework, gpu_sku,
                         framework_flags, profile)


def _fusion_fired(model: str, framework: str, gpu_sku: Optional[str],
                  framework_flags: str, precision: Any) -> Dict[str, Any]:
    """The `/fusion/fired` answer; runs in a compute worker (the rewrite
    walks the whole model)."""
    binding = get_binding(framework)
    if binding is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No framework binding shipped for {framework!r}. "
                f"Available: {sorted(FRAMEWORK_BINDINGS)!r}"
            ),
        )
    # Resolve spec via the same path the rest of the inference API uses.

    config = analyzer.fetch_config(model)
    if config is None:
        raise HTTPException(
            status_code=404, detail=analyzer.explain_config_fetch_failure(model),
        )
    resolved = SpecLoader().match(config, model_id=model)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No spec matched config for {model!r}. "
                "Add a model spec or run /generate-spec first."
            ),
        )

    # Run the pipeline at a minimal workload (batch=1, one token) once per
    # language-model phase: a binding entry may run only in one phase
    # (`when: phase`), and a prefill walk and a decode walk are rewritten
    # separately, as an estimate rewrites them. No distributed deployment.
    # The kernels the framework runs only under conditions (`when:`) are
    # answered for the deployment the query describes: the GPU's device,
    # the model's activation dtype under the profile, the server flags on.
    binding = binding.deployed(ServingDeployment.at(
        gpu_sku=gpu_sku,
        activation_dtype=ResolvedPrecision(precision, Program.of(resolved)).model_dtype,
        server_flags={f.strip(): True
                      for f in framework_flags.split(",") if f.strip()}))

    # MatchInfo emits `aggregation_id` (`f"{template}|{display_path}"`),
    # which is also the canonical id the renderer-dict / frontend graph
    # uses for compute-graph nodes — single keyspace from SpecIRNode to
    # ELK input, no boundary translation. /fusion/fired's `anchor` /
    # `matched` arrays line up directly with the compute-graph node ids
    # the UI clicks on.
    SAMPLE_CAP = 8
    bound_ops = set(binding.fused_ops())
    fired_by_op: Dict[str, Dict[str, Any]] = {}
    phases: Dict[str, Dict[str, Any]] = {}
    for phase in (PREFILL, DECODE):         # a language-model step's phases
        result = analyze(
            resolved,
            parallel=ParallelConfig.from_dict({}),
            framework_binding=binding,
            batch_size=1,
            avg_input_tokens=1, avg_output_tokens=0,
            phase=phase,
            precision=precision,
            breakdown=True,
        )
        counts: Dict[str, int] = {}
        for match in result.fusion_matches:
            op = match["fused_op"]
            counts[op] = counts.get(op, 0) + 1
            bucket = fired_by_op.setdefault(
                op, {"match_count": 0, "matches": [], "phases": {}})
            if bucket["phases"] and phase not in bucket["phases"]:
                continue                # samples come from its first phase
            bucket["match_count"] += 1
            if len(bucket["matches"]) < SAMPLE_CAP:
                bucket["matches"].append({
                    "anchor": match["fused_aggregation_id"],
                    "matched": list(match["matched_aggregation_ids"]),
                })
        for op, count in counts.items():
            fired_by_op[op]["phases"][phase] = count
        # Bound, but its `when: phase` leaves this phase out: not a
        # mismatch with the model, a kernel the framework runs elsewhere.
        elsewhere = sorted(op for op in bound_ops
                           if (whens := tuple(when
                                              for impl in binding.implementations_for(op)
                                              for when in impl.whens))
                           and all(when.graph.phase
                                   and phase not in when.graph.phase
                                   for when in whens))
        phases[phase] = {
            "fired": sorted(counts),
            "not_in_phase": elsewhere,
            "unfired_bound": sorted(bound_ops - set(counts) - set(elsewhere)),
        }

    known_ops = set(FUSION_SCHEMES.keys())
    fired_ops = set(fired_by_op.keys())
    # "unfired" = bound by this framework but produced no match on this
    # model in any phase it runs in. Plans NOT bound by this framework are
    # reported separately so the UI can show "bound but unfired" vs "not
    # bound by this framework".
    unfired = sorted(bound_ops - fired_ops)
    unbound = sorted(known_ops - bound_ops)

    return {
        "model": model,
        "framework": framework,
        "audited_at": binding.audited_at.tag,
        "fired": fired_by_op,
        "unfired_bound": unfired,
        "unbound": unbound,
        "phases": phases,
        # The walks are single-rank: an entry whose `when:` asks a world's
        # degree (`world_local`) was answered at degree 1 only.
        "projection": "single_rank",
        "world_conditioned": sorted(
            op for op in bound_ops
            if any(when.graph.world_local is not None
                   for impl in binding.implementations_for(op)
                   for when in impl.whens)),
    }
