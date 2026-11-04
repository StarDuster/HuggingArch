"""/analyze and /compare — the two big endpoints.

`/analyze` runs the analyzer + spec match + builders to produce the
full payload the frontend's Inspector tab consumes. The endpoint body
is thin; `build_response` does the dict assembly.

`/compare` runs `/analyze` on two models in parallel, then computes
side-by-side diffs (metrics, component roles, dim values,
per-modality dim tables).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

from fastapi import HTTPException
from fastapi.responses import Response

from backend import analyzer
from backend.analyzer import RESOLVED_KEY, NoSpecMatch, analyze_model
from backend.arch.loader import SpecLoadError
from backend.arch.loader import SpecLoader
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.projection import display_formulas as display_formulas
from backend.arch.spec_migrate import symbol_renames
from backend.rendering import (
    build_architecture_view,
    build_block_graph_data,
    build_model_config,
    model_name,
)
from backend.views.composition import CALL_KINDS
from backend.views.graph_view import stamp_l1
from backend.spec_worker.outcome import RunErrorKind

from .._app import app
from ..offload import (
    call_key,
    ensure_spec_generation,
    offload_response,
    render_json,
    run_off_loop,
    spec_generation,
    within_deadline,
)
from .._helpers import fetch_forward_source_payload
from .._models import AnalyzeRequest, CompareRequest, MODALITY_REGISTRY


# ---- Formatters ----------------------------------------------------------

def _format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def _format_shape(shape) -> str:
    if not shape:
        return "N/A"
    return "[" + ", ".join(str(s) if s is not None else "?" for s in shape) + "]"


# ---- Sub-builders for /analyze response ---------------------------------

def _build_formulas(resolved_spec, batch_size: int, seq_len: int) -> Dict[str, Any]:
    """Run the engine to produce per-role flop/param formulas. Falls
    back to a placeholder when the spec match failed (no spec → no
    formulas)."""
    if resolved_spec is None:
        return {
            "unsupported": "model has no matching spec; formulas not available"
        }
    return display_formulas(
        resolved_spec, batch_size=batch_size, seq_len=seq_len,
    )


def _build_dashboard(summary: Dict[str, Any], req: AnalyzeRequest) -> Dict[str, Any]:
    """Flatten summary into the dashboard sub-dict the frontend reads."""
    total_params = summary.get("total_params", 0)
    # Both embedding figures come straight from the backend summary — the engine's
    # per-node accounting when a spec matched, a config estimate otherwise. The
    # route never re-derives them (no vocab*hidden here), so "does this model even
    # have an embedding layer" is answered once, in the backend.
    embedding_params = summary.get("embedding_params", 0)
    non_embedding_params = summary.get("non_embedding_params", 0)
    attention_info = summary.get("attention_info", {})
    ffn_info = summary.get("ffn_info", {})
    return {
        "total_params": total_params,
        "total_active_params": summary.get("total_active_params", total_params),
        "total_params_formatted": _format_params(total_params),
        "total_active_params_formatted": _format_params(
            summary.get("total_active_params", total_params)
        ),
        # Per prompt token in prefill; None where the summary did not walk
        # a spec. `total_active_params` is per decoded token.
        "total_decode_active_params": summary.get("total_decode_active_params"),
        "total_prefill_active_params": summary.get("total_prefill_active_params"),
        "prefill_full_from": summary.get("prefill_full_from"),
        "main_prefill_active_params": summary.get("main_prefill_active_params"),
        "lookup_table_params": summary.get("lookup_table_params"),
        "embedding_params": embedding_params,
        "non_embedding_params": non_embedding_params,
        "flops_per_token": summary.get("total_flops", 0),
        "flops_per_token_g": f"{summary.get('total_flops_gflops', 0):.2f}G",
        # Per-role flops breakdown — the Inspector compare view's
        # "FLOPs (per token)" card (Total / Attention / FFN) reads these
        # (web/src/pages/Inspector.tsx).
        "total_flops": summary.get("total_flops", 0),
        "attention_flops": summary.get("attention_flops", 0),
        "ffn_flops": summary.get("ffn_flops", 0),
        "attention_type": attention_info.get("attention_type") or None,
        # The cache families behind it (`cache.family`), for what keys on
        # the kind of attention rather than its display name.
        "attention_families": attention_info.get("attention_families") or [],
        "num_layers": summary.get("num_layers", 0),
        "hidden_size": summary.get("hidden_size", 0),
        # Note: HF-derived logical-params fields (hf_params,
        # hf_quantization_metadata_params, hf_checkpoint_params_total)
        # were removed — they're misleading on mixed-precision
        # checkpoints because `packed_unpack_factor` resolves one
        # quant_method per checkpoint and silently misroutes the rest.
        # Storage-bytes view below is the canonical descriptor.
        "quant_metadata_params": summary.get("quant_metadata_params", 0),
        "quant_metadata_breakdown": summary.get("quant_metadata_breakdown", []),
        "quant_metadata_scale_dtype": summary.get("quant_metadata_scale_dtype"),
        "quant_metadata_bytes": summary.get("quant_metadata_bytes", 0),
        "quant_metadata_formats": summary.get("quant_metadata_formats", []),
        # Deprecated compatibility field: no single per-scale width exists;
        # read `quant_metadata_formats`. Null when the dtypes are mixed.
        "quant_metadata_bytes_per_scale": summary.get(
            "quant_metadata_bytes_per_scale"),
        # The main model alone (no MTP, no encoder): every headline's
        # Total / Active. `total_params` above is every stack the
        # checkpoint holds. A summary that walked no spec has one figure.
        "main_total_params": summary.get("main_total_params", total_params),
        "main_active_params": summary.get(
            "main_active_params",
            summary.get("total_active_params", total_params)),
        "main_flops": summary.get("main_flops", 0),
        "aux_params": summary.get("aux_params", {}),
        "aux_active_params": summary.get("aux_active_params", {}),
        "aux_flops": summary.get("aux_flops", {}),
        "hf_model_storage_bytes": summary.get("hf_model_storage_bytes", 0),
        "hf_quantization_metadata_bytes": summary.get("hf_quantization_metadata_bytes", 0),
        "hf_checkpoint_storage_bytes": summary.get("hf_checkpoint_storage_bytes", 0),
        "spec_metadata_bytes": summary.get("spec_metadata_bytes", 0),
        "spec_checkpoint_bytes": summary.get("spec_checkpoint_bytes", 0),
        "metadata_storage_diff": summary.get("metadata_storage_diff", 0),
        "metadata_storage_diff_pct": summary.get("metadata_storage_diff_pct", 0.0),
        "all_tensor_diff": summary.get("all_tensor_diff", 0),
        "all_tensor_diff_pct": summary.get("all_tensor_diff_pct", 0.0),
        "checkpoint_anchored": summary.get("checkpoint_anchored", False),
        "hf_index_total_size": summary.get("hf_index_total_size"),
        "measured_tensor_diff": summary.get("measured_tensor_diff"),
        "measured_tensor_diff_pct": summary.get("measured_tensor_diff_pct"),
        # Everything the spec stores (parameters + quantisation scales +
        # buffers; the sum of the drawn nodes' weight_bytes), and the
        # parameters alone — the half `spec_storage_bytes` compares.
        "weight_bytes": summary.get("weight_bytes", 0),
        "param_bytes": summary.get("param_bytes", 0),
        "checkpoint_copies": summary.get("checkpoint_copies", []),
        # Canonical headline diff — bytes-vs-bytes (mixed-precision safe).
        # See analyzer.py for why this replaced `params_diff`.
        "spec_storage_bytes": summary.get("spec_storage_bytes", 0),
        "storage_diff": summary.get("storage_diff", 0),
        "storage_diff_pct": summary.get("storage_diff_pct", 0.0),
        "spec_unresolved_note": summary.get("spec_unresolved_note"),
        # The dtype holding most weight bytes, and its width; null when the
        # analysis names none (no weights), never a guessed BF16.
        "dtype_bytes": summary.get("dtype_bytes"),
        "weight_dtype": summary.get("weight_dtype"),
        "precision": req.precision_profile().to_json(),
        "moe_enabled": summary["moe_enabled"],
        "moe_layers": summary.get("num_moe_layers", 0),
        # Compare-panel fields (the N-way metric table + the compare cards read
        # these off the dashboard). Without them the "Dense layers"/"Vocab"/"MTP
        # params" rows were dropped and the "MTP / aux" card read 0.
        "num_dense_ffn_layers": summary.get("num_dense_ffn_layers", 0),
        "vocab_size": summary.get("vocab_size", 0),
        # The aux stacks that are not encoders (MTP heads); an encoder's
        # parameters are their own figure, never under the MTP label.
        "aux_total_params": sum(
            n for name, n in (summary.get("aux_params") or {}).items()
            if name not in (summary.get("encoder_params") or {})),
        "encoder_params": summary.get("encoder_params", {}),
        # Each encoder stack's modality (`aux_kind`), and the FLOPs one
        # media item costs it, in that item's own axes.
        "encoder_stacks": summary.get("encoder_stacks", {}),
        "encoder_flops_per_item": summary.get("encoder_flops_per_item", {}),
        "encoder_total_params": sum(
            (summary.get("encoder_params") or {}).values()),
        # Stored bytes over the language model (`lm`) and over the whole
        # checkpoint, encoder towers included (`with_encoders`): a page shows
        # one of the two, never sums them. Parameter headlines are the main
        # model's (`main_total_params` / `main_active_params`) in every view.
        "weight_totals": summary.get("weight_totals"),
        "moe_active_experts": ffn_info.get("top_k", 0),
        "moe_total_experts": ffn_info.get("num_experts", 0),
    }


def _build_layer_groups(layer_groups) -> list:
    """Project the analyzer's per-block-group output into the frontend
    shape (only known fields, with shape strings formatted). A block's
    template is the analyzer's, named by the invocation it came from."""
    out = []
    for group in layer_groups:
        idx = group.get("block_index", -1)
        template = group.get("template")
        out.append({
            "name": group.get("name", ""),
            "layer_type": group.get("layer_type", ""),
            "block_index": idx,
            # `stack_name` lets the frontend split main-backbone blocks
            # from auxiliary modules (MTP / vision / audio towers) into
            # separate sections. `bucket_id` is the engine's structural-
            # identity key — frontend uses it to bucket layers that share
            # the exact same template AND dim_overrides into one
            # representative card.
            "stack_name": group.get("stack_name"),
            "aux_kind": group.get("aux_kind"),
            "bucket_id": group.get("bucket_id"),
            # A derived group's declaration and history length
            # (`BlockMeta.derived_group`): the page buckets its layers under
            # the declaration, one child per length.
            **({"derived_group": group["derived_group"]}
               if "derived_group" in group else {}),
            "template": template,
            "count": group.get("count", 1),
            "total_params": group.get("total_params", 0),
            "total_flops": group.get("total_flops", 0),
            "attention_params": group.get("attention_params", 0),
            "attention_flops": group.get("attention_flops", 0),
            "ffn_params": group.get("ffn_params", 0),
            "ffn_flops": group.get("ffn_flops", 0),
            "input_shape": _format_shape(group.get("input_shape")),
            "output_shape": _format_shape(group.get("output_shape")),
            "flops_formula": group.get("total_flops", 0),
            "memory_bytes": group.get("total_memory_bytes", 0),
            "sub_layers": [
                {
                    "name": layer.get("name", ""),
                    "layer_type": layer.get("layer_type", ""),
                    "layer_class": layer.get("layer_class", ""),
                    "param_count": layer.get("param_count", 0),
                    "flops": layer.get("flops", 0),
                    "in_features": layer.get("in_features", 0),
                    "out_features": layer.get("out_features", 0),
                }
                for layer in group.get("sub_layers", [])
            ],
        })
    return out


def build_response(req: AnalyzeRequest, result: Dict[str, Any],
                    elapsed_ms: float) -> Dict[str, Any]:
    """Assemble the full /analyze response from analyzer output. Every view
    — compute graph, formulas, labels — is of the spec `analyze_model`
    analysed (`RESOLVED_KEY`): a signed-in user's private spec when it ran
    on their overlay, never a second match."""
    summary = result.get("summary", {})
    layer_groups = result.get("layer_groups", [])
    resolved_spec = result.get(RESOLVED_KEY)
    payload = {key: value for key, value in result.items()
               if key != RESOLVED_KEY}
    formulas = _build_formulas(resolved_spec, req.batch_size, req.seq_len)

    if resolved_spec is not None:
        block_graph, unmatched = build_block_graph_data(
            resolved_spec, req.batch_size, req.seq_len,
            precision=req.precision_profile(),
            workload_symbolic=req.symbolic_shapes,
        )
        for graph in (block_graph or {}).values():
            stamp_l1(graph.get("nodes") or [])
    else:
        block_graph, unmatched = None, []

    return {
        # `success` is this endpoint's OUTCOME envelope, not its error
        # channel: False means the analyzer answered "I cannot analyse that"
        # — an unknown model, a spec that does not validate — and the page
        # renders `message`. A bug in the analyzer is not an outcome; it
        # leaves as a 500 so it shows up as one instead of being rendered to
        # the user as `'NoneType' object has no attribute ...`.
        "success": True,
        "model_id": req.model_id,
        "analysis_time_ms": elapsed_ms,
        "model_info": {
            "family": model_name(summary, resolved_spec),
            "hidden_size": summary.get("hidden_size", 0),
            "num_layers": summary.get("num_layers", 0),
            "vocab_size": summary.get("vocab_size", 0),
        },
        "dashboard": _build_dashboard(summary, req),
        "layer_groups": _build_layer_groups(layer_groups),
        "representative_template": result.get("representative_template"),
        # The model's blocks with their scenario-free figures (params by
        # role, stored bytes by dtype, KV per token) — `views.composition`.
        "composition": result.get("composition"),
        "block_graph": block_graph,
        "unmatched_nodes": unmatched,
        "unmatched_params_total": sum(u["params"] for u in unmatched),
        "formulas": formulas,
        "architecture_view": build_architecture_view(
            summary, layer_groups,
            resolved_spec, _format_params, result.get("config", {}),
        ),
        "model_config": build_model_config(summary, resolved_spec),
        "raw_data": payload,
        "model_str": result.get("model_str", ""),
        "config_json": result.get("raw_config_json", {}),
        "config_processed": result.get("config", {}),
        "forward_source": fetch_forward_source_payload(req.model_id, resolved_spec),
    }


# ---- /analyze ------------------------------------------------------------

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    # The analysis and the payload around it are seconds of pure-Python
    # CPU (config fetch, the walk, the builders): a compute worker runs
    # them, so the event loop stays free for /health and every other
    # request — a loop held by one analysis is what Cloudflare answers
    # with a 524.
    return await offload_response(analyze_payload, req)


def analyze_payload(req: AnalyzeRequest) -> dict:
    """The `/analyze` answer for one model."""
    start_time = time.time()
    try:
        result = analyze_model(
            model_id=req.model_id,
            batch_size=req.batch_size,
            seq_len=req.seq_len,
            dtype=req.dtype,
            # NEVER True on this anonymous endpoint: the no-spec skeleton path
            # executes an arbitrary repo's modeling code (trust_remote_code),
            # so True here is a public RCE. A model no spec matches is refused
            # (the analyzer points at /generate-spec); only the print(model)
            # skeleton is skipped for the matched ones.
            trust_remote_code=False,
        )
        if not result.get("success"):
            # Config-fetch failures are a client-visible refusal, not an HTML
            # 500. `analyze_model` has already classified the fetch through
            # explain_config_fetch_failure on this path.
            raise HTTPException(
                status_code=404,
                detail=result.get("error", "Unknown error"),
            )
        elapsed_ms = (time.time() - start_time) * 1000
        return build_response(req, result, elapsed_ms)
    except SpecLoadError as e:
        # A spec that exists but fails validation is a SPEC error — say so
        # explicitly instead of letting it read as a generic failure.
        return {"success": False, "error_kind": RunErrorKind.SPEC_INVALID.value, "message": str(e)}
    except (ValueError, KeyError) as e:
        # The request named something the analyzer cannot answer for — an
        # unknown model, a dim that does not resolve. An outcome, not a fault.
        # `no_spec`: no spec matches, the one case spec generation answers.
        return {"success": False, "message": str(e),
                **({"no_spec": True} if isinstance(e, NoSpecMatch) else {})}


# ---- /compare ------------------------------------------------------------

def _all_same(values: list) -> bool:
    """True when every entry (including None for a missing side) is equal —
    i.e. the field does not distinguish the compared models."""
    return len(set(values)) <= 1


def _metric_pairs(results: list) -> list:
    """Build the N-way metric table from the models' /analyze payloads.
    Each row carries a `values` list (aligned to the model order) + a `same`
    flag so the frontend can dim non-distinguishing rows. Every row is a
    quantity of the model — fixed by its spec and checkpoint, not by a batch
    or length: the dashboard's counts and bytes, and the KV figures of its
    composition (`views.composition`)."""
    dashboards = [r.get("dashboard") or {} for r in results]
    totals = [(r.get("composition") or {}).get("totals") or {} for r in results]
    # The dtypes each model's KV figures are priced at (a declared FP8 entry
    # beside native BF16 ones is both), not only the native default.
    kv_dtypes = [" · ".join(e["dtype"] for e in t.get("kv_dtypes") or ()) or None
                 for t in totals]
    # Weight bytes over the language model, and over the whole checkpoint
    # (encoder towers included): a row carries both, the page shows one.
    scoped = {scope: [(d.get("weight_totals") or {}).get(scope) or {}
                      for d in dashboards]
              for scope in ("lm", "with_encoders")}
    metric_defs = [
        # The main model's (no MTP, no encoder), as the single view and the
        # gallery state them; MTP and encoders are their own rows.
        ("Total params", "main_total_params", "params", dashboards),
        ("Active params", "main_active_params", "params", dashboards),
        ("MTP params", "aux_total_params", "params", dashboards),
        ("Encoder params", "encoder_total_params", "params", dashboards),
        ("Weight bytes", "weight_bytes", "bytes", scoped),
        ("Weight bytes by dtype", "storage_by_dtype", "bytes_by_dtype", scoped),
        ("KV cache / token", "kv_per_token_bytes", "bytes", totals),
        ("KV state / sequence", "kv_constant_bytes", "bytes", totals),
        ("Layers", "num_layers", "number", dashboards),
        ("MoE layers", "moe_layers", "number", dashboards),
        ("Dense layers", "num_dense_ffn_layers", "number", dashboards),
        ("Hidden size", "hidden_size", "number", dashboards),
        ("Vocab", "vocab_size", "number", dashboards),
        ("Attention type", "attention_type", "text", dashboards),
    ]
    out = []
    for label, key, kind, source in metric_defs:
        if isinstance(source, dict):
            values = [d.get(key) for d in source["lm"]]
            with_encoders = [d.get(key) for d in source["with_encoders"]]
        else:
            values = [d.get(key) for d in source]
            with_encoders = None
        if all(v is None for v in values):
            continue  # drop rows no model has
        row = {"label": label, "key": key, "kind": kind,
               "values": values, "same": _all_same(
                   [repr(v) for v in values])}
        if with_encoders is not None:
            row["values_with_encoders"] = with_encoders
            row["same_with_encoders"] = _all_same(
                [repr(v) for v in with_encoders])
        out.append(row)
        if key == "kv_per_token_bytes":
            # The width the KV rows are priced at, for each model.
            out.append({"label": "KV dtype", "key": "kv_dtypes",
                        "kind": "text", "values": kv_dtypes,
                        "same": _all_same(kv_dtypes)})
    return out


def _flat_calls(signature: dict, path: str = "") -> List[tuple]:
    """A call signature as `(path, component)` pairs, its sub-calls by path."""
    out = []
    for name, sub in (signature.get("calls") or {}).items():
        where = f"{path}.{name}" if path else name
        out.append((where, sub["component"]))
        out.extend(_flat_calls(sub, where))
    return out


def _component_call_diff(results: list) -> list:
    """Per role family, the calls each model's layers make, from each
    analysis's composition (`views.composition.calls_of`): a component
    with its slots as bound, and how many layers make that call. A
    component name alone cannot tell two bindings of one component apart;
    the signature does. `same` compares the calls whole. A cell names each
    call by its component, and where calls of one component differ — across
    the models or across a model's layers — the sub-calls they differ in."""
    per_model = []
    for r in results:
        calls: Dict[str, Dict[str, list]] = {}
        for row in (r.get("composition") or {}).get("rows") or []:
            prefix = f"{row['stack']}: " if row.get("aux_kind") else ""
            for call in row.get("calls") or []:
                key = prefix + json.dumps(call["signature"], sort_keys=True)
                held = calls.setdefault(call["role"], {}).setdefault(
                    key, [prefix, call["signature"], 0])
                held[2] += int(call["layers"])
        per_model.append(calls)
    out = []
    for role in CALL_KINDS:
        found = [m.get(role, {}) for m in per_model]
        if not any(found):
            continue
        # Per component, the sub-calls every call of it makes alike.
        shared: Dict[str, set] = {}
        for calls in found:
            for _prefix, signature, _n in calls.values():
                pairs = set(_flat_calls(signature))
                name = signature["component"]
                shared[name] = (pairs if name not in shared
                                else shared[name] & pairs)

        def show(prefix: str, signature: dict, layers: int) -> str:
            alike = shared[signature["component"]]
            slots = [f"{path}={component}"
                     for path, component in _flat_calls(signature)
                     if (path, component) not in alike]
            bound = f" [{', '.join(slots)}]" if slots else ""
            return f"{prefix}{signature['component']}{bound} ×{layers}"

        values = [[show(*held) for held in calls.values()] for calls in found]
        keys = [tuple(sorted((k, held[2]) for k, held in calls.items()))
                for calls in found]
        out.append({"slot": role, "values": values, "same": _all_same(keys)})
    return out


_DIM_RENAMES = symbol_renames()


def _binding_ints(resolved) -> dict:
    """This run's Binding.this as plain ints (Program.base_binding)."""
    if not resolved:
        return {}
    out = {}
    for name, raw in Program.of(resolved).base_binding.flat().items():
        value = value_of(raw)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    return out


def _dim_get(dims: dict, key: str):
    """Look up `key`, mapping a pre-rename letter onto the current name."""
    if key in dims:
        return dims.get(key)
    mapped = _DIM_RENAMES.get(key)
    if mapped is not None:
        return dims.get(mapped)
    return None


def _modality_diffs(dims_list: list, dashboards: list) -> list:
    """Modality-aware N-way diff: each model's encoders (`encoder_stacks`,
    the stacks the spec's structure makes encoders) grouped by the
    modality their stack declares (`aux_kind`), surfaced beside the
    language model. `present`/`params`/`values` are aligned to the model
    order; `params` sums a model's encoders of that modality. A modality
    with display metadata in `MODALITY_REGISTRY` is shown."""
    kinds = [d.get("encoder_stacks") or {} for d in dashboards]
    params = [d.get("encoder_params") or {} for d in dashboards]
    seen = sorted({kind for stacks in kinds for kind in stacks.values()
                   if kind in MODALITY_REGISTRY})
    out = []
    for modality in seen:
        meta = MODALITY_REGISTRY[modality]
        rows = []
        for key, label in meta["dims"]:
            values = [_dim_get(d, key) for d in dims_list]
            if all(v is None for v in values):
                continue
            rows.append({"key": key, "label": label,
                         "values": values, "same": _all_same(values)})
        present = [modality in stacks.values() for stacks in kinds]
        out.append({
            "name": modality,
            "label": meta["label"],
            "icon": meta.get("icon", "module"),
            "present": present,
            "params": [
                sum(n for stack, n in own.items()
                    if stacks.get(stack) == modality) if here else None
                for stacks, own, here in zip(kinds, params, present,
                                             strict=True)],
            "rows": rows,
        })
    return out


@app.post("/compare")
async def compare(req: CompareRequest):
    """Run /analyze on 2..5 models and return each payload plus an N-way diff."""
    return await within_deadline(call_key(_compare, (req,), {}),
                                 lambda: _compare(req))


async def _compare(req: CompareRequest) -> Any:
    """The analyses side by side, each in its own compute worker; then the
    diff and its JSON response, in one more."""
    def sub(model_id: str) -> AnalyzeRequest:
        return AnalyzeRequest(
            model_id=model_id,
            batch_size=req.batch_size, seq_len=req.seq_len,
            symbolic_shapes=req.symbolic_shapes,

        )

    generation = spec_generation()
    try:
        results = list(await asyncio.gather(
            *[run_off_loop(analyze_payload, sub(m)) for m in req.models]))
    except (ValueError, KeyError) as e:
        return {"success": False, "message": f"Compare failed: {e}"}
    # The analyses must have read one version of every spec.
    ensure_spec_generation(generation)
    return await run_off_loop(compare_response, req, results)


def compare_response(req: CompareRequest, results: list) -> Response:
    return render_json(compare_payload(req, results))


def compare_payload(req: CompareRequest, results: list) -> dict:
    """The `/compare` answer from the models' `/analyze` payloads."""

    if not all(r.get("success") for r in results):
        return {
            "success": False,
            "message": "One or more analyses failed",
            "errors": [None if r.get("success") else r.get("message") for r in results],
            "results": [r if r.get("success") else None for r in results],
        }

    loader = SpecLoader()
    resolved_list = []
    for m in req.models:
        cfg = analyzer.fetch_config(m)
        # Pass model_id so the custom:/overlay-only hook fires (same as the
        # per-model /analyze path). Without it, a model resolvable only through
        # the model_id route (a custom model, a spec-miss overlay) came back None
        # → a blank dims/roles/spec column in the compare grid.
        resolved_list.append(loader.match(cfg, model_id=m) if cfg else None)

    dashboards = [r.get("dashboard") or {} for r in results]
    metrics = _metric_pairs(results)
    components = _component_call_diff(results)

    dims_list = [_binding_ints(r) for r in resolved_list]
    all_keys = sorted(set().union(*[set(d) for d in dims_list])) if dims_list else []
    dim_rows = []
    for k in all_keys:
        values = [d.get(k) for d in dims_list]
        if _all_same(values):
            continue  # keep only dims that distinguish the models
        dim_rows.append({"key": k, "values": values, "same": False})

    modality_diffs = _modality_diffs(dims_list, dashboards)

    diff = {
        "metrics": metrics,
        "components": components,
        "dims": dim_rows,
        "modalities": modality_diffs,
    }

    from backend.views.graph_view import diff_representative_blocks
    graph_diff = diff_representative_blocks(results, resolved_list, req.models)

    return {
        "success": True,
        "results": results,
        "specs": [r.spec.name if r else None for r in resolved_list],
        "diff": diff,
        "graph_diff": graph_diff,
    }
