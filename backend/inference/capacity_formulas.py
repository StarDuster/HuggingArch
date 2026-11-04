"""Popover formulas for the capacity response.

These strings are display, not the budget. They live next to `capacity`
because inference must not import rendering (§2.7); the API is the
composition root that may hand them to a page.
"""
from __future__ import annotations

from typing import Any, Dict, List


def capacity_formulas(
    *,
    preparation: Any,
    ceilings: Any,
    weight_breakdown: Dict[str, Any],
) -> Dict[str, Any]:
    """The `formulas` object on an estimate — HBM / usable / deployment,
    KV budget, max tokens, max batch, per-request KV tree, weight-per-rank.
    Numbers come in already computed; this only writes the strings that
    explain them.
    """
    p = preparation
    per_token_per_rank = ceilings.per_token_per_rank
    request_tokens = ceilings.request_tokens
    effective_storage_tokens = ceilings.effective_storage_tokens
    per_request_kv_bytes = ceilings.per_request_kv_bytes
    mem_total_bytes = p.mem_total_bytes
    weight_bytes_per_rank = ceilings.weight_bytes_per_rank
    # Weights kept in host memory: in the tree, not in the sum.
    host_bytes = (p.host_offload_bytes[ceilings.pp_stage or 0]
                   if p.host_offload else 0.0)
    kv_budget_bytes = ceilings.kv_budget_bytes
    weight_deficit_bytes = float(ceilings.weight_deficit_bytes or 0)
    kv_static_per_rank = ceilings.kv_static_per_rank
    max_tokens = ceilings.max_tokens
    max_tokens_total = ceilings.max_tokens_total
    max_batch_total = ceilings.max_batch_total
    parallel = p.parallel
    max_batch = ceilings.max_batch
    max_prefill_batch = ceilings.max_prefill_batch
    overhead_bytes = ceilings.overhead_bytes
    usable_for_tokens = ceilings.usable_for_tokens
    peak_act_per_b1 = ceilings.peak_act_per_b1
    wb = weight_breakdown

    # Per-request KV: the reader's own formula at the request length — the
    # same object that produced `per_request_kv_bytes` (under pp, the bound
    # stage's), so the popover cannot drift from the number it explains.
    # Per rank, so the label says per GPU.
    _per_request_kv_formula = ceilings.kv_reader.request_formula(request_tokens)
    _per_request_kv_formula["label"] = _per_request_kv_formula["label"].replace(
        "Per-request KV", "Per-request KV / GPU", 1)

    kv_budget_substituted = (
        f"{mem_total_bytes:.0f} × {p.gpu_memory_fraction} − "
        f"{weight_bytes_per_rank:.0f}"
    )
    if not ceilings.fits:
        kv_budget_substituted += " → 0"
    usable_bytes = p.usable_bytes
    # The breakdown's own split of the deployment total (`weight_breakdown`
    # folds it once off the leaves); the page shows the same two numbers.
    data_full = float(wb["deployment_data_bytes"])
    scale_full = float(wb["deployment_scale_bytes"])
    deploy_full = float(wb.get("deployment_total_bytes") or 0)
    deploy_parts_sym = ["data_bytes_full"]
    deploy_parts_sub = [f"{data_full:.0f}"]
    deploy_subterms: List[dict] = [{
        "symbol": "data",
        "label": "Data",
        "symbolic": "Σ data_bytes_full",
        "substituted": f"{data_full:.0f}",
        "result_bytes": data_full,
    }]
    if scale_full > 0:
        deploy_parts_sym.append("scale_bytes_full")
        deploy_parts_sub.append(f"{scale_full:.0f}")
        deploy_subterms.append({
            "symbol": "scale",
            "label": "Scale",
            "symbolic": "Σ scale_bytes_full",
            "substituted": f"{scale_full:.0f}",
            "result_bytes": scale_full,
        })
    # Scale metadata no weight leaf claims is its own declared term of the
    # deployment total (`weight_breakdown` warns about it), not a residual.
    orphan_scale = wb.get("unattributed_scale_bytes")
    if orphan_scale is not None and orphan_scale > 0:
        orphan_scale = float(orphan_scale)
        deploy_parts_sym.append("unattributed_scale_bytes")
        deploy_parts_sub.append(f"{orphan_scale:.0f}")
        deploy_subterms.append({
            "symbol": "unattributed_scale",
            "label": "Scale with no parent leaf",
            "symbolic": "Σ unattributed scale bytes",
            "substituted": f"{orphan_scale:.0f}",
            "result_bytes": orphan_scale,
        })
    formulas: Dict[str, Any] = {
        "mem_total": {
            "label": "HBM capacity / GPU",
            "symbolic": "gpu.hbm_bytes",
            "substituted": f"{mem_total_bytes:.0f}",
            "result_bytes": mem_total_bytes,
        },
        "usable": {
            "label": "Usable HBM / GPU",
            "symbolic": "mem_total × gpu_memory_fraction",
            "substituted": f"{mem_total_bytes:.0f} × {p.gpu_memory_fraction}",
            "result_bytes": usable_bytes,
        },
        "deployment": {
            "label": "Deployment total",
            "symbolic": " + ".join(deploy_parts_sym),
            "substituted": " + ".join(deploy_parts_sub),
            "result_bytes": deploy_full,
            "subterms": deploy_subterms,
        },
        "kv_budget": {
            "label": "KV budget / GPU",
            "symbolic": "mem_total × fraction − weight_bytes_per_rank",
            "substituted": kv_budget_substituted,
            "result_bytes": kv_budget_bytes,
        },
        "max_tokens": {
            "label": "Max tokens / GPU (= per KV-scope)",
            "symbolic": ("(kv_budget − static_scale − constant_floor) "
                         "/ per_token_bytes_per_rank"),
            "substituted": (
                f"({kv_budget_bytes:.0f} − {kv_static_per_rank:.0f} − "
                f"{ceilings.kv_constant_state_per_rank:.0f}) / "
                f"{per_token_per_rank:.4g}"
            ),
            "result": max_tokens,
            **({} if max_tokens is not None else {
                "description": (
                    "The cache does not grow with length (every cached "
                    "position sits in a window or a fixed state), so no "
                    "token count exhausts it: unbounded. The budget bounds "
                    "the batch instead."),
            }),
        },
        "max_tokens_total": {
            "label": "Max unique tokens / cluster",
            "symbolic": "max_tokens × dp_attention × dp_replicas",
            "substituted": (
                f"{'unbounded' if max_tokens is None else max_tokens} × "
                f"{parallel.dp_attention} × {parallel.dp_replicas}"
            ),
            "result": max_tokens_total,
            "description": (
                "Only DP-attention and DP-replicas carry independent batches; "
                "TP/PP/CP/EP share the same token set across ranks."
            ),
        },
        "max_batch": {
            "label": "Max decode batch / DP rank",
            "symbolic": "(kv_budget − static_scale) / per_request_kv_bytes",
            "substituted": (
                f"({kv_budget_bytes:.0f} − {kv_static_per_rank:.0f}) / "
                f"{per_request_kv_bytes:.0f}"
                + (
                    f"  [SWA cap: min({request_tokens}, s_window) = {effective_storage_tokens}]"
                    if effective_storage_tokens < request_tokens else ""
                )
            ),
            "result": max_batch,
        },
        "max_batch_total": {
            "label": "Unique requests / cluster",
            "symbolic": "max_batch × dp_attention × dp_replicas",
            "substituted": (
                f"{max_batch} × {parallel.dp_attention} × {parallel.dp_replicas}"
            ),
            "result": max_batch_total,
            "description": (
                "TP/PP/EP share the same request set and CP splits one "
                "sequence across ranks — none of them run more requests. "
                "Only DP-attention and DP-replicas carry independent batches."
            ),
        },
        "max_prefill_batch": {
            "label": "Max prefill batch / DP rank",
            "symbolic": (
                "((1 − gpu_memory_fraction) × mem_total − overhead_bytes"
                " + kv_pool_bytes) / peak_activation_bytes_per_rank"
                "(B=1, seq=min(input, chunk))"
            ),
            "substituted": (
                f"((1 − {p.gpu_memory_fraction}) × {mem_total_bytes:.0f} − "
                f"{overhead_bytes:.0f} + {max(0.0, usable_for_tokens):.0f})"
                f" / {peak_act_per_b1:.0f}"
            ),
            "result": max_prefill_batch,
            "description": (
                "Prefill is activation-bound, not KV-bound, and does not coexist "
                "with decode: the KV pool is empty during the prefill forward, so "
                "its bytes are free for activations. Budget = the residual outside "
                "gpu_memory_fraction (activation + overhead) PLUS the whole KV "
                "pool decode would otherwise hold — i.e. the card minus weights "
                "and overhead. The per-stream footprint is the peak at B=1 and "
                "seq=min(input, prefill_chunk_size), so the ceiling plateaus "
                "once the prompt is longer than one chunk."
            ),
        },
        "per_request_kv": _per_request_kv_formula,
        "weight_per_rank": {
            "label": "Weight bytes / GPU (hierarchical)",
            "symbolic": ("Σ_node (bytes_full / divisor) − weights in host memory" if host_bytes else
                         "Σ_node (bytes_full / divisor)"),
            "substituted": (
                f"{len(wb.get('groups', []))} block group(s) + "
                f"{len(wb.get('model_level', []))} model-level node(s)"
                + (f" − {host_bytes:.0f}" if host_bytes else "")
            ),
            "result_bytes": weight_bytes_per_rank,
        },
    }
    if weight_deficit_bytes > 0:
        formulas["weight_deficit"] = {
            "label": "Weight deficit / GPU",
            "symbolic": "weight_bytes_per_rank − mem_total × fraction",
            "substituted": (
                f"{weight_bytes_per_rank:.0f} − {mem_total_bytes:.0f} × "
                f"{p.gpu_memory_fraction}"
            ),
            "result_bytes": weight_deficit_bytes,
        }
    return formulas
