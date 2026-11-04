"""Prefill/decode disaggregation: the SOL first-order metrics.

Phase-C disaggregation splits serving into a prefill deployment and a decode
deployment. This module answers the two questions that split raises — what
the KV handoff costs, and how decode-heavy the pipeline is — from two
`estimate_capacity` results. It lives here, not in the API layer, because
both answers are speed-of-light semantics over estimate outputs.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from backend.inference.gpu_specs import gbps_to_bytes_per_sec, inter_node_bw_gbps


def disagg_handoff(
    prefill_result: Dict[str, Any],
    decode_result: Dict[str, Any],
    *,
    kv_transfer_bw_gbps: Optional[float],
    avg_output_tokens: Optional[int],
) -> Dict[str, Any]:
    """Phase-C disaggregation metrics (SOL first-order):

    * KV handoff — the prefill deployment produces the prompt's KV and ships it
      to the decode deployment before decode starts. Latency ≈ per-rank KV
      bytes / per-rank link bandwidth (the transfer parallelizes across ranks,
      so the decode receiver's per-rank shard bounds the wall-clock). Default
      link = the decode GPU's RDMA NIC (the same `inter_node_bw_gbps` the
      comm roofline reads; NVLink and PCIe are intra-node and never carry
      a cross-node handoff). A SKU without an RDMA column has no handoff
      latency to report. Override via `kv_transfer_bw_gbps`.
    * Rate matching — the per-request WALL-TIME ratio (decode time per request)
      / (prefill time per request). A first-order sizing signal for how
      decode-heavy the pipeline is; the true balanced decode:prefill replica
      ratio is this DIVIDED BY each deployment's request-level batch factor
      (decode serves many concurrent requests per step), so this is an upper
      bound on the replica ratio, not the ratio itself."""
    dc_kv = decode_result.get("kv_cache", {}) or {}
    per_req_kv = float(dc_kv.get("per_request_kv_bytes", 0.0) or 0.0)
    gpu = decode_result.get("gpu", {}) or {}
    bw_gbps = kv_transfer_bw_gbps or inter_node_bw_gbps(gpu.get("rdma_gbps"))
    bw_bytes = gbps_to_bytes_per_sec(bw_gbps) or 0.0
    handoff_s = (per_req_kv / bw_bytes) if bw_bytes > 0 else None

    # A phase may decline to publish a step time — `t_sol_s is None` under
    # pipeline parallelism, where the rows are one stage's and the model's step
    # is every stage in sequence plus a bubble nobody modelled. `or 0.0` would
    # turn that refusal into `prefill_time_per_request_s: 0.0` and a rate match
    # of infinity; the disaggregation answer is simply absent instead, with the
    # phase's own reason carried alongside it.
    pf_ph = (prefill_result.get("phases", {}) or {}).get("prefill", {}) or {}
    dc_ph = (decode_result.get("phases", {}) or {}).get("decode", {}) or {}
    pf_raw, dc_raw = pf_ph.get("t_sol_s"), dc_ph.get("t_sol_s")
    unavailable = " · ".join(
        f"{name}: {r}" for name, r in
        (("prefill", pf_ph.get("time_unpublished_reason")),
         ("decode", dc_ph.get("time_unpublished_reason"))) if r)
    pf_t = float(pf_raw) if pf_raw is not None else None
    dc_t = float(dc_raw) if dc_raw is not None else None
    decode_time_per_req = (int(avg_output_tokens or 0) * dc_t
                           if dc_t is not None else None)
    ratio = (decode_time_per_req / pf_t
             if decode_time_per_req is not None and pf_t else None)
    return {
        "kv_transfer": {
            "per_request_kv_bytes": per_req_kv,
            "link_bw_gbps": float(bw_gbps or 0.0),
            "handoff_latency_s": handoff_s,
        },
        "rate_match": {
            "prefill_time_per_request_s": pf_t,
            "decode_time_per_request_s": decode_time_per_req,
            # Per-request wall-time ratio — an UPPER bound on the balanced
            # decode:prefill replica ratio (batch parallelism lowers the true
            # ratio; see docstring). Named for what it is, not overclaimed.
            "decode_to_prefill_time_ratio": ratio,
            # Non-empty exactly when a phase above is null, and says which one
            # and why — so a missing rate match reads as withheld, not as zero.
            "unavailable_reason": unavailable,
        },
    }
