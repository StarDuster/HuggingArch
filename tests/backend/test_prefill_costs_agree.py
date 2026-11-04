"""Every page that shows a prefill block's cost shows the estimate's.

A phase runs some components in another form (`transforms.phase_form`): a
sparse (DSA) MLA prefills absorbed. The architecture dashboard (`/analyze`),
the Inspector's block graph and the estimate each read a walk of the model;
when the first two read the authored graph and the estimate the phase's, one
page priced DeepSeek-V3.2's attention expanded and the next absorbed — two
consumers, two answers to one question (CLAUDE.md §1). Per layer template,
at one shape, the three agree; a dense MLA model is held to the same.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.analyzer import analyze_model
from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig
from backend.rendering.builders import build_block_graph_data
from tests.snapshots import resolve_snapshot, shared_loader, snapshot_config

SEQ = 2048
#: A sparse MLA (prefills absorbed) and a dense one (prefills expanded).
MODELS = ("deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3")


#: Deployments that hold the weights (one that does not runs no step and
#: publishes no table). The FLOPs compared are one replica's unsharded
#: ones (`Analysis.replica_cost`), so a tensor-parallel and a DP-attention
#: + expert-parallel deployment state the same per-layer figure — the
#: second used to quote the expert bank at every replica's tokens.
DEPLOYMENTS = {
    "tp8": dict(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8),
    "dp8ep8": dict(dp_attention=8, ep_moe=8),
}


def _estimate_per_layer(model_id: str, resolved, parallel: dict) -> dict:
    """Prefill FLOPs per layer of each backbone template, as the estimate's
    per-template rows publish them."""
    result = estimate_capacity(
        config=snapshot_config(model_id), resolved=resolved, num_gpus=8,
        parallel=ParallelConfig(**parallel), gpu_sku="H200",
        avg_input_tokens=SEQ, avg_output_tokens=1, prefill_batch_size=1,
        selection_mode="theoretical_only")
    return {row["template_name"]: row["flops_full"] / row["layer_count"]
            for row in result["phases"]["prefill"]["flops_by_module"]
            if row.get("role") == "block_template"}


@pytest.mark.parametrize("deployment", sorted(DEPLOYMENTS))
@pytest.mark.parametrize("model_id", MODELS)
def test_dashboard_inspector_and_estimate_price_prefill_alike(model_id, deployment):
    resolved = resolve_snapshot(model_id, shared_loader())
    estimate = _estimate_per_layer(model_id, resolved, DEPLOYMENTS[deployment])
    assert estimate

    dashboard = {group["template"]: group["total_flops"]
                 for group in analyze_model(model_id, batch_size=1,
                                            seq_len=SEQ)["layer_groups"]
                 if group.get("template") in estimate}
    blocks, _ = build_block_graph_data(resolved, batch_size=1, seq_len=SEQ)
    inspector = {name: sum(int(node.get("flops") or 0)
                           for node in entry["nodes"])
                 for name, entry in blocks.items() if name in estimate}

    assert set(dashboard) == set(inspector) == set(estimate)
    for template, flops in estimate.items():
        assert dashboard[template] == pytest.approx(flops, rel=1e-12), template
        assert inspector[template] == pytest.approx(flops, rel=1e-12), template
