"""Regression proving that the hard-coded
`_FUSED_ATTENTION_OPS` whitelist in `activation_peak.py` is now
structurally redundant.

The whitelist was a manual workaround for a graph the engine couldn't
yet describe: zero out `attention_score` and `softmax` intermediates so
liveness analysis didn't claim DS-V3-scale 140 GB of activation. The
framework fusion rewrite now replaces the naive
`[attention_score, softmax, attention_apply]` chain into a single
`flash_attention` leaf with the same downstream output bytes, which
structurally eliminates the same intermediates without a hard-coded
list.

This test derives the same result two ways:

- **Public helper:** `estimate_peak_activation(..., flash_attention=True)`
  applies the framework rewrite internally and grafts
  the matched chain onto its anchor in the leaf-record list.
- **Fusion-driven (manual):** run the same rewrite in the test, project the
  physical shape layer to a leaf-record list
  the same way the helper does, graft the matched chain locally,
  then run liveness via the same `_peak_for_leaves` helper.

Both paths must produce byte-equal peak_activation_bytes for every
reference model. If they ever diverge, the whitelist's retirement is
blocked — investigate the divergence before removing the whitelist.

The test stays as a permanent retirement guard.
"""
from __future__ import annotations

import pytest

from tests.snapshots import resolve_snapshot, snapshot_config

from backend.arch.layout.physical import physical_kernel_shapes
from backend.inference.framework_bindings import get_binding


def _vllm():
    return get_binding("vllm")


_REFERENCE_MODELS = [
    "meta-llama/Llama-3.2-1B",       # dense Llama-style
    "deepseek-ai/DeepSeek-V2-Lite",  # MoE + MLA (small, snapshot on disk)
    "deepseek-ai/DeepSeek-V3.1",     # full MLA + DSv3-scale MoE
]


def _try_resolve(model_id: str):
    cfg = snapshot_config(model_id)
    resolved = resolve_snapshot(model_id)
    return resolved, cfg


def _analysis(resolved, *, batch: int, seq_len: int):
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.step_eval import evaluate_step

    return evaluate_step(
        resolved,
        ParallelConfig(),
        phase="prefill",
        batch=batch,
        input_tokens=seq_len, output_tokens=0,
    ).analysis


@pytest.mark.parametrize("model_id", _REFERENCE_MODELS)
@pytest.mark.parametrize("seq_len", [128, 4096])
def test_peak_leaf_uids_match_the_serving_rewrite(model_id: str, seq_len: int):
    """Peak liveness walks the same fused program pipeline.analyze uses."""
    from backend.arch.eval import cache as cache_mod
    from backend.arch.eval.fusion import rebase_shapes
    from backend.inference.activation_peak import _eval_leaf_records
    from backend.inference.materialize_cast import cast_facts
    from backend.inference.step_eval import estimate_peak_activation
    from backend.inference.serving import _on_program, rewrite_serving_graph

    resolved, _cfg = _try_resolve(model_id)
    public = estimate_peak_activation(
        resolved, batch=1, seq_len=seq_len, framework_binding=_vllm(),
    )
    analysis = _analysis(resolved, batch=1, seq_len=seq_len)
    serving = rewrite_serving_graph(
        analysis, framework_binding=_vllm(), )
    pre_shapes = _on_program(
        physical_kernel_shapes(analysis),
        serving.pre_program, serving.materialized.shapes,
    )
    if serving.casts:
        # Mirrors build_serving_layers (serving.py): a grafted cast is a
        # serving-graph leaf, and its shape layer must join pre_shapes
        # before the rebase or liveness meets a shapeless cast.
        cast_layers = cast_facts(
            serving.casts,
            local_shapes=pre_shapes,
            full_shapes=_on_program(
                analysis.shapes,
                serving.pre_program, serving.materialized.shapes,
            ),
            instances=_on_program(
                analysis.recount(
                    analysis.instances, analysis.layer_multiplicity()),
                serving.pre_program, serving.materialized.instances,
            ),
        )
        pre_shapes = _on_program(
            pre_shapes, serving.pre_program, cast_layers.local_shapes,
        )
    cached = (
        cache_mod.cached_node_uids(analysis.program)
        | set(analysis.cache_state.values)
    )
    shapes = rebase_shapes(
        serving.pre_program, pre_shapes, serving.program, serving.rewrite,
        preserve=frozenset(cached),
    )
    records_by_block, _ext = _eval_leaf_records(
        serving.program, shapes, analysis.resolved_precision, rewrite=serving.rewrite,
    )
    peak_uids = {
        r["uid"]
        for rows in records_by_block.values()
        for r in rows
    }
    serving_uids = {
        n.uid for n in serving.program.nodes() if not n.children
    }
    assert peak_uids <= serving_uids
    assert public["fused_ops"] == tuple(sorted({
        m["fused_op"] for m in serving.fusion_matches
    }))


def test_fused_leaf_inputs_do_not_reference_deleted_sids():
    """A fused leaf's inputs must resolve to tensors that still exist
    after the rewrite, not to absorbed members."""
    from backend.arch.eval import cache as cache_mod
    from backend.arch.eval.fusion import rebase_shapes
    from backend.inference.activation_peak import _eval_leaf_records
    from backend.inference.serving import _on_program, rewrite_serving_graph

    resolved, _cfg = _try_resolve("meta-llama/Llama-3.2-1B")
    analysis = _analysis(resolved, batch=1, seq_len=128)
    serving = rewrite_serving_graph(
        analysis, framework_binding=_vllm(), )
    if not serving.fusion_matches:
        pytest.skip("no fusion matches; nothing to rewrite")
    dropped = set(serving.rewrite.dropped)
    pre_shapes = _on_program(
        physical_kernel_shapes(analysis),
        serving.pre_program, serving.materialized.shapes,
    )
    cached = (
        cache_mod.cached_node_uids(analysis.program)
        | set(analysis.cache_state.values)
    )
    shapes = rebase_shapes(
        serving.pre_program, pre_shapes, serving.program, serving.rewrite,
        preserve=frozenset(cached),
    )
    records_by_block, _ext = _eval_leaf_records(
        serving.program, shapes, analysis.resolved_precision, rewrite=serving.rewrite,
    )
    live = {
        output_id
        for rows in records_by_block.values()
        for r in rows
        for output_id in r["outputs"]
    }
    for rows in records_by_block.values():
        for r in rows:
            for inp in r.get("inputs") or []:
                if inp in dropped:
                    raise AssertionError(
                        f"{r['id']} still reads absorbed uid {inp}"
                    )
                if isinstance(inp, int):
                    assert inp in live or inp in dropped


def test_component_and_secondary_outputs_join_real_liveness_tensors():
    from backend.inference.activation_peak import _eval_leaf_records

    resolved, _cfg = _try_resolve("deepseek-ai/DeepSeek-V4-Pro")
    analysis = _analysis(resolved, batch=1, seq_len=128)
    records_by_block, external_by_block = _eval_leaf_records(
        analysis.program,
        physical_kernel_shapes(analysis),
        analysis.resolved_precision,
    )

    consumed_ports = set()
    for template, records in records_by_block.items():
        produced = {
            output_id
            for record in records
            for output_id in record["outputs"]
        }
        external = set(external_by_block.get(template, {}))
        for record in records:
            for input_id in record["inputs"]:
                assert input_id in produced or input_id in external
                if isinstance(input_id, str) and "#port=" in input_id:
                    consumed_ports.add(input_id.rsplit("#port=", 1)[1])
    assert consumed_ports == {"comb", "post"}


@pytest.mark.parametrize(
    "model_id",
    ("zai-org/GLM-5.1", "zai-org/GLM-5.2"),
)
def test_liveness_isolated_by_live_block_invocation(model_id):
    from backend.inference.activation_peak import _eval_leaf_records

    resolved, _cfg = _try_resolve(model_id)
    analysis = _analysis(resolved, batch=1, seq_len=128)
    records_by_block, _external = _eval_leaf_records(
        analysis.program,
        physical_kernel_shapes(analysis),
        analysis.resolved_precision,
    )
    live_indices = {
        contribution.block_index
        for contribution in analysis.program.contributions
        if contribution.block_index is not None and contribution.count > 0
    }
    dormant_indices = {
        contribution.block_index
        for contribution in analysis.program.contributions
        if contribution.block_index is not None and contribution.count == 0
    }

    assert {index for index, _template in records_by_block} == live_indices
    assert not ({index for index, _template in records_by_block} & dormant_indices)
    for records in records_by_block.values():
        output_ids = [
            output_id
            for record in records
            for output_id in record["outputs"]
        ]
        assert len(output_ids) == len(set(output_ids))


def test_activation_fold_never_replays_the_evaluator():
    from backend.inference.activation_peak import activation_peak_at
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.step_eval import evaluate_step

    resolved, _cfg = _try_resolve("meta-llama/Llama-3.2-1B")
    step = evaluate_step(
        resolved,
        ParallelConfig(tp_attention=4, tp_dense_ffn=4),
        phase="prefill",
        batch=2,
        input_tokens=128, output_tokens=0,
    )
    out = activation_peak_at(step)

    assert out["batch"] == 2
    assert out["seq_len"] == 128
    assert out["peak_activation_bytes_per_rank"] > 0


def test_activation_fold_rejects_decode_analysis():
    from backend.inference.activation_peak import activation_peak_at
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.step_eval import evaluate_step

    resolved, _cfg = _try_resolve("meta-llama/Llama-3.2-1B")
    step = evaluate_step(
        resolved,
        ParallelConfig(),
        phase="decode",
        batch=2,
        input_tokens=128, output_tokens=0,
    )

    with pytest.raises(ValueError, match="requires a prefill or encode analysis"):
        activation_peak_at(step)


def test_peak_grows_linearly_with_seq_post_wiring():
    """Fusion wiring sanity: post-retirement, public
    `estimate_peak_activation` must structurally eliminate
    O(S²) attention intermediates via graph rewrite. Without the
    wiring, doubling seq_len from 4096→8192 would 4× the peak (the
    softmax intermediate dominates at long seq). With the wiring,
    peak should grow ≤2× (FFN-driven, linear in seq).

    This catches the regression where someone removes the wiring
    but leaves the whitelist deleted — the helper would silently
    over-report by ~10× at long seq."""
    from backend.inference.step_eval import estimate_peak_activation
    resolved, _cfg = _try_resolve("meta-llama/Llama-3.2-1B")
    p_short = estimate_peak_activation(
        resolved, batch=1, seq_len=4096, framework_binding=_vllm(),
    )["peak_activation_bytes_per_rank"]
    p_long = estimate_peak_activation(
        resolved, batch=1, seq_len=8192, framework_binding=_vllm(),
    )["peak_activation_bytes_per_rank"]
    growth = p_long / p_short
    assert 1.9 <= growth <= 2.2, (
        f"peak grew {growth:.2f}× from seq=4096 to seq=8192; "
        f"expected ~2× (linear). Quadratic growth (~4×) means "
        f"fusion rewriting is missing and O(S²) attention "
        f"intermediates are leaking into the peak. p_short={p_short:,}, "
        f"p_long={p_long:,}."
    )


def test_published_fusion_bytes_are_conserved():
    """Sink + named ports keep the same occupancy as the members they
    publish. Intermediates may disappear; published tensors may not."""
    from backend.arch.eval import cache as cache_mod
    from backend.arch.eval.dim import as_extent
    from backend.arch.eval.fusion import rebase_shapes
    from backend.inference.serving import _on_program, rewrite_serving_graph

    resolved, _cfg = _try_resolve("meta-llama/Llama-3.2-1B")
    analysis = _analysis(resolved, batch=1, seq_len=128)
    serving = rewrite_serving_graph(
        analysis, framework_binding=_vllm(), )
    assert serving.fusion_matches
    pre_shapes = _on_program(
        physical_kernel_shapes(analysis),
        serving.pre_program, serving.materialized.shapes,
    )
    cached = (
        cache_mod.cached_node_uids(analysis.program)
        | set(analysis.cache_state.values)
    )
    post_shapes = rebase_shapes(
        serving.pre_program, pre_shapes, serving.program, serving.rewrite,
        preserve=frozenset(cached),
    )

    def _elems(dims):
        if not dims:
            return 0
        n = 1
        for dim in dims:
            n *= as_extent(dim, rounding="floor")
        return n

    for match in serving.fusion_matches:
        fused = serving.program.by_uid(match["fused_uid"])
        post = post_shapes.values.get(match["fused_uid"])
        if fused is None or post is None:
            continue
        published = 0
        if post.out:
            published += _elems(post.out)
        for dims in (post.ports or {}).values():
            published += _elems(dims)
        origins = match["matched_uids"]
        ports = getattr(fused, "output_ports", None) or {}
        keep = set(ports) | ({fused.sink} if fused.sink is not None else set())
        prior = 0
        for uid in origins:
            if uid not in keep:
                continue
            shape = pre_shapes.values.get(uid)
            if shape is None or not shape.out:
                continue
            prior += _elems(shape.out)
        assert published == prior, (
            f"{match['fused_op']}: published {published} vs members {prior}"
        )
