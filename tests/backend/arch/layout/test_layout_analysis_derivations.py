"""Analysis-derived physical layers and pipeline count projection."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from tests.snapshots import shared_loader
from backend.arch.eval.dim import value_of
from backend.arch.eval import cache as cache_mod
from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.layer import Layer
from backend.arch.layout.persistent import (
    artifacts_for_layers,
    physical_artifacts,
    resident_weights,
)
from backend.arch.layout.physical import physical_kernel_shapes
from backend.arch.layout.physical_cache import (
    cache_geometry,
    physical_cache_view,
)
from backend.arch.layout.physical_cost import (
    physical_access,
    physical_flops,
    physical_traffic,
    physical_traffic_geometry,
)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import _flops_rows
from backend.arch.layout.stages import StagePlacement
from backend.inference.step_eval import evaluate_step


SNAPSHOTS = Path(__file__).resolve().parents[4] / "backend" / "arch" / "snapshots"
TP8 = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)


def _step(
    snapshot: str = "meta-llama__Llama-3.2-3B.json",
    *,
    pc: ParallelConfig = TP8,
    phase: str = "prefill",
):
    payload = json.loads((SNAPSHOTS / snapshot).read_text())
    resolved = shared_loader().match(
        payload["config"], model_id=Path(snapshot).stem.replace("__", "/", 1))
    assert resolved is not None
    return evaluate_step(
        resolved,
        pc,
        phase=phase,
        batch=1,
        input_tokens=4096, output_tokens=0,
        # Prefill: 128 new tokens over 4096 positions, the rest cached.
        s_prefix_cached=4096 - 128 if phase == "prefill" else 0,
    )


@pytest.mark.parametrize(
    "snapshot",
    (
        "meta-llama__Llama-3.2-3B.json",
        "openai__gpt-oss-20b.json",
        "deepseek-ai__DeepSeek-V3.json",
        "deepseek-ai__DeepSeek-V4-Pro.json",
        "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.json",
    ),
)
@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_physical_layers_derive_from_one_analysis_without_replay(
    snapshot, phase
):
    step = _step(snapshot, phase=phase)
    analysis = step.analysis

    shapes = physical_kernel_shapes(analysis)
    flops = physical_flops(analysis, shapes)
    artifacts = physical_artifacts(analysis)
    cache = physical_cache_view(analysis)

    assert shapes.program is analysis.program
    assert flops.program is analysis.program
    assert flops.values
    assert artifacts
    assert cache.source_analysis is analysis
    assert cache.entries
    total_flops = _public_flops(analysis)
    assert not getattr(value_of(total_flops), "free_symbols", None)
    assert all(
        not getattr(value_of(flops), "free_symbols", None)
        for flops in flops.values.values()
    )
    assert resident_weights(artifacts).resident > 0


def test_ep_divides_the_expert_axis_not_a_copy_count():
    """Expert parallelism shards an expert bank's weights along their
    leading `n_expert_copies` axis, like any other shape split; the walk's
    instance count is the layers, on every rank."""
    from backend.arch.layout.persistent import STORED, physical_artifacts

    ep = 2
    step = _step(
        "XiaomiMiMo__MiMo-V2.5.json",
        pc=ParallelConfig(
            tp_attention=4,
            tp_dense_ffn=4,
            tp_moe_expert=4,
            ep_moe=ep,
            dp_attention=2,
        ),
    )
    analysis = step.analysis
    layers = analysis.layer_multiplicity()
    experts = {
        node.uid for node in analysis.program.nodes()
        if node.role == "ffn_moe" and node.name == "gate_proj"
    }
    assert experts
    rows = [row for row in physical_artifacts(analysis)
            if row.node_uid in experts and row.surface == STORED]
    assert rows
    for row in rows:
        assert row.global_shape[0] == row.runtime_shape[0] * ep
        assert row.runtime_instances == layers.values[row.node_uid]
        assert (analysis.instances.values[row.node_uid]
                == layers.values[row.node_uid])


@pytest.mark.parametrize(
    ("snapshot", "pc"),
    (
        (
            "stepfun-ai__Step-3.7-Flash.json",
            ParallelConfig(
                tp_attention=8,
                tp_dense_ffn=8,
                tp_moe_expert=8,
            ),
        ),
        ("openai__gpt-oss-20b.json", ParallelConfig(ep_moe=4)),
    ),
)
def test_a_packed_bank_is_one_node_per_layer(snapshot, pc):
    """Every weight of a packed bank runs once per layer — its experts are
    an axis of its tensors, not copies of it."""
    from backend.arch.eval.storage import leads_with_expert_axis

    analysis = _step(snapshot, pc=pc).analysis
    layer_counts = analysis.layer_multiplicity()
    banks = [
        uid for uid, tensors in analysis.storage.values.items()
        if any(leads_with_expert_axis(t) for t in tensors)
    ]
    assert banks
    for uid in banks:
        assert analysis.instances.values[uid] == layer_counts.values[uid]


def _ownership(analysis):
    out = {}
    for contribution in analysis.program.contributions:
        pending = [contribution.node]
        while pending:
            node = pending.pop()
            out[node.uid] = contribution
            pending.extend(node.children or ())
    return out


def _public_flops(analysis, counts=None) -> float:
    counts = analysis.layer_multiplicity() if counts is None else counts
    included = analysis.program.included_stack_names()
    flops = physical_flops(analysis)
    total = 0.0
    for contribution in analysis.program.contributions:
        if contribution.stack not in included:
            continue
        value = flops.values.get(contribution.node.uid)
        if value is None:
            continue
        total += float(value_of(value)) * counts.values[contribution.node.uid]
    return total


def test_pipeline_stage_projects_storage_cache_and_compute_together():
    pc = ParallelConfig(
        tp_attention=8,
        tp_dense_ffn=8,
        tp_moe_expert=8,
        pp=16,
    )
    step = _step("openai__gpt-oss-20b.json", pc=pc)
    analysis = step.analysis
    whole_artifacts = physical_artifacts(analysis)
    whole_cache = physical_cache_view(analysis)
    placement = StagePlacement.of(analysis, pc.pp)
    scope = analysis.program.included_stack_names(include_aux=True)
    views = tuple(placement.view(stage) for stage in range(pc.pp))
    for view in views:
        multiplicity = view.rows(scope)
        phase_rows = _flops_rows(step, multiplicity)
        assert sum(
            row["flops_per_rank"]
            for row in phase_rows if row["depth"] == 0
        ) == _public_flops(analysis, multiplicity)
        assert all(row.get("flops_per_rank") is not None for row in phase_rows)

    whole_cache_bytes = whole_cache.storage_bytes_at(4096)
    assert sum(
        whole_cache.on(view).storage_bytes_at(4096)
        for view in views
    ) == whole_cache_bytes

    # Every artifact is held where its node runs: summed over stages, a
    # block's layers are the model's, and a model-level node is on one stage
    # (gpt-oss reads no table through `tied_weight`).
    by_key = {row.key: row for row in whole_artifacts}
    assert by_key
    staged = [{item.key: item for item in artifacts_for_layers(
                  analysis, whole_artifacts, view.weights(scope))}
              for view in views]
    for key, row in by_key.items():
        assert sum(stage[key].runtime_instances
                   for stage in staged) == row.runtime_instances
        assert sum(stage[key].resident_bytes_per_rank
                   for stage in staged) == row.resident_bytes_per_rank
    assert any(item.runtime_instances == 0
               for stage in staged for item in stage.values())


@pytest.mark.parametrize(
    "snapshot",
    ("zai-org__GLM-5.1.json", "zai-org__GLM-5.2.json"),
)
def test_phase_fold_keeps_repeated_template_invocations_separate(snapshot):
    step = _step(snapshot)
    analysis = step.analysis
    rows = _flops_rows(step)

    assert sum(
        row["flops_per_rank"] for row in rows if row["depth"] == 0
    ) == _public_flops(analysis)
    assert all(row.get("flops_per_rank") is not None for row in rows)


@pytest.mark.parametrize(
    ("snapshot", "tail_sid"),
    (
        ("XiaomiMiMo__MiMo-V2.5.json", "visual.patch_embed.proj"),
        ("google__gemma-4-12B-it.json", "vision_embedder.patch_ln1"),
        (
            "meituan-longcat__LongCat-Flash-Thinking-2601.json",
            "mtp.embed_tokens",
        ),
    ),
)
def test_phase_fold_excludes_inactive_auxiliary_rows(snapshot, tail_sid):
    step = _step(snapshot)
    analysis = step.analysis
    rows = _flops_rows(step)

    assert all(row["aggregation_id"] != tail_sid for row in rows)
    assert _public_flops(analysis) == sum(
        row["flops_per_rank"] for row in rows if row["depth"] == 0
    )


def test_traffic_geometry_requires_complete_layers_from_one_program():
    analysis = _step().analysis
    geometry = physical_traffic_geometry(analysis)

    with pytest.raises(ValueError, match="every shape owner"):
        traffic_mod.derive(
            analysis,
            geometry=replace(
                geometry,
                shapes=Layer(analysis.program, {}),
            ),
        )
    shape_uid, shape = next(
        (uid, row)
        for uid, row in geometry.shapes.values.items()
        if row.out and len(row.out) >= 2
    )
    shapes = dict(geometry.shapes.values)
    shapes[shape_uid] = replace(
        shape,
        out=(shape.out[1], shape.out[0], *shape.out[2:]),
    )
    with pytest.raises(ValueError, match="shape axis identity"):
        traffic_mod.derive(
            analysis,
            geometry=replace(
                geometry,
                shapes=Layer(analysis.program, shapes),
            ),
        )
    with pytest.raises(ValueError, match="every cache owner"):
        traffic_mod.derive(
            analysis,
            geometry=replace(
                geometry,
                cache=Layer(analysis.program, {}),
            ),
        )

    storage_uid = next(iter(geometry.storage.values))
    storage = dict(geometry.storage.values)
    storage[storage_uid] = ()
    with pytest.raises(ValueError, match="storage cardinality"):
        traffic_mod.derive(
            analysis,
            geometry=replace(
                geometry,
                storage=Layer(analysis.program, storage),
            ),
        )

    storage_uid, tensor_index, tensor = next(
        (uid, index, row)
        for uid, rows in geometry.storage.values.items()
        for index, row in enumerate(rows)
        if len(row.shape) >= 2
    )
    storage = dict(geometry.storage.values)
    tensors = list(storage[storage_uid])
    tensors[tensor_index] = replace(
        tensor,
        shape=(tensor.shape[1], tensor.shape[0], *tensor.shape[2:]),
    )
    storage[storage_uid] = tuple(tensors)
    with pytest.raises(ValueError, match="storage tensor identity"):
        traffic_mod.derive(
            analysis,
            geometry=replace(
                geometry,
                storage=Layer(analysis.program, storage),
            ),
        )

    cache_uid = next(iter(geometry.cache.values))
    caches = dict(geometry.cache.values)
    caches[cache_uid] = replace(caches[cache_uid], tensors=())
    with pytest.raises(ValueError, match="cache tensor identity"):
        traffic_mod.derive(
            analysis,
            geometry=replace(
                geometry,
                cache=Layer(analysis.program, caches),
            ),
        )

    foreign = _step("Qwen__Qwen3-30B-A3B.json").analysis
    with pytest.raises(ValueError, match="cache belongs to a different Program"):
        traffic_mod.derive(
            analysis,
            geometry=replace(
                geometry,
                cache=cache_mod.derive(
                    foreign, geometry=cache_geometry(foreign)
                ),
            ),
        )


@pytest.mark.parametrize(
    "snapshot",
    (
        "meta-llama__Llama-3.2-3B.json",
        "openai__gpt-oss-20b.json",
        "Qwen__Qwen3-30B-A3B.json",
    ),
)
def test_phase_flops_fold_consumes_the_analysis(snapshot):
    step = _step(snapshot)
    rows = _flops_rows(step)
    analysis = step.analysis

    assert sum(
        row["flops_per_rank"] for row in rows if row["depth"] == 0
    ) == _public_flops(analysis)
    assert all(row.get("flops_per_rank") is not None for row in rows)


@pytest.mark.parametrize(
    "snapshot",
    (
        "meta-llama__Llama-3.2-3B.json",
        "openai__gpt-oss-20b.json",
        "Qwen__Qwen3-30B-A3B.json",
    ),
)
def test_physical_compute_layers_share_the_analysis_program(snapshot):
    step = _step(snapshot)
    analysis = step.analysis
    shapes = physical_kernel_shapes(analysis)
    flops = physical_flops(analysis, shapes)

    assert shapes.program is analysis.program
    assert flops.program is analysis.program
    assert set(flops.values) == set(analysis.cost.values)
    assert set(shapes.values) == set(analysis.shapes.values)
    assert all(not getattr(value_of(value), "free_symbols", None)
               for value in flops.values.values())


@pytest.mark.parametrize(
    "snapshot",
    (
        "meta-llama__Llama-3.2-3B.json",
        "openai__gpt-oss-20b.json",
        "Qwen__Qwen3-30B-A3B.json",
    ),
)
@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_local_traffic_uses_the_complete_layout_geometry(snapshot, phase):
    step = _step(snapshot, phase=phase)
    analysis = step.analysis

    actual = physical_traffic(analysis)

    assert actual.values
    for uid, row in actual.values.items():
        fields = ("weight_reads", "metadata_reads", "act_reads", "act_writes", "kv_reads")
        assert all(value_of(getattr(row, name)) >= 0 for name in fields), (
            analysis.program.by_uid(uid).display_path,
            row,
        )


@pytest.mark.parametrize(
    "snapshot",
    (
        "meta-llama__Llama-3.2-3B.json",
        "openai__gpt-oss-20b.json",
        "Qwen__Qwen3-30B-A3B.json",
    ),
)
def test_physical_access_is_a_program_layer(snapshot):
    step = _step(snapshot)
    analysis = step.analysis

    actual = physical_access(analysis, scale_dtype_bytes=4.0)

    assert actual.program is analysis.program
    assert actual.values
    assert all(row.per_rank >= 0 for row in actual.values.values())


@pytest.mark.parametrize(
    "snapshot",
    (
        "meta-llama__Llama-3.2-3B.json",
        "openai__gpt-oss-20b.json",
        "Qwen__Qwen3-30B-A3B.json",
    ),
)
def test_pipeline_rewrites_complete_physical_access_layers(snapshot):
    from backend.inference.pipeline import analyze
    from backend.inference.framework_bindings import get_binding

    step = _step(snapshot)
    binding = get_binding("vllm")
    result = analyze(
        step.spec,
        parallel=step.analysis.rule.parallel,
        framework_binding=binding,
        step_eval=step,
        breakdown=False,
    )

    assert result.access.program is result.program
    assert result.rewrite.origins
    assert all(row.per_rank >= 0 for row in result.access.values.values())


def test_pipeline_projection_rejects_invalid_counts():
    step = _step(pc=ParallelConfig(
        tp_attention=8,
        tp_dense_ffn=8,
        tp_moe_expert=8,
        pp=2,
    ))
    analysis = step.analysis

    root = analysis.program.contributions[0].node.uid
    with pytest.raises(ValueError, match="no accounting root"):
        analysis.program.multiplicity({10**9: 1})
    with pytest.raises(ValueError, match="keeps -1"):
        analysis.program.multiplicity({root: -1})
