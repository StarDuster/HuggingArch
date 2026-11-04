"""Physical persistent artifacts derived from one Analysis."""
from __future__ import annotations

import json
from math import prod
from pathlib import Path

import pytest

from tests.snapshots import shared_loader
from backend.arch.layout import persistent
from backend.arch.layout.persistent import (
    physical_artifacts,
    resident_weights,
)
from backend.arch.layout.propagation import propagate_step
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step


SNAPSHOTS = Path(__file__).resolve().parents[4] / "backend" / "arch" / "snapshots"
TP8 = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)


def _step(snapshot: str, pc: ParallelConfig = TP8, phase: str = "prefill"):
    payload = json.loads((SNAPSHOTS / snapshot).read_text())
    resolved = shared_loader().match(
        payload["config"], model_id=Path(snapshot).stem.replace("__", "/", 1))
    assert resolved is not None
    return evaluate_step(
        resolved,
        pc,
        phase=phase,
        batch=1,
        input_tokens=128, output_tokens=0,
        # Prefill: 8 new tokens over 128 positions, the rest cached.
        s_prefix_cached=120 if phase == "prefill" else 0,
    )


def _storage(step):
    return physical_artifacts(propagate_step(step))


def test_local_artifact_keys_are_complete_and_unique():
    step = _step("XiaomiMiMo__MiMo-V2.5-Pro.json")
    analysis = propagate_step(step)
    artifacts = physical_artifacts(analysis)
    expected = set()
    for uid in analysis.storage.values:
        reps = analysis.formats.values[uid]
        for tensor_index, reps_of_tensor in enumerate(reps):
            # One artifact set per layer segment of the tensor — one for
            # every tensor of this model, which declares no layer-scoped
            # override.
            for segment, rep in enumerate(reps_of_tensor):
                expected.add(
                    (uid, tensor_index, segment, persistent.STORED, None))
                expected.update(
                    (uid, tensor_index, segment,
                     persistent.SAVED_METADATA, index)
                    for index in range(len(rep.saved.metadata))
                )
                expected.update(
                    (uid, tensor_index, segment,
                     persistent.RESIDENT_METADATA, index)
                    for index in range(len(rep.resident.metadata))
                )

    keys = [row.key for row in artifacts]
    assert len(keys) == len(set(keys))
    assert set(keys) == expected


def test_implicit_metadata_with_expert_tp_uses_the_production_projection():
    pc = ParallelConfig(
        tp_attention=2,
        tp_dense_ffn=2,
        dp_attention=4,
        tp_moe_expert=2,
        ep_moe=4,
    )
    step = _step("MiniMaxAI__MiniMax-M2.7.json", pc)
    analysis = propagate_step(step)
    artifacts = physical_artifacts(analysis)
    metadata = [
        row for row in artifacts
        if row.surface == persistent.RESIDENT_METADATA
        and "block_sparse_moe.experts" in row.artifact_name
    ]
    assert len(metadata) == 3
    # The rank's 64 experts lead every scale grid, as they lead the weights;
    # one tensor per layer holds them.
    assert {row.runtime_shape for row in metadata} == {
        (64, 6, 24), (64, 24, 6)}
    assert {row.runtime_instances for row in metadata} == {62}


def test_packed_stored_tensor_reports_bytes_without_inventing_a_slot_axis():
    artifacts = _storage(_step("openai__gpt-oss-20b.json"))
    packed = [
        row for row in artifacts
        if row.surface == persistent.STORED
        and row.resident_packing_factor not in (None, 1)
    ]
    assert packed
    for row in packed:
        assert row.resident_physical_slot_shape is None
        assert row.resident_bytes_per_instance == (
            prod(row.runtime_shape) // row.resident_packing_factor
        )


@pytest.mark.parametrize(
    "snapshot",
    (
        "meta-llama__Llama-3.2-3B.json",
        "openai__gpt-oss-20b.json",
        "moonshotai__Kimi-K3.json",
        "XiaomiMiMo__MiMo-V2.5-Pro.json",
        "deepseek-ai__DeepSeek-V4-Pro.json",
    ),
)
@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_tier_models_build_grounded_physical_artifacts(snapshot, phase):
    step = _step(snapshot, phase=phase)
    artifacts = _storage(step)

    assert artifacts
    resident = resident_weights(artifacts)
    assert resident.resident > 0
    assert resident.resident == sum(
        row.resident_bytes_per_rank
        for row in artifacts
        if row.surface in {
            persistent.STORED,
            persistent.RESIDENT_METADATA,
        }
    )
    assert all(row.runtime_instances >= 0 for row in artifacts)
    assert all(
        row.metadata_geometry.stored_transform in {"matrix", "flatten_leading"}
        for row in artifacts
        if row.metadata_geometry is not None
        and row.metadata_geometry.axes is not None
        and any(
            symbol in " ".join(row.metadata_geometry.axes)
            for symbol in ("in", "out")
        )
    )
