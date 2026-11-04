"""The Scaling panel's Context x Batch payload, point by point, as pinned at
the commit the golden names (`tests/golden/scaling_payload.json`,
`pinned_at`).

Every point, after the payload's projection
(`sweep_runtime.served_sweep_point`), hashes to the pinned one. A change
that moves a point regenerates the golden in the same commit and says, in
its message, which points moved and why.

This replaced a pin against b91ea40ae (`scaling_since_b91`) that no
commit of this repository had passed since its first (520ad8501): the
corrections it named were all in the module table, which a sweep point no
longer carries (`batch_sweep.phase_at`), and the rest of its drift could no
longer be told apart.

Every case is a deployment that holds its weights — an expert mesh
(DeepSeek-V3 DP16/EP16), a dense model (Llama-2-7b), a sequence cut
(Kimi-K2.5 CP16/EP16) and a hybrid (Qwen3-Coder-Next TP4), all on H100s —
because one that does not publishes no step (`capacity.without_step`) and
its points would pin only that: the first re-pin had three such cases, 9
distinct digests in 36 points each. A context whose cache does not fit
still pins its no-step point beside the others."""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from backend.api.sweep_runtime import served_sweep_point
from backend.inference.capacity import estimate_capacity_sweep
from backend.inference.parallel_config import ParallelConfig
from backend.inference.sweep_grid import ROUTINE_BATCH_GRID, ROUTINE_SEQ_LENS
from tests.snapshots import snapshot_config

pytestmark = pytest.mark.regression

_GOLDEN = json.loads((pathlib.Path(__file__).resolve().parents[2]
                      / "golden" / "scaling_payload.json").read_text())


def _digest(point) -> str:
    return hashlib.sha256(json.dumps(point, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("case", _GOLDEN["cases"],
                         ids=[c["model"] for c in _GOLDEN["cases"]])
def test_the_payload_is_the_pinned_one(case):
    parallel = ParallelConfig(**case["parallel"])
    swept = estimate_capacity_sweep(
        seq_lens=list(ROUTINE_SEQ_LENS),
        batch_grid=[dict(bp) for bp in ROUTINE_BATCH_GRID],
        config=snapshot_config(case["model"]), model_id=case["model"],
        gpu_sku="H100-SXM", parallel=parallel, num_gpus=parallel.cluster_size,
        gpu_memory_fraction=0.9, avg_output_tokens=256,
        selection_mode="theoretical_only", measurement_scope=None)
    # The payload as served, round-tripped through JSON as a response is.
    payload = json.loads(json.dumps(
        {key: [served_sweep_point(point) for point in points]
         for key, points in swept.items()}, default=str))
    for key, digests in case["digests"].items():
        assert [_digest(point) for point in payload[key]] == digests, key
