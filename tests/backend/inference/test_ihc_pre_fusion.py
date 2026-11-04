"""Hy4's iHC pre-mix is two fused launches, and its intermediates are gone.

Both vLLM and sglang run the independent-Hyper-Connections pre of a sublayer
boundary as two Triton launches — the mix accumulation and then the gates with
the pre-weighted reduce — and neither writes the tensors between them. Before
`ihc_pre_mix` / `ihc_pre_gate_reduce` existed the step priced the whole pre
unfused, which materialised two [tokens, hc, d_model]-sized intermediates per
boundary that the kernels keep in registers: the flattened row the norm scales
and the gate-weighted copy the reduce sums.

Falsification targets:
  * the plans exist, both frameworks bind them, and the second declares that
    its launch spans the `hc_weighted_reduce` component the reduce lives in;
  * Hy4's step under each deployed binding has no standalone pre-mix leaf
    left — every `rmsnorm_unscaled` and every weighted-reduce `sum_reduce`
    is inside a launch — while the pre-fusion step has them all;
  * each fused row's bytes are the absorbed ops' bytes less exactly the
    round trip of the intermediates, and the largest of those intermediates
    is the [B, s, hc, d_model] copy, sized from the config rather than from
    the engine.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.analyzer import fetch_config
from backend.inference.framework_bindings import ServingDeployment, get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.gpu_specs import get_gpu, peak_hbm_bw
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from tests.snapshots import shared_loader

MODEL = "tencent/Hy4-preview"
CARD = "H100-SXM"
PROMPT = 4096
#: The two launches of one iHC pre, in the order the frameworks make them.
PRE_LAUNCHES = ("ihc_pre_mix", "ihc_pre_gate_reduce")
#: Leaves of the naive pre that no launch may leave standing: the
#: unweighted-RMS row the mix GEMM reads (absorbed by `ihc_pre_mix`) and the
#: copies reduce (absorbed by `ihc_pre_gate_reduce`).
PRE_MIX_LEAF_OPS = ("rmsnorm_unscaled", "sum_reduce")

_WALKS: dict = {}


def _walk(framework: str, phase: str):
    """Hy4's step on one card under one framework's deployed binding, walked
    once per (framework, phase) however many cases ask for it."""
    key = (framework, phase)
    if key in _WALKS:
        return _WALKS[key]
    resolved = shared_loader().match(fetch_config(MODEL), model_id=MODEL)
    gpu = get_gpu(CARD)
    binding = get_binding(framework).deployed(
        ServingDeployment.at(gpu_sku=CARD, activation_dtype="BF16"))
    _WALKS[key] = analyze(
        resolved, parallel=ParallelConfig(), framework_binding=binding,
        batch_size=1, avg_input_tokens=PROMPT, avg_output_tokens=256,
        phase=phase, gpu=gpu, peak_hbm_bw_bytes_per_sec=peak_hbm_bw(gpu),
        breakdown=False)
    return _WALKS[key]


# -- the plans and their bindings -----------------------------------------

def test_the_pre_is_two_plans_not_one():
    """A fused node is one launch, and the pre is two of them, so the two
    plans partition the naive pre rather than one covering it."""
    for name in PRE_LAUNCHES:
        assert name in FUSION_SCHEMES, f"{name}: no such plan"
    mix = FUSION_SCHEMES["ihc_pre_mix"].naive_pattern.expanded_nodes()
    gate = FUSION_SCHEMES["ihc_pre_gate_reduce"].naive_pattern.expanded_nodes()
    assert set(mix) == {"flat", "norm", "mixes"}
    # The mix GEMM is the handover: launch 1 writes it, launch 2 reads it as
    # an external. Were it a member of both, one of them would not be a launch.
    assert "external.mixes" in gate["pre_logits"].inputs
    assert "external.mixes" in gate["post_logits"].inputs
    assert "mixes" not in gate
    # The copies reduce is launch 2's, and its weighted copy is internal:
    # no io_mapping port names it, so nothing outside reads it.
    assert gate["weighted"].inputs == ["external.x", "gate_spread.out"]
    assert gate["reduced"].inputs == ["weighted.out"]
    ports = FUSION_SCHEMES["ihc_pre_gate_reduce"].naive_pattern.io_mapping
    assert ports["output"] == "reduced.out"
    assert ports["post_out"] == "post.out"
    assert "weighted" not in {t.partition(".")[0] for t in ports.values()}


def test_both_frameworks_bind_both_launches():
    """All four entries at once, in ONE case rather than one per framework:
    "both frameworks run this as two launches" is this change's whole claim,
    and `tests/conftest.py` marks every parameter case but the first
    `regression`, so a per-framework split would leave the default suite
    watching vLLM only."""
    for framework in ("vllm", "sglang"):
        binding = get_binding(framework)
        assert binding is not None, f"no {framework} binding"
        for name in PRE_LAUNCHES:
            impl = binding.get(name)
            assert impl is not None, f"{framework} must bind {name}"
            assert impl.source_anchor, f"{framework}/{name}: no source anchor"
        # Launch 2 computes the block's gate chain and the reduce that lives
        # in the `hc_weighted_reduce` sub-component, so its match spans a
        # component boundary; without the declaration the matcher confines it
        # to one scope and the plan never fires.
        assert binding.get("ihc_pre_gate_reduce").crosses == "component", (
            f"{framework}/ihc_pre_gate_reduce must declare the crossing")
        # Launch 1 does NOT cross one: all three of its members are the
        # block's own, and `crosses` states a fact about the launch rather
        # than narrowing where the plan may match.
        assert binding.get("ihc_pre_mix").crosses is None, (
            f"{framework}/ihc_pre_mix crosses no component boundary")


# -- the step the bindings produce ----------------------------------------

@pytest.mark.parametrize("framework,phase", [
    # Prefill under vLLM first: prefill is where the intermediates dominate
    # the step (decode moves one token against the whole weight set), so it
    # is the case that reddens if a plan stops firing.
    ("vllm", "prefill"), ("vllm", "decode"),
    ("sglang", "prefill"), ("sglang", "decode"),
])
def test_no_standalone_pre_mix_launch_survives(framework, phase):
    result = _walk(framework, phase)
    before = Counter(r["op"] for r in result.pre_per_op_timing or [])
    after = Counter(r["op"] for r in result.per_op_timing or [])
    for op in PRE_MIX_LEAF_OPS:
        assert before[op] > 0, (
            f"{framework}/{phase}: the unfused step should have {op} rows; "
            f"the test is not looking at the naive pre")
    assert after["rmsnorm_unscaled"] == 0, (
        f"{framework}/{phase}: a pre-mix norm row survives the rewrite")
    # The one reduce left is the network exit's: both frameworks run the
    # head as plain torch, so its copies reduce is a launch of its own. The
    # head's mix GEMM is the one leaf `ihc_pre_mix` claims there anyway.
    leftover = [r for r in result.per_op_timing or [] if r["op"] == "sum_reduce"]
    assert [r["aggregation_id"] for r in leftover] == [
        "hc_head.reduced.reduced__hc_weighted_reduce"], (
        f"{framework}/{phase}: the surviving copies reduces are "
        f"{[r['aggregation_id'] for r in leftover]}, not the exit head's alone")


@pytest.mark.parametrize("framework,phase", [
    ("vllm", "prefill"), ("vllm", "decode"),
    ("sglang", "prefill"), ("sglang", "decode"),
])
def test_every_boundary_fires_both_launches(framework, phase):
    """One pre per sublayer boundary, each as two launches — and `ihc_pre_mix`
    once more for the network-exit head, whose first three leaves have the
    same shape and which both frameworks run eagerly (the trade
    `ihc_pre_mix.yaml` states)."""
    fired = Counter(m["fused_op"] for m in _walk(framework, phase).fusion_matches)
    boundaries = fired["ihc_post"]
    assert boundaries > 0
    assert fired["ihc_pre_gate_reduce"] == boundaries
    assert fired["ihc_pre_mix"] == boundaries + 1


# -- what the fused rows cost ---------------------------------------------

@pytest.mark.parametrize("framework,phase", [
    ("vllm", "prefill"), ("vllm", "decode"),
    ("sglang", "prefill"), ("sglang", "decode"),
])
def test_a_fused_row_is_its_members_less_the_intermediates(framework, phase):
    """A launch moves what crosses its boundary: its members' bytes less the
    round trip of every tensor that stays inside it. The one other saving is
    the gate parameters — the naive form reads `hc_scale` and `hc_base` once
    per gate chain and the launch reads each once — so the slack above the
    round trip is bounded by what the members read of them, and does not
    grow with the token count."""
    result = _walk(framework, phase)
    fused_rows = {r["uid"]: r for r in result.per_op_timing or []}
    naive_rows = {r["uid"]: r for r in result.pre_per_op_timing or []}
    checked = 0
    for match in result.fusion_matches:
        if match["fused_op"] not in PRE_LAUNCHES:
            continue
        published = {aid for aid, _port in match["absorbed_output_ports"]}
        members = [naive_rows[uid] for uid in match["matched_uids"]]
        inside = [m for m in members if m["aggregation_id"] not in published]
        kept_in_registers = sum(float(m["act_out_bytes_per_rank"])
                                for m in inside)
        saved = (sum(float(m["access_bytes_per_rank"]) for m in members)
                 - float(fused_rows[match["fused_uid"]]["access_bytes_per_rank"]))
        parameters = sum(float(m["weight_bytes_per_rank"]) for m in members)
        assert kept_in_registers > 0
        slack = saved - 2.0 * kept_in_registers
        assert 0.0 <= slack <= 2.0 * parameters, (
            f"{framework}/{phase}/{match['fused_op']}: the launch saves "
            f"{saved} bytes; the round trip of what stays inside it is "
            f"{2.0 * kept_in_registers} and its members read {parameters} "
            f"bytes of gate parameters, so {slack} is unaccounted for")
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("framework,phase", [
    ("vllm", "prefill"), ("vllm", "decode"),
    ("sglang", "prefill"), ("sglang", "decode"),
])
def test_the_weighted_copy_is_the_intermediate_that_stops_being_written(
        framework, phase):
    """The largest tensor a pre launch stops writing is the gate-weighted
    copy of the residual stack, [B, s, hc, d_model] — sized here from the
    checkpoint config and the walk's token count, not from the engine's own
    byte columns. A row stands for every layer of its template, so its bytes
    are read back per occurrence."""
    result = _walk(framework, phase)
    config = fetch_config(MODEL)
    copies = int(config["hc_mult"])
    d_model = int(config["hidden_size"])
    tokens = PROMPT if phase == "prefill" else 1
    expected = tokens * copies * d_model * 2        # BF16 activations

    naive_rows = {r["uid"]: r for r in result.pre_per_op_timing or []}
    biggest = []
    for match in result.fusion_matches:
        if match["fused_op"] != "ihc_pre_gate_reduce":
            continue
        published = {aid for aid, _port in match["absorbed_output_ports"]}
        inside = [naive_rows[uid] for uid in match["matched_uids"]
                  if naive_rows[uid]["aggregation_id"] not in published]
        biggest.append(max(float(m["act_out_bytes_per_rank"])
                           / int(m["count_per_step"]) for m in inside))
    assert biggest, "no ihc_pre_gate_reduce match to check"
    assert set(biggest) == {float(expected)}, (
        f"{framework}/{phase}: the biggest intermediate each pre launch "
        f"stops writing is {sorted(set(biggest))}, not the "
        f"[{tokens}, {copies}, {d_model}] copy at {expected} bytes")
