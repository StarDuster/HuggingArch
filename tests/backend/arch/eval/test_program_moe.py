"""`Analysis.moe`: the routing numbers a MoE model declares, read from the
graph's expert banks and the spec's registered dims as the walk bound them
— and None for a dense model."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.analysis import MoeFacts
from backend.arch.eval.transforms import Deployment
from tests.snapshots import shared_program, shared_walk


def _moe(model_id: str):
    return shared_walk(Deployment(shared_program(model_id))).moe


def test_deepseek_v3_declares_its_routing():
    assert _moe("deepseek-ai/DeepSeek-V3") == MoeFacts(
        n_expert_copies=256, active_copies=8, n_experts=256,
        n_experts_active=8, n_route_targets=256, d_ffn_expert=2048,
        n_shared_experts=1)


def test_longcat_routes_to_targets_no_node_stands_for():
    # 768 routing destinations, 256 of them zero-compute: the router picks
    # 12, the bank runs 8 — the two active counts are two facts.
    facts = _moe("meituan-longcat/LongCat-Flash-Thinking-2601")
    assert (facts.n_route_targets, facts.n_experts, facts.n_expert_copies) == (768, 512, 512)
    assert (facts.n_experts_active, facts.active_copies) == (12, 8)
    assert facts.n_shared_experts == 0


def test_a_packed_bank_of_leaves_is_a_bank():
    # gpt-oss's experts are custom fused leaves carrying `repeat:`, not a
    # component; the width is the value they bind to `d_ffn_expert`.
    assert _moe("openai/gpt-oss-20b") == MoeFacts(
        n_expert_copies=32, active_copies=4, n_experts=32, n_experts_active=4,
        n_route_targets=32, d_ffn_expert=2880, n_shared_experts=0)


def test_kimi_k3_shared_count_is_the_declared_dim_not_the_fused_node():
    # Two shared experts fused into one node of twice the width: the count
    # is the spec's `n_shared_experts`, the node is the implementation.
    facts = _moe("moonshotai/Kimi-K3")
    assert (facts.n_shared_experts, facts.n_expert_copies, facts.d_ffn_expert) == (2, 896, 3072)


def test_a_dense_model_has_no_facts():
    assert _moe("meta-llama/Llama-3.2-1B") is None


def test_every_moe_snapshot_reads_its_bank_and_every_dense_one_is_none():
    # Provenance, not coincidence: a MoE model's copies come from a routed
    # bank whose weights lead with the expert axis (so the counts are
    # positive and the bank was found), a dense model has no facts. Whether
    # the axis equals the dispatch denominator, or the bank's width equals the
    # registered symbol, is a per-model fact pinned on the models above, not a
    # corpus law.
    from tests.snapshots import SNAPSHOT_DIR
    moe = dense = 0
    for path in sorted(SNAPSHOT_DIR.glob("*.json")):
        program = shared_program(path.stem)
        facts = shared_walk(Deployment(program)).moe
        if program.has_dispatch_zone:
            moe += 1
            assert facts is not None, path.name
            assert facts.n_expert_copies > 0 and facts.active_copies > 0, path.name
            assert facts.n_experts > 0 and facts.n_experts_active > 0, path.name
        else:
            dense += 1
            assert facts is None, path.name
    assert moe >= 20 and dense >= 10


class TestACallRebindsAWholeDispatch:
    """An expert bank's copies and the destinations its router chooses among
    count the same experts, so a call that rebinds one rebinds the other.

    DeepSeek-V4.1-Flash's three DSpark draft stages are the case: they run a
    128-expert bank at top-3 where the backbone runs 384 at top-6. The rows
    one expert receives are `B·s·k/destinations`, so a stage that rebound
    only its bank would divide its own tokens by the backbone's 384 and run
    a third of the rows its router sends.
    """

    def test_a_draft_stage_dispatches_over_its_own_destinations(self):
        # 8 x 2048 tokens x top-3 over 128 destinations = 384 rows per expert.
        analysis = shared_walk(
            Deployment(shared_program("deepseek-ai/DeepSeek-V4.1-Flash"))
            .at(batch=8, input_tokens=2048))
        buffers = {
            node.template: tuple(int(extent) for extent in
                                 analysis.shapes.values[node.uid].out)
            for node in analysis.program.nodes() if node.op == "permute"}
        drafts = {name: shape for name, shape in buffers.items()
                  if name.startswith("dspark_stage_")}
        assert len(drafts) == 3, sorted(buffers)
        assert set(drafts.values()) == {(128, 384, 5120)}, drafts
        # The backbone's own bank, for contrast: 384 experts at top-6.
        assert {shape for name, shape in buffers.items()
                if name not in drafts} == {(384, 256, 5120)}
