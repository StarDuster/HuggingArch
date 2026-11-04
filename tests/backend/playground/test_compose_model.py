"""Whole-model composition editor — layer groups, swaps, dims.

A no-edit composition must be byte-for-byte the Inspector's projection of the
base; a swap changes its own group's rows and nothing else; the base spec and
every template another group shares are never modified. The corpus-wide
versions of the first two gates are in `test_compose_corpus.py`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from backend.arch.eval.program import Program
from backend.arch.eval.projection import project_model
from backend.arch.eval.transforms import Deployment
from backend.arch.loader import SpecLoader
from backend.arch.spec_tree import iter_blocks
from backend.arch.structure import iter_reachable_components
from backend.arch.validate._cross_layer_channels import run_cross_layer_checks
from backend.playground import compose as cm
from backend.arch.eval.precision import AS_SHIPPED
from backend.inference.parallel_config import ParallelConfig
from backend.measure.cache import KeyedCache, MemoryBackend
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot, snapshot_config

L = 2048


@pytest.fixture(scope="module")
def loader() -> SpecLoader:
    return SpecLoader()


def _composition(model_id, loader):
    return cm.Composition(resolve_snapshot(model_id, loader),
                          model_id=model_id, loader=loader)


def _cost(composition):
    return composition.cost(batch=1, input_tokens=L, precision=AS_SHIPPED)


# -- the unedited cost, once per model -----------------------------------
#
# Six tests here read a model's cost BEFORE making their edit, and three of
# those models are DeepSeek-V3. The before is the same report whatever the
# test goes on to edit, so the module keeps it per model and the file's
# tests share it. They only read it (`_changed_groups` copies out of it),
# and `cost()` returns a report rather than a view -- which is why `after`
# differs from `before` at all -- so an edit made later cannot reach a
# `before` already handed out. The memo is this module's and dies with it.

_BASE_COSTS: dict = {}


def _base_cost(model_id, loader):
    """This model's cost with no edit made: the `before` of every edit."""
    # The loader is in the key, not its `id`: this file holds its own
    # `SpecLoader` (the editor writes forked templates into it), and a key
    # that held only an id would stop naming the loader it came from once
    # that loader was collected.
    key = (model_id, loader)
    if key not in _BASE_COSTS:
        _BASE_COSTS[key] = _cost(_composition(model_id, loader))
    return _BASE_COSTS[key]


@pytest.fixture(scope="module", autouse=True)
def _forget_costs():
    """Bound the memo to this module: a module-level dict would
    otherwise hold the costs it kept for the rest of the session,
    and one file's worth resident at a time is the point."""
    yield
    _BASE_COSTS.clear()


_ROW_FACTS = ("count", "total_params", "total_active_params", "total_flops",
              "attention_params", "attention_flops", "ffn_params", "ffn_flops",
              "quant_metadata_params", "total_storage_bytes", "kv_bytes_at_L")


def _changed_groups(before, after):
    return {b["group_id"] for b, a in zip(before["rows"], after["rows"], strict=True)
            if any(b[k] != a[k] for k in _ROW_FACTS)}


def _assert_real_swap_gate(composition, group_id, model_id):
    """§3.1 step 3: the repointed declaration's `params` are the new
    template's binding, and the consumers that read them still walk."""
    declaration = composition.declaration(group_id)
    block = composition.resolved().spec.get_block(declaration.template)
    if block.synth_template:
        assert declaration.params == block.synth_params
    resolved = composition.resolved()
    list(iter_reachable_components(resolved))
    run_cross_layer_checks(model_id, resolved)


@pytest.mark.parametrize("model_id", [
    "meta-llama/Llama-3.2-3B",              # dense
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B",                   # MoE (active != total)
    "meituan-longcat/LongCat-Flash-Thinking-2601",  # one backbone group + MTP
])
def test_no_edit_equals_project_model_base(loader, model_id):
    """Against an INDEPENDENT `project_model` of the base (the Inspector's
    path), so the editor's total cannot drift from the Inspector."""
    base = resolve_snapshot(model_id, loader)
    got = cm.Composition(base, model_id=model_id, loader=loader).cost(
        batch=1, input_tokens=L, precision=AS_SHIPPED, kv_cache_lens=[L])
    ref = project_model(base, batch=1, input_tokens=L)
    assert got["params"] == ref.total_params
    assert got["active_params"] == ref.total_active_params
    assert got["flops"] == ref.total_flops
    assert got["kv_bytes_at_L"] > 0  # kv is really computed, not a stubbed 0
    at_L = next(p for p in got["kv_storage_by_cache_len"] if p["cache_len"] == L)
    assert at_L["storage_bytes"] == got["kv_bytes_at_L"]


@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen3-8B",                        # GQA, 36 layers
    "deepseek-ai/DeepSeek-V3",              # MLA, dense + MoE layers, MTP aside
])
def test_decode_step_reads_every_layers_cache(loader, model_id):
    """A full-attention decode step reads the whole cache the prompt wrote,
    in every layer: its KV read is the prefill's whole-batch KV write. The
    editor once summed one copy of each leaf and reported one layer's read
    (Qwen3-8B, B=8, L=4096: 134,217,728 B for 4,831,838,208)."""
    got = _composition(model_id, loader).cost(
        batch=8, input_tokens=4096, precision=AS_SHIPPED)
    phases = got["phases"]
    assert phases["decode"]["kv_read_bytes"] == phases["prefill"]["kv_step_bytes"]


def test_step_traffic_is_the_serving_steps(loader):
    """The editor's step traffic is serving's per-op rows summed: the same
    step `pipeline.analyze` prices, every layer as often as it runs."""
    model_id, batch, input_tokens = "Qwen/Qwen3-8B", 8, 4096
    base = resolve_snapshot(model_id, loader)
    got = cm.Composition(base, model_id=model_id, loader=loader).cost(
        batch=batch, input_tokens=input_tokens, precision=AS_SHIPPED)
    for phase in ("prefill", "decode"):
        rows = analyze(
            base, parallel=ParallelConfig(), batch_size=batch,
            avg_input_tokens=input_tokens, avg_output_tokens=0,
            phase=phase,
            precision=AS_SHIPPED,
            breakdown=False,
        ).per_op_timing
        kv = sum(r["kv_read_bytes_per_rank"] for r in rows)
        step = got["phases"][phase]
        # A phase's FLOPs are on its bytes' basis: the serving step's rows.
        assert step["flops"] == sum(r["flops_per_rank"] for r in rows)
        # A conversion no cast could be grafted for is read-side traffic
        # (`StepTraffic.of_row`).
        assert step["act_bytes_naive"] == sum(
            r["act_in_bytes_per_rank"] + r["act_out_bytes_per_rank"]
            + r["convert_bytes_per_rank"]
            for r in rows) - kv
        assert step["weight_read_bytes"] == sum(
            r["weight_read_bytes_per_rank"] + r["metadata_bytes_per_rank"]
            for r in rows)
        # A fusion only removes intermediates from HBM.
        assert 0 < step["act_bytes_fused"] < step["act_bytes_naive"]
        if phase == "decode":
            assert step["kv_read_bytes"] == kv


@pytest.mark.parametrize("model_id", [
    "deepseek-ai/DeepSeek-V3",   # FP8 casts, MoE, MTP aside: every difference
    "meta-llama/Llama-3.2-3B",   # BF16 dense: the logits alone
])
@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_phase_flops_are_the_main_stacks_walk_plus_its_casts(
        loader, model_id, phase):
    """A phase's FLOPs, by a second route: the walk of the main stack at the
    positions the step samples logits for — every leaf at the copies the walk
    runs it at, every layer as often as the model runs it — plus the
    activation casts serving grafts. A layer-count multiplier lost or doubled
    on either side moves one and not the other."""
    from backend.inference.materialize_cast import CAST_OP
    from backend.inference.step_eval import sampled_logit_positions

    base = resolve_snapshot(model_id, loader)
    got = cm.Composition(base, model_id=model_id, loader=loader).cost(
        batch=1, input_tokens=L, precision=AS_SHIPPED)
    walk = Deployment(Program.of(base), precision=AS_SHIPPED).at(
        batch=1, input_tokens=L,
        logit_positions=sampled_logit_positions(phase)).during(phase).analyze()
    program = walk.program
    main = set(program.included_stack_names(include_aux=False))
    walked = sum(
        int(cost.flops * walk.instances.values.get(uid, 1))
        for uid, cost in walk.cost.leaves().values.items()
        if program.stack_of(program.by_uid(uid)) in main)
    casts = sum(
        int(row["flops_per_rank"]) for row in analyze(
            base, parallel=ParallelConfig(), batch_size=1,
            avg_input_tokens=L, avg_output_tokens=0, phase=phase,
            precision=AS_SHIPPED, breakdown=False).pre_per_op_timing
        if row["op"] == CAST_OP)
    assert got["phases"][phase]["flops"] - casts == walked


def test_groups_are_declarations(loader):
    # Kimi-K3's vision tower is declared first, then layer 0, the MLA
    # layers, the seven restart layers and the other 61. Its decoder is a
    # `read: all` stack, so the MLA and KDA declarations each stand for one
    # `iter_blocks` record per history length; a group is the declaration.
    composition = _composition("moonshotai/Kimi-K3", loader)
    groups = composition.groups()
    assert [(g.meta.aux_kind, g.meta.count, g.editable) for g in groups] == [
        ("vision", 1, False), (None, 1, True), (None, 24, True),
        (None, 7, True), (None, 61, True)]
    assert len(iter_blocks(composition.base)) == 25
    assert groups[3].layers() == tuple(range(12, 85, 12))


def test_nested_slots_are_discovered(loader):
    # DeepSeek-V3's MTP layer wraps a block; its slots sit one level down.
    groups = _composition("deepseek-ai/DeepSeek-V3", loader).groups()
    assert groups[2].meta.aux_kind == "mtp"
    assert [s.path for s in groups[2].slots()] == [
        "transformer/self_attn", "transformer/mlp"]


def test_glm4_wrapper_param_slots_are_discovered(loader):
    """GLM-4's chatglm_block puts attention/ffn in a wrapper node's `params`
    (not a node `op`); they are slots too."""
    (group,) = [g for g in _composition("THUDM/glm-4-9b", loader).groups()
                if g.editable]
    slots = group.slots()
    roles = {s.role for s in slots}
    assert {"attention"} <= roles and any(r.startswith("ffn") for r in roles)
    assert all(s.binding != "op" for s in slots)


def test_slot_candidates_are_by_role(loader):
    group = _composition("deepseek-ai/DeepSeek-V3", loader).groups()[1]
    attn, ffn = group.slot("self_attn"), group.slot("mlp")
    roles_of = {n: loader.component(n).role for n in attn.candidates() + ffn.candidates()}
    assert {roles_of[n] for n in attn.candidates()} <= {"attention", "attention_swa"}
    assert {roles_of[n] for n in ffn.candidates()} == {"ffn_dense", "ffn_moe"}
    assert "attention_indexer" not in roles_of.values()
    assert "router" not in roles_of.values()


def test_moe_attention_swap_leaves_mtp_alone(loader):
    """The MTP layer's inner block IS the MoE group's template (same name, one
    object). Rebinding that template in place would change 59 layers when the
    user asked for 58; the swap builds a new template instead."""
    moe = "pre_norm__attn-mla_attention_ffn-moe_ffn_gate_routing_bias_norm-rmsnorm"
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    base_moe = composition.base.spec.get_block(moe)
    before = _base_cost("deepseek-ai/DeepSeek-V3", loader)
    assert composition.swap(1, "self_attn", "mla_output_gate_attention")
    after = _cost(composition)
    assert _changed_groups(before, after) == {1}
    assert after["aux_params"] == before["aux_params"]
    # the shared template is untouched, in the base and in the edited copy
    edited = composition.resolved().spec
    assert composition.base.spec.get_block(moe) is base_moe
    assert edited.get_block(moe) is base_moe
    assert base_moe.graph["self_attn"].op == "mla_attention"
    _assert_real_swap_gate(composition, 1, "deepseek-ai/DeepSeek-V3")


def test_role_kind_mismatch_is_refused(loader):
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    assert not composition.swap(1, "self_attn", "swiglu_ffn")
    assert not composition.swap(0, "no_such_node", "mla_attention")
    assert not composition.swap(0, "self_attn", "chatglm_block")
    kinds = [r["kind"] for r in composition.refusals()]
    assert kinds == ["role_kind", "unknown_slot", "role_kind"]
    assert all(r["group_id"] in (0, 1) for r in composition.refusals())


def test_read_only_group_is_refused(loader):
    composition = _composition("moonshotai/Kimi-K3", loader)
    assert not composition.swap(0, "anything", "mla_attention")
    assert composition.refusals()[0]["kind"] == "read_only_group"


def test_global_dim_override_moves_derived_dims(loader):
    """`n_experts` feeds `n_expert_copies` / `n_route_targets`; the
    override re-resolves the spec, so the edited model
    is exactly the model whose config says 64 experts."""
    model_id = "Qwen/Qwen3-30B-A3B"
    composition = _composition(model_id, loader)
    composition.set_dims({"n_experts": 64})
    dims = composition.resolved().dim_values
    assert {dims[k] for k in ("n_experts", "n_expert_copies",
                              "n_route_targets")} == {64}
    base = composition.base
    direct = loader.resolve(base.spec, {**snapshot_config(model_id), "num_experts": 64},
                            model_id=model_id)
    ref = project_model(direct, batch=1, input_tokens=L)
    got = _cost(composition)
    assert got["params"] == ref.total_params
    assert got["params"] < project_model(base, batch=1, input_tokens=L).total_params


def test_layer_count_below_a_literal_table_is_refused(loader):
    """Step-3.7's MoE layers are a literal `layers: [4, 8, ..., 44]` table;
    at 12 layers most of them are not layers of the stack. (MiMo-V2-Flash,
    the earlier example, selects its groups from config lists since
    8e117207d, and a per-layer schedule list is clamped, not refused.)"""
    composition = _composition("stepfun-ai/Step-3.7-Flash", loader)
    composition.set_dims({"n_layers": 12})
    (refusal,) = composition.refusals()
    assert refusal["kind"] == "layer_span" and "12 layers" in refusal["detail"]


def test_unresolvable_dim_override_is_refused(loader):
    composition = _composition("meta-llama/Llama-3.2-3B", loader)
    composition.set_dims({"no_such_dim": 3})
    (refusal,) = composition.refusals()
    assert refusal["kind"] == "dims" and "no_such_dim" in refusal["detail"]


def test_group_dims_bind_only_that_group(loader):
    """gemma-4: put the full-attention component on the 40 SWA layers with the
    full group's own head binding; only that group moves, and the binding is
    kept as written."""
    composition = _composition("google/gemma-4-12B-it", loader)
    before = _base_cost("google/gemma-4-12B-it", loader)
    # groups 0 and 1 are the vision embedder's image and video calls (the
    # video call since 613dc20a3 / 0c7c0b73c); 2 the full layers, 3 the SWA
    assert composition.swap(3, "self_attn", "gqa_qknorm_keqv_attention")
    plain = _cost(composition)
    assert composition.set_group_dims(
        3, {"n_kv_heads": "n_kv_heads_full", "d_head": "d_head_full"})
    bound = _cost(composition)
    assert _changed_groups(before, plain) == {3}
    assert _changed_groups(before, bound) == {3}
    assert composition.declaration(3).dims == {
        "n_kv_heads": "n_kv_heads_full", "d_head": "d_head_full"}
    # the full group's binding keeps the fewer kv heads: less cache than plain
    assert bound["kv_bytes_at_L"] < plain["kv_bytes_at_L"]


# --- refusals (§5.3) --------------------------------------------------------


def test_removing_the_topk_producers_is_refused(loader):
    """GLM-5.2's shared layers read the top-k of the nearest full layer
    before them (`reads_from: {nearest: {from: config.indexer_types, is:
    full}}`). Replacing the 18 full layers' DSA with plain MLA leaves full
    layers in the set that no longer write it: refused on the reading
    group, naming them and the layers that still write it."""
    composition = _composition("zai-org/GLM-5.2", loader)
    assert composition.swap(1, "self_attn", "mla_attention")
    (refusal,) = composition.refusals()
    assert refusal["kind"] == "stream" and refusal["group_id"] == 2
    assert "layers 6, 10, 14" in refusal["detail"]
    assert "are in the set and do not write 'dsa_topk'" in refusal["detail"]
    assert "layers 0-2 write it" in refusal["detail"]


def test_every_layer_its_own_indexer_passes(loader):
    """The shared layers given the full indexer (GLM-5.1's layout): every
    layer produces its own top-k and none reads one, so the group's
    `reads_from` goes with the shared component it described."""
    composition = _composition("zai-org/GLM-5.2", loader)
    before = _base_cost("zai-org/GLM-5.2", loader)
    assert composition.swap(2, "self_attn", "mla_dsa_attention")
    assert composition.refusals() == []
    assert composition.declaration(2).reads_from is None
    assert _changed_groups(before, _cost(composition)) == {2}


@pytest.mark.parametrize("group_id", [0, 2])   # dense layers, MTP layer
def test_a_group_cannot_change_its_interface_width(loader, group_id):
    """`d_model` is the width of the residual a backbone group takes in and
    hands on, and of what the MTP layer takes from the backbone; one group
    changing it would not compose with its neighbours."""
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    assert "d_model" in composition.interface_dims(group_id)
    assert not composition.set_group_dims(group_id, {"d_model": 4096})
    (refusal,) = composition.refusals()
    assert refusal == {**refusal, "kind": "interface_dim", "group_id": group_id}
    assert "d_model" in refusal["detail"]
    assert composition.resolved().spec is composition.base.spec  # not applied


def test_a_swap_to_the_wrong_kind_is_refused_on_gpt_oss(loader):
    composition = _composition("openai/gpt-oss-20b", loader)
    assert not composition.swap(0, "self_attn", "swiglu_ffn")
    assert [r["kind"] for r in composition.refusals()] == ["role_kind"]


def test_a_stale_skeleton_is_refused(loader):
    """The editor sends back each group's template, count and assignment;
    all three must still match `iter_blocks(base)[group_id]`."""
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    meta = composition.groups()[1].meta
    now = {"template_name": meta.template_name, "count": meta.count,
           "assignment_index": meta.assignment_index}
    assert composition.check_skeleton(1, **now)
    for field, stale in (("template_name", "block_that_was"),
                         ("count", meta.count + 1),
                         ("assignment_index", 0)):
        assert not composition.check_skeleton(1, **{**now, field: stale})
    kinds = [(r["kind"], r["group_id"]) for r in composition.refusals()]
    assert kinds == [("stale_skeleton", 1)] * 3


# --- per-group dims (§3.4) ---------------------------------------------------


def test_a_bare_template_group_takes_a_dims_edit(loader):
    """Llama's one group is declared by a bare `template:`. Its dims edit is
    applied by declaring the stack as the one `rest:` assignment it is
    equivalent to: more query heads, more attention weights, nothing
    refused, and the group keeps its id."""
    composition = _composition("meta-llama/Llama-3.2-3B", loader)
    before = _base_cost("meta-llama/Llama-3.2-3B", loader)
    assert composition.groups()[0].address[1] is None
    assert composition.set_group_dims(0, {"n_heads": 48}), composition.refusals()
    assert composition.refusals() == []
    assert composition.group_dim_changes(0)["n_heads"]["group"] == 48
    assert composition.declaration(0).dims == {"n_heads": "48"}
    after = _cost(composition)
    assert [r["group_id"] for r in after["rows"]] == [0]
    assert after["params"] > before["params"]


@pytest.mark.parametrize("edit, said", [
    ({"n_heads": -8}, "whole number, 0 or more"),
    ({"not_a_dim": 3}, "'not_a_dim' is not a dim"),
])
def test_a_dim_edit_that_is_not_one_is_refused_in_words(loader, edit, said):
    for scope in ("group", "global"):
        composition = _composition("meta-llama/Llama-3.2-3B", loader)
        if scope == "group":
            assert not composition.set_group_dims(0, edit)
        else:
            composition.set_dims(edit)
        (refusal,) = composition.refusals()
        assert refusal["kind"] == "dims" and said in refusal["detail"], scope


def test_group_dims_move_what_derives_from_them(loader):
    """Halving the experts of DeepSeek-V3's MoE group moves the dims derived
    from the count inside that group only, as a global edit would move them
    everywhere."""
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    before = _base_cost("deepseek-ai/DeepSeek-V3", loader)
    assert composition.set_group_dims(1, {"n_experts": 128})
    changes = composition.group_dim_changes(1)
    for name in ("n_experts", "n_expert_copies", "n_route_targets"):
        assert changes[name] == {"global": 256, "group": 128}, name
    # What the group's declaration says: the edited names, and no
    # `derived:` name — its frame derives those again, to the same values
    # the report gives.
    layer_path, index = composition._address(1)
    written = composition._layer(
        composition.resolved().spec, layer_path).assignments[index].dims
    assert written["n_experts"] == "128"
    assert not {"n_expert_copies", "n_route_targets"} & set(written)
    pool = Deployment(Program.of(composition.resolved())).analyze().moe
    assert (pool.n_expert_copies, pool.n_route_targets) == (128, 128)
    after = project_model(composition.resolved(), batch=1, input_tokens=L)
    changed = {
        row["group_id"]
        for row, (_, _, block) in zip(
            before["rows"], after.block_results, strict=True)
        if int(block.total_params) != row["total_params"]
    }
    assert changed == {1}
    assert after.total_params < before["params"]
    assert composition.resolved().dim_values["n_experts"] == 256


def test_a_derived_dim_is_not_an_edit(loader):
    """`n_expert_copies` / `n_route_targets` are derived in every frame; the
    edit that moves them is `n_experts` (or `n_zero_experts`), globally or in
    a group, and the refusal says so."""
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    composition.set_dims({"n_expert_copies": 64})
    assert not composition.set_group_dims(1, {"n_route_targets": 128})
    refusals = composition.refusals()
    assert [r["kind"] for r in refusals] == ["dims", "dims"]
    assert "`n_expert_copies` is derived" in refusals[0]["detail"]
    assert "`n_route_targets` is derived" in refusals[1]["detail"]
    assert "`n_zero_experts`" in refusals[1]["detail"]
    assert composition.resolved().dim_values["n_expert_copies"] == 256


def test_group_expert_edit_does_not_bill_resident_as_weight_read(loader):
    """A group-local expert count still prices a step the serving way:
    prefill streams the (smaller) bank, decode streams the routed share.
    Neither phase's weight_read_bytes is the resident storage copy."""
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    assert composition.set_group_dims(1, {"n_experts": 128})
    cost = _cost(composition)
    resident = cost["weight_bytes"]
    prefill = cost["phases"]["prefill"]["weight_read_bytes"]
    decode = cost["phases"]["decode"]["weight_read_bytes"]
    assert 0 < decode < prefill < resident


def test_a_refused_group_edit_does_not_ride_along(loader):
    """A refused per-group edit is not kept: the next edit of that group is
    judged on its own keys, not refused again for the first one's."""
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    before = _base_cost("deepseek-ai/DeepSeek-V3", loader)
    assert not composition.set_group_dims(1, {"d_model": 4096})
    assert composition.set_group_dims(1, {"n_experts": 128})
    assert composition.declaration(1).dims["n_experts"] == "128"
    assert "d_model" not in composition.declaration(1).dims
    assert "d_model" not in composition.group_dim_changes(1)
    assert [r["kind"] for r in composition.refusals()] == ["interface_dim"]
    after = project_model(composition.resolved(), batch=1, input_tokens=L)
    changed = {
        row["group_id"]
        for row, (_, _, block) in zip(
            before["rows"], after.block_results, strict=True)
        if int(block.total_params) != row["total_params"]
    }
    assert changed == {1}


def test_group_dims_follow_a_later_global_edit(loader):
    """A group edit written over the model's dims is derived again when the
    global dims move, whichever came first."""
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    assert composition.set_group_dims(1, {"n_experts": "n_experts // 2"})
    composition.set_dims({"n_experts": 64})
    assert composition.group_dim_changes(1)["n_expert_copies"] == {
        "global": 64, "group": 32}


def test_local_dims_show_the_groups_own_binding(loader):
    """gemma-4's full-attention group binds `n_kv_heads` to its own knob
    (value 1, a flag-sized value the symbolic walk keeps concrete — the
    closure scan from the block template is what finds it)."""
    composition = _composition("google/gemma-4-12B-it", loader)
    # groups 0 and 1 are the vision embedder's image and video calls
    # (613dc20a3 / 0c7c0b73c); 2 the full layers
    dims = composition.local_dims(2)
    assert dims["n_kv_heads"] == {"value": 1, "global": 8,
                                  "binding": "n_kv_heads_full",
                                  "group": "Backbone"}
    assert dims["d_model"]["interface"] is True
    assert "n_vocab" not in dims and "n_layers" not in dims


def test_local_dims_come_under_their_registry_section(loader):
    """Each dim names the section `dims.yaml` files it under, and the dims
    come in the registry's section order — the editor groups by that, not by
    what a name looks like."""
    from backend.arch import dims as dims_mod
    sections = dims_mod.groups()
    title = {g.slug: g.title for g in sections}
    rank = {g.title: i for i, g in enumerate(sections)}
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    dims = composition.local_dims(1)
    for name, row in dims.items():
        sym = dims_mod.get(name)
        assert row["group"] == (title[sym.group] if sym else None), name
    order = [rank.get(row["group"], len(rank)) for row in dims.values()]
    assert order == sorted(order)
    assert {"Backbone", "MoE routing", "MLA (Multi-head Latent Attention)"} <= {
        row["group"] for row in dims.values()}


# --- missing dims are errors, not refusals (§5.3-6) --------------------------


def test_missing_dims_are_scanned_from_the_group_template(loader):
    """MLA in Llama's one group: the scan starts at the group's block
    template (which the component closure alone cannot bind) and names
    MLA's latent dims against the slot that brought them."""
    composition = _composition("meta-llama/Llama-3.2-3B", loader)
    assert composition.swap(0, "self_attn", "mla_attention")
    template = composition.declaration(0).template
    from backend.playground.module_bench import unbound_dims
    assert unbound_dims(composition.resolved(), {}, component=template)
    (error,) = composition.unbound_dims()
    assert (error["group_id"], error["slot"], error["component"]) == (
        0, "self_attn", "mla_attention")
    assert "d_kv_lora" in error["missing_dims"]


@pytest.fixture
def offline(monkeypatch):
    """compose_groups goes through fetch_config; feed it the snapshot config so
    the endpoint-level path is exercised without the network."""
    monkeypatch.setattr(cm, "fetch_config", snapshot_config)
    monkeypatch.setattr(cm, "fetch_metadata", lambda m, overlay_dir=None: {})


def _echo(model_id, group_id=0):
    g = cm.model_skeleton(model_id)["groups"][group_id]
    return {k: g[k] for k in ("group_id", "template_name", "count",
                              "assignment_index")}


def test_swap_changes_the_total(offline):
    # gqa -> mla needs MLA's latent dims; supply them via dim_overrides
    r = cm.compose_groups(
        base_model_id="meta-llama/Llama-3.2-3B",
        groups=[{**_echo("meta-llama/Llama-3.2-3B"),
                 "slots": {"self_attn": "mla_attention"}}],
        dim_overrides={"d_kv_lora": 512, "d_q_lora": 1536, "d_qk_nope": 128,
                       "d_qk_rope": 64, "d_v_head": 128},
        precision=AS_SHIPPED)
    assert r["success"], r
    assert r["composed"]["params"] != r["base"]["params"]


def test_missing_dim_names_the_dim_and_does_not_crash(offline):
    """Swapping MLA into Llama without its latent dims fails gracefully with the
    missing dim named (not a raw sympy Relational)."""
    r = cm.compose_groups(
        base_model_id="meta-llama/Llama-3.2-3B",
        groups=[{**_echo("meta-llama/Llama-3.2-3B"),
                 "slots": {"self_attn": "mla_attention"}}],
        precision=AS_SHIPPED)
    assert r["success"] is False
    missing = r["errors"][0]["missing_dims"]
    assert any(d.startswith("d_") for d in missing), missing  # e.g. d_kv_lora


def test_a_walk_that_fails_with_every_dim_bound_is_an_evaluation_error(
        offline, monkeypatch):
    def fails(self, **_):
        raise ValueError("conflicting extents")
    monkeypatch.setattr(cm.Composition, "cost", fails)
    r = cm.compose_groups(base_model_id="meta-llama/Llama-3.2-3B",
                          groups=[], precision=AS_SHIPPED)
    assert r["success"] is False
    # ffb087175: the detail says the edited model does not evaluate (and
    # names the edited dims, none here) before the walk's own message
    assert r["errors"] == [{"kind": "evaluation", "group_id": None,
                            "detail": "the edited model does not evaluate: "
                                      "conflicting extents"}]


def test_no_edit_compose_reports_decode_and_kv_sweep(offline, loader):
    r = cm.compose_groups(
        base_model_id="meta-llama/Llama-3.2-3B", groups=[], precision=AS_SHIPPED,
        kv_cache_lens=[L, 4096])
    assert r["success"], r
    assert r["composed"]["phases"] == r["base"]["phases"]
    assert r["composed"]["kv_storage_by_cache_len"] == r["base"]["kv_storage_by_cache_len"]
    assert r["contract"]["kv_cache_lens"] == [L, 4096]
    at_L = next(p for p in r["composed"]["kv_storage_by_cache_len"]
                if p["cache_len"] == L)
    assert at_L["storage_bytes"] == r["composed"]["kv_bytes_at_L"]
    spec = resolve_snapshot("meta-llama/Llama-3.2-3B", loader)
    decode = analyze(spec, parallel=ParallelConfig(), batch_size=1,
                     avg_input_tokens=L, avg_output_tokens=0, phase="decode",
                     precision=AS_SHIPPED, breakdown=False)
    assert r["composed"]["phases"]["decode"]["flops"] == sum(
        row["flops_per_rank"] for row in decode.per_op_timing)
    assert r["contract"]["gpu_sku"] is None
    assert "timings" not in r["composed"]



def test_a_cost_is_computed_once_per_identity(offline, monkeypatch):
    """A compose's base side is the same cost whatever the edit, and a
    compose that edits nothing has it on both sides: one computation per
    identity (`Composition.cost_identity`), a request that changes what
    the cost is a function of computes again, and each caller's report is
    its own."""
    monkeypatch.setattr(cm, "_COSTS", KeyedCache("test", MemoryBackend()))
    computed = []
    real = cm.Composition._cost

    def counted(self, **kwargs):
        computed.append(kwargs)
        return real(self, **kwargs)

    monkeypatch.setattr(cm.Composition, "_cost", counted)

    def compose(**kwargs):
        return cm.compose_groups(base_model_id="meta-llama/Llama-3.2-3B",
                                 groups=[], precision=AS_SHIPPED, **kwargs)

    first = compose()
    assert first["success"], first
    assert len(computed) == 1
    flops = first["base"]["phases"]["prefill"]["flops"]
    first["base"]["phases"]["prefill"]["flops"] = -1
    assert compose()["base"]["phases"]["prefill"]["flops"] == flops
    assert len(computed) == 1
    compose(kv_cache_lens=[L])
    assert len(computed) == 2

def test_gpu_sku_times_the_composed_model(offline, monkeypatch):
    """A SKU adds SOL / measured time; empty measured coverage stays null."""
    monkeypatch.setattr(
        "backend.playground.module_bench._measurement_scope", lambda: None)
    r = cm.compose_groups(
        base_model_id="meta-llama/Llama-3.2-3B", groups=[], precision=AS_SHIPPED,
        gpu_sku="B200")
    assert r["success"], r
    assert r["contract"]["gpu_sku"] == "B200"
    for side in ("composed", "base"):
        for phase in ("prefill", "decode"):
            t = r[side]["timings"][phase]
            assert t["gpu_sku"] == "B200"
            assert t["t_sol_s"] > 0
            assert t["t_measured_s"] is None
            assert t["total_rows"] > 0


def test_unknown_gpu_sku_is_an_evaluation_error(offline):
    r = cm.compose_groups(
        base_model_id="meta-llama/Llama-3.2-3B", groups=[], precision=AS_SHIPPED,
        gpu_sku="not-a-gpu")
    assert r["success"] is False
    assert r["errors"][0]["kind"] == "evaluation"
    assert "unknown gpu_sku" in r["errors"][0]["detail"]


@pytest.mark.parametrize("layers, label", [
    ((0, 1, 2), "0–2"),
    ((7,), "7"),
    ((0, 2, 4, 6), "0, 2, …, 6 (every 2)"),
    ((3, 7, 11, 15, 16), "3, 7, …, 15 (every 4) · 16"),
    ((1, 2, 4, 5, 6, 8, 9, 10, 12), "9 layers: 1, 2, 4, 5, …"),
])
def test_layers_label_describes_the_resolved_layers(loader, layers, label):
    group = _composition("deepseek-ai/DeepSeek-V3", loader).groups()[1]
    group.meta = replace(group.meta, layer_indices=layers, count=len(layers))
    assert group.layers_label() == label


def test_bound_with_crosses_backbone_and_mtp(loader):
    composition = _composition("deepseek-ai/DeepSeek-V3", loader)
    peers = composition.bound_with(composition.groups()[1].slot("self_attn"))
    assert [(g.group_id, s.path) for g, s in peers] == [
        (0, "self_attn"), (1, "self_attn"), (2, "transformer/self_attn")]
