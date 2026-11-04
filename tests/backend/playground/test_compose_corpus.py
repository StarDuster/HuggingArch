"""Corpus gates of the whole-model editor.

1. No edit = the Inspector, on every snapshot: the five totals against their
   independent sources (`project_model`, the weight ledger `capacity.weights`,
   `cache_view`), and every numeric field of every `block_results` row.
2. Identity swap, on every snapshot: every slot of every backbone / MTP group
   swapped to the component it already holds, through the real swap (repoint
   by declaration, params synced, synthesis or fork as the slot dictates) —
   the numbers do not move by a byte.
3. Locality: a real swap moves its own group's rows and nothing else.
4. Every editable group has a boundary (`interface_dims`) and a readable set
   of its own dims (`local_dims`).
"""

from __future__ import annotations

import dataclasses

import pytest

from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.arch.eval.capacity import weights
from backend.arch.eval.projection import project_model
from backend.arch.loader import SpecLoader
from backend.arch.structure import iter_reachable_components
from backend.arch.validate._cross_layer_channels import run_cross_layer_checks
from backend.playground import compose as cm
from backend.arch.eval.precision import AS_SHIPPED
from tests.snapshots import SNAPSHOT_DIR, resolve_snapshot

L = 2048
_SNAPSHOTS = sorted(p.stem for p in SNAPSHOT_DIR.glob("*.json"))
_ROW_FACTS = ("count", "total_params", "total_active_params", "total_flops",
              "attention_params", "attention_flops", "ffn_params", "ffn_flops",
              "quant_metadata_params", "total_storage_bytes", "kv_bytes_at_L")


@pytest.fixture(scope="module")
def loader() -> SpecLoader:
    return SpecLoader()


def _model_id(stem: str) -> str:
    return stem.replace("__", "/", 1)


def _composition(stem, loader):
    return cm.Composition(resolve_snapshot(stem, loader),
                          model_id=_model_id(stem), loader=loader)


def _cost(composition):
    return composition.cost(batch=1, input_tokens=L, precision=AS_SHIPPED)


# -- the unedited cost, once per model -----------------------------------
#
# Every swap test reads the model's cost BEFORE its edit, and the before is
# the same report however the test goes on to edit: three of the swap cases
# are DeepSeek-V3's groups, three touch gpt-oss, and the walkthrough test
# asks for three models the parametrized cases already costed. Eleven reads
# for five models.
#
# So the module keeps the unedited cost per model and its tests share it.
# The tests only read it (`_facts` / `_changed_groups` copy out of it), and
# the report is a snapshot rather than a view -- `cost()` is what makes the
# numbers, which is why `after` differs from `before` at all -- so a later
# edit of the composition cannot reach a `before` already handed out. The
# memo is this module's and dies with it.

_BASE_COSTS: dict = {}


def _base_cost(stem, loader):
    """This model's cost with no edit made: the `before` of every edit."""
    # The loader is in the key, not its `id`: these files each hold their
    # own `SpecLoader` (the editor writes forked templates into it), and a
    # key that held only an id would stop naming the loader it came from
    # once that loader was collected.
    key = (stem, loader)
    if key not in _BASE_COSTS:
        _BASE_COSTS[key] = _cost(_composition(stem, loader))
    return _BASE_COSTS[key]


@pytest.fixture(scope="module", autouse=True)
def _forget_costs():
    """Bound the memo to this module: a module-level dict would
    otherwise hold the costs it kept for the rest of the session,
    and one file's worth resident at a time is the point."""
    yield
    _BASE_COSTS.clear()


def _facts(cost):
    """Every number the editor reports, template names aside (a forked
    template has a new name by construction)."""
    return (
        {k: v for k, v in cost.items() if k not in ("rows", "aux_params", "model_level")},
        [{k: row[k] for k in (*_ROW_FACTS, "group_id", "block_index", "aux_kind")}
         for row in cost["rows"]],
        cost["aux_params"], cost["model_level"])


@pytest.mark.regression  # every snapshot
@pytest.mark.parametrize("stem", _SNAPSHOTS)
def test_no_edit_equals_the_inspector(loader, stem):
    base = resolve_snapshot(stem, loader)
    got = cm.Composition(base, model_id=_model_id(stem), loader=loader).cost(
        batch=1, input_tokens=L, precision=AS_SHIPPED, kv_cache_lens=[L])
    # The Inspector's report: the whole checkpoint, each encoder walked in
    # its own encode binding (`project_model(..., checkpoint=True)`).
    ref = project_model(base, batch=1, input_tokens=L, checkpoint=True)
    view = cache_view_from_analysis(ref.analysis)
    assert (got["params"], got["active_params"], got["flops"]) == (
        ref.total_params, ref.total_active_params, ref.total_flops)
    # The stored model: parameters, their scales and buffers.
    ledger = weights(ref.analysis)
    assert got["weight_bytes"] == ledger.resident
    assert got["kv_bytes_at_L"] == int(view.storage_bytes_at(L))
    at_L = next(p for p in got["kv_storage_by_cache_len"] if p["cache_len"] == L)
    assert at_L["storage_bytes"] == got["kv_bytes_at_L"]
    assert got["aux_params"] == ref.aux_params
    assert len(got["rows"]) == len(ref.block_results)
    numeric = [f.name for f in dataclasses.fields(ref.block_results[0][2])
               if f.type in ("int", int)] if ref.block_results else []
    for row, (template, count, block) in zip(got["rows"], ref.block_results, strict=True):
        assert (row["template"], row["count"]) == (template, count)
        for name in numeric:
            assert row[name] == getattr(block, name), (template, name)
        # A group stores its layers' parameters, and beside them only their
        # scales and buffers — none when the checkpoint has neither.
        parameters = count * block.total_storage_bytes
        assert row["group_totals"]["storage_bytes"] >= parameters, template
        if not (ledger.metadata or ledger.buffers):
            assert row["group_totals"]["storage_bytes"] == parameters, template
    backbone_kv = sum(r["kv_bytes_at_L"] for r in got["rows"] if r["kv_bytes_at_L"] is not None)
    assert backbone_kv == pytest.approx(got["kv_bytes_at_L"], abs=len(got["rows"]))
    # a group's totals are its whole contribution: the groups plus the model
    # level are the whole model, in every column the editor shows per group
    for key, whole in (("params", "params"), ("active_params", "active_params"),
                       ("flops", "flops"), ("storage_bytes", "weight_bytes")):
        assert (sum(r["group_totals"][key] for r in got["rows"])
                + got["model_level_totals"][key]) == got[whole], key


@pytest.mark.regression  # every snapshot
@pytest.mark.parametrize("stem", _SNAPSHOTS)
def test_identity_swap_is_byte_identical(loader, stem):
    """Also: no dim is missing (the static scan from each group's template
    finds nothing on an identity swap), no stream reads out of reach, and an
    empty per-group dims edit writes nothing and is applied on every group,
    a bare `template:` one included."""
    before = _composition(stem, loader)
    composition = _composition(stem, loader)
    for group in composition.groups():
        if not group.editable:
            continue
        for slot in group.slots():
            assert composition.swap(group.group_id, slot.path, slot.component), \
                composition.refusals()
    assert composition.refusals() == []
    assert composition.unbound_dims() == []
    def written(group_id):
        declared = composition.declaration(group_id).dims or {}
        return {k: str(v) for k, v in declared.items()}

    for group in composition.groups():
        if group.editable:
            before_dims = written(group.group_id)
            composition.set_group_dims(group.group_id, {})
            assert written(group.group_id) == before_dims
    assert composition.refusals() == []
    assert _facts(_cost(composition)) == _facts(_cost(before))


@pytest.mark.regression  # every snapshot
@pytest.mark.parametrize("stem", _SNAPSHOTS)
def test_a_group_dim_set_to_its_own_value_is_byte_identical(loader, stem):
    """Every editable group takes a per-group dims edit. One that binds a
    dim to itself (`n_heads: n_heads`) writes a `dims:` — on a bare
    `template:` stack by declaring it as the one `rest:` assignment it is
    equivalent to — and no number moves."""
    before = _composition(stem, loader)
    composition = _composition(stem, loader)
    edited = []
    for group in composition.groups():
        if not group.editable:
            continue
        own = {name: row["value"] for name, row
               in composition.local_dims(group.group_id).items()
               if not row.get("interface") and "global" not in row}
        if not own:
            continue
        name = sorted(own)[0]
        if composition.set_group_dims(group.group_id, {name: name}):
            edited.append(group.group_id)
    assert composition.refusals() == []
    for group_id in edited:
        assert composition.declaration(group_id).dims, group_id
    assert _facts(_cost(composition)) == _facts(_cost(before))


@pytest.mark.regression  # every snapshot
@pytest.mark.parametrize("stem", _SNAPSHOTS)
def test_every_group_has_an_interface_and_its_own_dims(loader, stem):
    """Every editable group's boundary is an expression in the residual
    width, and its own dims — read off the model's walk — carry it, marked."""
    composition = _composition(stem, loader)
    for group in composition.groups():
        if not group.editable:
            continue
        assert "d_model" in composition.interface_dims(group.group_id)
        dims = composition.local_dims(group.group_id)
        assert dims["d_model"].get("interface") is True, group.template


def test_identity_swap_dispatches_by_slot(loader):
    # DeepSeek-V3's groups are synthesized from pre_norm + params: the swap
    # re-synthesizes to the same name and syncs params with the binding.
    ds = _composition("deepseek-ai__DeepSeek-V3", loader)
    group = ds.groups()[2]   # MTP: nested param binding
    assert ds.swap(2, "transformer/self_attn", "mla_attention")
    decl = ds.declaration(2)
    assert decl.template == group.template
    assert decl.params == ds.resolved().spec.get_block(decl.template).synth_params
    # Kimi-K3's blocks are handwritten: the swap forks the block.
    kimi = _composition("moonshotai__Kimi-K3", loader)
    kda = kimi.groups()[4].slot("self_attn").component
    assert kimi.swap(4, "self_attn", kda)
    assert kimi.declaration(4).template == f"block_kda_moe__self_attn-{kda}"


# (model, group_id, slot, component)
_SWAPS = [
    ("deepseek-ai__DeepSeek-V3", 0, "mlp", "moe_ffn_gate_routing_bias"),
    ("deepseek-ai__DeepSeek-V3", 1, "self_attn", "mla_output_gate_attention"),
    ("deepseek-ai__DeepSeek-V3", 2, "transformer/self_attn", "mla_output_gate_attention"),
    pytest.param("moonshotai__Kimi-K3", 4, "self_attn", "mla_output_gate_attention",
                 marks=pytest.mark.regression),  # minutes-long
    ("Qwen__Qwen3-Coder-Next", 1, "layer::attn",
     "gqa_qknorm_partial_rope_sigmoid_gated_attention"),
    ("openai__gpt-oss-20b", 0, "self_attn", "gqa_bias_sink_attention"),
    ("nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", 1, "mixer", "gqa_attention"),
    # groups 0 and 1 are the vision embedder called for images and for video
    # frames, declared ahead of the merge they feed
    ("google__gemma-4-12B-it", 3, "self_attn", "gqa_qknorm_keqv_attention"),
    ("google__gemma-4-12B-it", 2, "self_attn", "swa_qknorm_attention"),
]


def _changed_groups(before, after):
    return {b["group_id"] for b, a in zip(before["rows"], after["rows"], strict=True)
            if any(b[k] != a[k] for k in _ROW_FACTS)}


@pytest.mark.parametrize("stem,group_id,slot,component", _SWAPS)
def test_swap_moves_only_its_group(loader, stem, group_id, slot, component):
    composition = _composition(stem, loader)
    before = _base_cost(stem, loader)
    target = composition.groups()[group_id]
    assert composition.swap(group_id, slot, component), composition.refusals()
    after = _cost(composition)
    assert _changed_groups(before, after) == {group_id}
    assert after["model_level"] == before["model_level"]
    changed_aux = {k for k in before["aux_params"]
                   if before["aux_params"][k] != after["aux_params"][k]}
    assert changed_aux == ({target.meta.group_name} if target.meta.aux_kind else set())
    if target.meta.aux_kind:
        # an aux stack's cache is outside the whole-model KV (Inspector scope)
        assert after["kv_bytes_at_L"] == before["kv_bytes_at_L"]
    # the repointed declaration carries the new binding, and the consumers
    # reading declarations (reachability, cross-layer streams) still walk
    decl = composition.declaration(group_id)
    resolved = composition.resolved()
    block = resolved.spec.get_block(decl.template)
    if block.synth_template:
        assert decl.params == block.synth_params
    reachable = {name for name, _ in iter_reachable_components(resolved)}
    assert component in reachable
    run_cross_layer_checks(_model_id(stem), resolved)


def test_swap_facts_from_the_walkthrough(loader):
    """The direction of each walkthrough swap, as the model structure says it
    must go (not a stored number)."""
    def delta(stem, *swap):
        composition = _composition(stem, loader)
        before = _base_cost(stem, loader)
        assert composition.swap(*swap)
        after = _cost(composition)
        return {k: after[k] - before[k] for k in ("params", "flops", "kv_bytes_at_L")}

    # SWA and GQA sink attention have the same weights; full length > window
    d = delta("openai__gpt-oss-20b", 0, "self_attn", "gqa_bias_sink_attention")
    assert d["params"] == 0 and d["flops"] > 0 and d["kv_bytes_at_L"] > 0
    # a constant recurrent state becomes a cache growing with length
    d = delta("Qwen__Qwen3-Coder-Next", 1, "layer::attn",
              "gqa_qknorm_partial_rope_sigmoid_gated_attention")
    assert d["kv_bytes_at_L"] > 0
    # dense FFN → MoE adds expert banks
    d = delta("deepseek-ai__DeepSeek-V3", 0, "mlp", "moe_ffn_gate_routing_bias")
    assert d["params"] > 0 and d["kv_bytes_at_L"] == 0


def test_a_shared_template_is_not_edited_through_either_group(loader):
    """gpt-oss's SWA group re-synthesized to GQA lands on the full group's
    template; swapping the full group afterwards builds a new template and
    leaves the SWA group where the first swap put it."""
    composition = _composition("openai__gpt-oss-20b", loader)
    assert composition.swap(0, "self_attn", "gqa_bias_sink_attention")
    assert composition.declaration(0).template == composition.declaration(1).template
    first = _cost(composition)
    assert composition.swap(1, "self_attn", "swa_bias_sink_attention")
    second = _cost(composition)
    assert composition.declaration(0).template != composition.declaration(1).template
    assert _changed_groups(first, second) == {1}
