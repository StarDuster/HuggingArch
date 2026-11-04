"""Derived groups shown per declaration (`backend.views.declarations`).

The toy stack of `test_prefix_streams` (7 layers, K = 3) splits two
declarations by history length: `start` (layers 3 and 6, seeing 1 and 2
entries) and `plain` (layers 1-2 and 4-5, seeing 1 and 2). Its rows and its
serving graph carry each group as fields, and the view joins them back under
their declaration: a parent row per op whose columns are its children's sums,
and one folded box per run of a declaration's consecutive groups in the
graph. A model with no derived group passes through untouched.
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from backend.inference.framework_bindings import get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.rendering._compute_graph import serialize_post_tree
from backend.rendering.kv_marks import KVMarks
from backend.inference import declaration_groups
from backend.inference.declaration_groups import declaration_terms
from backend.views import declarations
from tests.backend.arch.test_prefix_streams import _resolved
from tests.snapshots import resolve_snapshot, snapshot_config


def _serve(resolved):
    return analyze(resolved, parallel=ParallelConfig.from_dict({}),
                   framework_binding=get_binding("vllm"),
                   batch_size=1, avg_input_tokens=4, avg_output_tokens=0, breakdown=False)


def _graph(resolved, result):
    return serialize_post_tree(
        result.post_tree, resolved, pre_tree=result.comm_tree,
        fusion_matches=result.fusion_matches, program=result.program,
        shapes=result.shapes, full_shapes=result.full_shapes,
        pre_program=result.pre_program, pre_shapes=result.pre_shapes,
        pre_full_shapes=result.pre_full_shapes,
        kv=KVMarks.of_serving(result))


@pytest.fixture(scope="module")
def toy():
    resolved = _resolved("decl_view_toy")
    result = _serve(resolved)
    return resolved, result


@pytest.fixture(scope="module")
def plain_model():
    model = "Qwen/Qwen3-8B"
    resolve_snapshot(model)
    from backend.arch.loader import SpecLoader
    resolved = SpecLoader().match(snapshot_config(model), model_id=model)
    return resolved, _serve(resolved)


def test_a_group_s_rows_name_its_declaration(toy):
    _resolved_spec, result = toy
    program = result.program
    assert sorted(b.meta.label for b in program.blocks if b.meta.visible) == [
        "plain[Sar=1]", "plain[Sar=2]", "start[Sar=1]", "start[Sar=2]"]
    seen = set()
    for row in result.per_op_timing:
        block = program.block_of(row["uid"])
        meta = program.blocks[block].meta if block is not None else None
        if meta is None or not meta.visible:
            assert "derived_group" not in row, row["aggregation_id"]
            continue
        g = row["derived_group"]
        assert g["base_template"] == meta.template_name
        assert g["visible"] == dict(meta.visible)["hist"]
        assert g["axis"] == "Sar" and g["stream"] == "hist"
        assert g["declaration"] == {
            "layer_path": list(meta.layer_path),
            "assignment_index": meta.assignment_index,
            "base_template": meta.template_name}
        assert g["layer_indices"] == list(meta.layer_indices)
        assert g["count"] == meta.count == len(meta.layer_indices)
        seen.add((g["base_template"], g["visible"], tuple(g["layer_indices"])))
    assert seen == {("start", 1, (3,)), ("start", 2, (6,)),
                    ("plain", 1, (1, 2)), ("plain", 2, (4, 5))}


def test_a_parent_row_is_the_sum_of_its_children(toy):
    _resolved_spec, result = toy
    rows = result.per_op_timing
    parents = declarations.per_op_rows(rows)
    derived = [i for i, r in enumerate(rows) if "derived_group" in r]
    children = [i for p in parents for i in p["children"]]
    # Every group row is under exactly one parent; nothing else is.
    assert sorted(children) == derived and len(set(children)) == len(children)
    # Parents stand where their first child stood.
    firsts = [p["children"][0] for p in parents]
    assert firsts == sorted(firsts)
    for p in parents:
        kids = [rows[i] for i in p["children"]]
        for col in (f.name for f in fields(declarations.RowSums)):
            assert p[col] == sum(k[col] for k in kids), (p["path"], col)
        # A sum of groups has no single bound; its intensity is the summed
        # FLOPs over the summed bytes.
        assert p["bound"] is None
        if p["access_bytes_per_rank"]:
            assert p["arithmetic_intensity"] == (
                p["flops_per_rank"] / p["access_bytes_per_rank"])
        assert p["layers"] == sum(k["derived_group"]["count"] for k in kids)
        assert p["layers"] == sum(h["layers"] for h in p["histogram"])
        assert {k["op"] for k in kids} == {p["op"]}
        decl = {k["derived_group"]["base_template"] for k in kids}
        assert decl == {p["declaration"]["base_template"]}
        assert [h["visible"] for h in p["histogram"]] == [1, 2]
    by_template = {p["declaration"]["base_template"] for p in parents}
    assert by_template == {"start", "plain"}
    plain = next(p for p in parents if p["declaration"]["base_template"] == "plain")
    assert plain["histogram"] == [
        {"visible": 1, "layers": 2, "layer_indices": [1, 2]},
        {"visible": 2, "layers": 2, "layer_indices": [4, 5]}]
    # One parent per op of each declaration: the table is back to the
    # declaration's size.
    assert len(parents) == len(derived) // 2


def test_a_parent_states_the_folds_measured_share_and_adds_nothing(toy):
    """The parent's measured time is the one measured formula's share of the
    terms under it (`batch_sweep.apply_measured` → `Folded.subtotal`), passed
    in by its name. The view adds no children up: it used to, which was a
    second measured aggregation served as the parent's production number, and
    it would have gone on agreeing with the formula only while the formula
    stayed a plain sum of per-row shares."""
    _resolved_spec, result = toy
    rows = result.per_op_timing
    # Children carrying times the view must NOT add: if it summed them, the
    # parent would read 7.0 instead of the share the fold states.
    priced = [{**r, "t_measured_s": 3.5} if "derived_group" in r else r
              for r in rows]
    names = [name for name, _indices in declaration_terms(priced)]
    assert names
    share = {name: 0.25 for name in names}
    share[names[0]] = None                      # not every term measured
    parents = declarations.per_op_rows(priced, share)
    got = {declaration_groups.group_key(
        declarations.declaration_key(rows[p["children"][0]]["derived_group"]),
        p["path"], p["op"]): p["t_measured_s"] for p in parents}
    assert got == share
    # No shares at all: nothing was measured, and nothing is invented.
    assert all(p["t_measured_s"] is None
               for p in declarations.per_op_rows(priced))


def test_the_history_combine_is_shown_varying(toy):
    """The op whose input is the history has a different shape per group;
    the parent says so rather than showing one of them."""
    _resolved_spec, result = toy
    parents = declarations.per_op_rows(result.per_op_timing)
    combine = next(p for p in parents
                   if p["declaration"]["base_template"] == "plain"
                   and p["op"] == "attn_res_combine")
    assert combine["shape_in"] is None
    assert combine["flops_per_rank"] == sum(
        result.per_op_timing[i]["flops_per_rank"] for i in combine["children"])


def test_groups_another_declaration_interleaves_stay_unfolded(toy):
    """In the toy, `plain` (1-2, 4-5) and `start` (3, 6) alternate in layer
    order, so no two consecutive stacks share a declaration and the graph
    comes back as it was drawn."""
    resolved, result = toy
    graph = _graph(resolved, result)
    assert [n["id"] for n in graph["nodes"] if n["op"] == "block_stack"] == [
        "__block__first", "__block__plain[Sar=1]", "__block__start[Sar=1]",
        "__block__plain[Sar=2]", "__block__start[Sar=2]"]
    assert declarations.fold_graph(graph) is graph


@pytest.fixture(scope="module")
def growing():
    """The toy with the history appended before every layer: each layer is
    its own group, so `plain` 1-2 and 4-5 are runs of two groups each."""
    resolved = _resolved("decl_view_growing", before='{ every: "1" }')
    return resolved, _serve(resolved)


def test_the_graph_folds_each_run_of_a_declaration(growing):
    resolved, result = growing
    graph = _graph(resolved, result)
    folded = declarations.fold_graph(graph)
    top = {n["id"]: n for n in folded["nodes"]}
    boxes = [n for n in folded["nodes"] if n["op"] == "block_stack"]
    assert [n["id"] for n in boxes] == [
        "__block__first", "__declaration__1:2@1", "__block__start[Sar=3]",
        "__declaration__1:2@4", "__block__start[Sar=6]"]
    early, late = top["__declaration__1:2@1"], top["__declaration__1:2@4"]
    assert [c["id"] for c in early["children"]] == [
        "__block__plain[Sar=1]", "__block__plain[Sar=2]"]
    assert early["count"] == late["count"] == 2
    assert early["derived_groups"]["histogram"] == [
        {"visible": 1, "layers": 1, "layer_indices": [1]},
        {"visible": 2, "layers": 1, "layer_indices": [2]}]
    # The boxes chain in layer order.
    assert early["inputs"] == ["__block__first"]
    assert top["__block__start[Sar=3]"]["inputs"] == ["__declaration__1:2@1"]
    assert late["inputs"] == ["__block__start[Sar=3]"]
    assert [(e["src"], e["dst"]) for e in late["child_edges"]] == [
        ("__block__plain[Sar=4]", "__block__plain[Sar=5]")]
    # Every root edge joins two top-level nodes; every top-level input names
    # one.
    for edge in folded["edges"]:
        assert edge["src"] in top and edge["dst"] in top, edge
        assert edge["id"].startswith(f"{edge['src']}->{edge['dst']}:")
    for node in folded["nodes"]:
        for ref in node.get("inputs") or []:
            assert ref in top, (node["id"], ref)
    # The groups inside a fold are the boxes the engine drew.
    unfolded = {n["id"]: n for n in graph["nodes"]}
    assert early["children"][1] is unfolded["__block__plain[Sar=2]"]


def test_a_model_without_derived_groups_is_untouched(plain_model):
    resolved, result = plain_model
    assert not any(b.meta.visible for b in result.program.blocks)
    assert not any("derived_group" in r for r in result.per_op_timing)
    assert declarations.per_op_rows(result.per_op_timing) == []
    graph = _graph(resolved, result)
    assert declarations.fold_graph(graph) is graph

