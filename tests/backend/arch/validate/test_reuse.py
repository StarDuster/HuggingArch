"""Drafts that restate one another, or a library component (`_reuse`).

A copy with parts deleted is still a copy: the check compares drafts by how
much of the smaller one the larger holds, edges by op, names ignored. The
comparison itself is `arch/component_reuse.py` (tested in
`tests/backend/arch/test_component_reuse.py`, including the pre-write query
that asks it on a candidate file); here it is the findings a finished spec
gets.
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.arch.component_reuse import edges, held_in
from backend.arch.spec_parsing import _parse_block_graph
from backend.arch.validate._reuse import run_reuse_check

_ATTENTION = {
    "wq": {"op": "linear", "in": ["x"], "dims": {"in": "d_model", "out": "d_model"}},
    "wk": {"op": "linear", "in": ["x"], "dims": {"in": "d_model", "out": "d_model"}},
    "wv": {"op": "linear", "in": ["x"], "dims": {"in": "d_model", "out": "d_model"}},
    "q": {"op": "head_split", "in": ["wq"]},
    "k": {"op": "head_split", "in": ["wk"]},
    "v": {"op": "head_split", "in": ["wv"]},
    "score": {"op": "attention_score", "in": ["q", "k"]},
    "probs": {"op": "softmax", "in": ["score"]},
    "apply": {"op": "attention_apply", "in": ["probs", "v"]},
    "merge": {"op": "head_merge", "in": ["apply"]},
    "wo": {"op": "linear", "in": ["merge"], "dims": {"in": "d_model", "out": "d_model"}},
}


def _graph(nodes, rename="", **extra):
    graph = {f"{rename}{name}": {**node, "in": [
        f"{rename}{ref}" if ref in nodes else ref for ref in node["in"]]}
        for name, node in nodes.items()}
    return _parse_block_graph({"inputs": ["x"], "graph": graph,
                               "outputs": [f"{rename}wo"], **extra})


def _resolved(drafts, library=None):
    library = library or {}
    loader = SimpleNamespace(
        listed_drafts=lambda name: tuple(drafts),
        component_names=lambda: frozenset(library),
        component=library.__getitem__)
    return SimpleNamespace(
        declared_by=loader,
        spec=SimpleNamespace(name="m", components={**drafts, **library}))


def test_names_do_not_matter_edges_do():
    assert held_in(edges(_graph(_ATTENTION)),
                   edges(_graph(_ATTENTION, rename="p_"))) == 1.0


def test_a_copy_with_a_part_deleted_is_reported():
    # The second attention reads its keys from elsewhere: no `wk`.
    cut = {n: v for n, v in _ATTENTION.items() if n != "wk"}
    cut["k"] = {"op": "head_split", "in": ["x"]}
    (finding,) = run_reuse_check(_resolved({
        "a_attention": _graph(_ATTENTION), "b_attention": _graph(cut)}))
    assert finding.kind == "drafts_restate_each_other"
    assert finding.extra["likeness"] >= 0.85


def test_a_decode_form_is_not_a_copy():
    assert run_reuse_check(_resolved({
        "a_attention": _graph(_ATTENTION, decode_variant="a_attention_decode"),
        "a_attention_decode": _graph(_ATTENTION)})) == []


def test_a_draft_restating_a_library_component_is_reported():
    (finding,) = run_reuse_check(_resolved(
        {"my_attention": _graph(_ATTENTION, rename="m_")},
        library={"mha": _graph(_ATTENTION)}))
    assert finding.kind == "draft_restates_library"
    assert finding.extra["library"] == "mha"
