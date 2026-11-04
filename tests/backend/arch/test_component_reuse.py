"""The reuse comparison a writer can run BEFORE authoring a draft
(`arch/component_reuse.py`) — the same one the validator runs after.

What these pin: the pre-write query answers with the share the validator's
finding reports, a graph nothing holds comes back with nothing above the
threshold, a draft sitting in a run overlay is comparable like any other
component, and roles are compared (a block that contains an attention is not
that attention written twice).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from types import SimpleNamespace

import pytest

from backend.arch.component_reuse import (
    CONTAINED,
    comparable,
    edges,
    held_in,
    likenesses,
    main,
    read_candidate,
    render,
    reusable_components,
)
from backend.arch.loader import SpecLoader, overlay_kwargs
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

#: An FFN: no edge of it is an edge of the attention above.
_FFN = {
    "gate": {"op": "linear", "in": ["x"], "dims": {"in": "d_model", "out": "d_ffn"}},
    "up": {"op": "linear", "in": ["x"], "dims": {"in": "d_model", "out": "d_ffn"}},
    "act": {"op": "silu", "in": ["gate"], "dim": "d_ffn"},
    "mul": {"op": "mul", "in": ["act", "up"], "dim": "d_ffn"},
    "down": {"op": "linear", "in": ["mul"], "dims": {"in": "d_ffn", "out": "d_model"}},
    "norm": {"op": "rmsnorm", "in": ["down"], "dim": "d_model"},
    "res": {"op": "add", "in": ["norm", "x"], "dim": "d_model"},
    "norm2": {"op": "rmsnorm", "in": ["res"], "dim": "d_model"},
    "out": {"op": "add", "in": ["norm2", "res"], "dim": "d_model"},
}


def _graph(nodes, rename="", **extra):
    graph = {f"{rename}{name}": {**node, "in": [
        f"{rename}{ref}" if ref in nodes else ref for ref in node["in"]]}
        for name, node in nodes.items()}
    outputs = [f"{rename}{next(reversed(nodes))}"]
    return _parse_block_graph({"inputs": ["x"], "graph": graph,
                               "outputs": outputs, **extra})


def _resolved(drafts, library=None):
    """A resolved spec shaped as `run_reuse_check` reads one."""
    library = library or {}
    loader = SimpleNamespace(
        listed_drafts=lambda name: tuple(drafts),
        component_names=lambda: frozenset(library),
        component=library.__getitem__)
    return SimpleNamespace(
        declared_by=loader,
        spec=SimpleNamespace(name="m", components={**drafts, **library}))


def test_the_pre_write_query_answers_what_the_validator_reports():
    # The draft is the library attention with its key projection deleted —
    # a copy with a part removed, which is what the check is about.
    cut = {n: v for n, v in _ATTENTION.items() if n != "wk"}
    cut["k"] = {"op": "head_split", "in": ["x"]}
    draft = _graph(cut, role="attention", cache={"family": "GQA"})
    library = {"mha": _graph(_ATTENTION, role="attention",
                             cache={"family": "MHA"})}

    (finding,) = run_reuse_check(_resolved({"my_attention": draft}, library))
    (found,) = likenesses(draft, library, name="my_attention")

    assert finding.kind == "draft_restates_library"
    assert finding.extra["library"] == found.name == "mha"
    assert finding.extra["likeness"] == round(found.share, 3)
    assert found.share >= CONTAINED


def test_a_novel_graph_has_nothing_above_the_threshold():
    novel = _graph(_FFN, role="ffn_dense")
    library = {"mha": _graph(_ATTENTION, role="ffn_dense")}
    ranked = likenesses(novel, library, name="my_ffn")
    assert [found.name for found in ranked] == ["mha"]
    assert ranked[0].share < CONTAINED
    assert run_reuse_check(_resolved({"my_ffn": novel}, library)) == []


def test_a_block_that_contains_an_attention_is_not_that_attention_twice():
    # A `role: block` component holds nearly every edge of any attention it
    # calls, so containment is near-total by construction. Reporting it would
    # advise a bind no attention slot can make.
    block = _graph({**_ATTENTION, **{f"ffn_{n}": {**v, "in": [
        f"ffn_{r}" if r in _FFN else "wo" for r in v["in"]]}
        for n, v in _FFN.items()}}, role="block")
    attention = _graph(_ATTENTION, role="attention", cache={"family": "MHA"})
    assert held_in(edges(attention), edges(block)) == 1.0
    assert not comparable("my_attention", attention, "a_block", block)
    assert likenesses(attention, {"a_block": block}, name="my_attention") == []
    assert run_reuse_check(
        _resolved({"my_attention": attention}, {"a_block": block})) == []


def test_a_draft_in_a_run_overlay_is_compared_like_any_component(tmp_path):
    drafts = tmp_path / "components" / "drafts"
    drafts.mkdir(parents=True)
    source = Path(PROJECT_ROOT) / "backend/arch/components/library/mha_attention.yaml"
    candidate = drafts / "my_mha.yaml"
    candidate.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    loader = SpecLoader(**overlay_kwargs(str(tmp_path)))
    name, graph = read_candidate(str(candidate))
    assert name == "my_mha"
    # The overlay draft joins the set a writer can reuse, and the loader is
    # asked where it came from rather than the directories re-globbed.
    assert "my_mha" in reusable_components(loader)
    assert loader.component_file("my_mha") == str(candidate)

    ranked = likenesses(graph, reusable_components(loader), name=name)
    best = [found for found in ranked if found.share >= CONTAINED]
    assert "mha_attention" in [found.name for found in best]
    # The candidate never matches itself: the overlay holds that same file.
    assert "my_mha" not in [found.name for found in ranked]

    text = render(name, graph, ranked, loader.component_file)
    assert "mha_attention" in text
    assert f"at or above {CONTAINED:.2f}" in text


def test_the_command_answers_for_a_draft_in_an_overlay(tmp_path, capsys,
                                                       monkeypatch):
    drafts = tmp_path / "components" / "drafts"
    drafts.mkdir(parents=True)
    source = Path(PROJECT_ROOT) / "backend/arch/components/library/mha_attention.yaml"
    (drafts / "my_mha.yaml").write_text(source.read_text(encoding="utf-8"),
                                        encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "component_reuse", str(drafts / "my_mha.yaml"),
        "--overlay-dir", str(tmp_path)])
    main()

    printed = capsys.readouterr().out
    assert "my_mha — role=attention" in printed
    # The file to read, spelled as the reader can open it.
    assert "backend/arch/components/library/mha_attention.yaml" in printed
    assert "1.00  mha_attention" in printed


def test_a_document_that_does_not_load_as_a_component_is_a_data_error(tmp_path):
    # The candidate is parsed the way the loader parses a component, so what
    # cannot be compared is what cannot load — and it says so as a data
    # error, not as an AttributeError from inside the parser.
    bad = tmp_path / "bad.yaml"
    bad.write_text("graph: [1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        read_candidate(str(bad))


def test_a_graph_too_small_to_compare_says_so():
    tiny = _parse_block_graph({
        "inputs": ["x"], "outputs": ["w"],
        "graph": {"w": {"op": "linear", "in": ["x"],
                        "dims": {"in": "d_model", "out": "d_model"}}}})
    assert likenesses(tiny, {"mha": _graph(_ATTENTION)}, name="t") == []
    assert "Too small to compare" in render("t", tiny, [], lambda name: None)
