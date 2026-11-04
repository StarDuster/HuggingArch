"""A graph's meaning does not depend on the order its nodes are written in.

`graph:` is a YAML mapping. A spec re-dumped with sorted keys — the form the
spec-generation bundle stored until it kept authored order — lists
`attn_residual` before `self_attn` and `mlp` before `post_attention_layernorm`;
it is the same forward pass and must evaluate to the same numbers.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import pytest
import yaml

from backend.arch.schema.graph import BlockGraph
from tests.snapshots import snapshot_config

LIB = Path(__file__).resolve().parents[3] / "backend" / "arch"
MIMO = "XiaomiMiMo/MiMo-7B-RL"


def _analysis(resolved):
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    return Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=64).analyze()


def test_a_sorted_redump_of_a_shipped_spec_evaluates_identically():
    from backend.arch.loader import SpecLoader
    from tests.snapshots import shared_loader
    config = snapshot_config(MIMO)
    stem = MIMO.replace("/", "__")
    doc = yaml.safe_load((LIB / "models" / f"{stem}.yaml").read_text())
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "models").mkdir()
        os.symlink(LIB / "components", base / "components")
        (base / "models" / f"{stem}.yaml").write_text(
            yaml.safe_dump(doc, sort_keys=True))
        # The inline block really is written consumer-first after the dump.
        sorted_graph = list(yaml.safe_load(
            (base / "models" / f"{stem}.yaml").read_text())
            ["blocks"]["mimo_mtp_block"]["graph"])
        assert sorted_graph.index("attn_residual") < sorted_graph.index("self_attn")
        redumped = SpecLoader(base_dir=str(base)).match(config, model_id=MIMO)
        shipped = shared_loader().match(config, model_id=MIMO)
        got, want = _analysis(redumped).total(), _analysis(shipped).total()
    assert (got.flops, got.params) == (want.flops, want.params)


def _graph(order):
    nodes = {
        "norm": {"op": "rmsnorm", "inputs": ["x"], "dim": "d_model"},
        "proj": {"op": "linear", "inputs": ["norm"],
                 "dims": {"in": "d_model", "out": "d_model"}},
        "res": {"op": "add", "inputs": ["x", "proj"], "dim": "d_model"},
    }
    return BlockGraph(graph={n: nodes[n] for n in order}, outputs=["res"])


def test_dependency_order_places_producers_first_and_keeps_written_order():
    assert _graph(["res", "proj", "norm"]).dependency_order() == [
        "norm", "proj", "res"]
    # A graph already written producer-first comes back unchanged.
    assert _graph(["norm", "proj", "res"]).dependency_order() == [
        "norm", "proj", "res"]


def test_a_port_reference_orders_after_its_producer():
    g = BlockGraph(inputs=["x", "scale", "base"], graph={
        "use": {"op": "mul", "inputs": ["x", "split.post"], "dim": "d"},
        "split": {"op": "hc_split_sinkhorn", "inputs": ["x", "scale", "base"],
                  "dims": {"d_hc_mix": "m", "n_hidden_copies": "n"}},
    }, outputs=["use"])
    assert g.dependency_order() == ["split", "use"]


def _collision(split_post_reads: str) -> BlockGraph:
    """A sibling named `split.post` beside `split`, whose op outputs `post`.
    By the `in:` grammar (`parse_input_ref`), `use`'s `split.post` is the
    port of `split`, not the node."""
    return BlockGraph(inputs=["x", "scale", "base"], graph={
        "use": {"op": "add", "inputs": ["split.post", "x"], "dim": "d"},
        "split.post": {"op": "rmsnorm", "inputs": [split_post_reads],
                       "dim": "d"},
        "split": {"op": "hc_split_sinkhorn", "inputs": ["x", "scale", "base"],
                  "dims": {"d_hc_mix": "m", "n_hidden_copies": "n"}},
    }, outputs=["use"])


def test_a_dotted_ref_is_read_by_the_input_grammar():
    """Codex's repros: the port reading puts `split` before `use`, and a
    node `split.post` reading `use` is a chain, not a cycle."""
    assert _collision("x").dependency_order() == ["split.post", "split", "use"]
    assert _collision("use").dependency_order() == ["split", "use", "split.post"]


def test_a_name_that_is_also_a_port_is_refused_as_ambiguous():
    from backend.arch.schema.graph import resolve_block_input_ports
    with pytest.raises(ValueError, match=re.escape(
            "names both the node 'split.post' "
            "and output 'post' of node 'split'")):
        resolve_block_input_ports(_collision("x"))


def test_a_cycle_is_refused_naming_only_its_nodes():
    g = BlockGraph(graph={
        "a": {"op": "add", "inputs": ["x", "b"], "dim": "d"},
        "b": {"op": "add", "inputs": ["x", "a"], "dim": "d"},
        "after": {"op": "rmsnorm", "inputs": ["a"], "dim": "d"},
    }, outputs=["after"])
    with pytest.raises(ValueError, match=r"graph nodes \['a', 'b'\] read each other"):
        g.dependency_order()
    from backend.arch.schema.graph import resolve_block_input_ports
    with pytest.raises(ValueError, match="cycle"):
        resolve_block_input_ports(g)


def test_a_ref_that_is_no_node_formal_or_stream_is_refused():
    """A name the graph does not declare has no producer to order after;
    it is refused rather than read as arriving from outside."""
    g = _graph(["norm", "proj", "res"])
    g.graph["res"].inputs = ["x", "projj"]
    with pytest.raises(ValueError, match=r"node 'res' reads 'projj'.*`inputs:` or `reads:`"):
        g.dependency_order()


def test_a_component_port_orders_after_its_node_before_resolution():
    """Before the loader resolves the graph, a port of a sibling whose op
    the registry does not know is still read from that sibling."""
    g = BlockGraph(graph={
        "use": {"op": "add", "inputs": ["x", "sub.aux"], "dim": "d"},
        "sub": {"op": "some_component", "inputs": ["x"]},
    }, outputs=["use"])
    assert g.dependency_order() == ["sub", "use"]
