"""Where a block publishes the inner streams it writes, drawn in its graph.

A block reads a stream from a `stream_in` box; it writes one into a
`stream_out` box fed by the top-level box the writer sits in, and the
writer's own box says `stream_writes` — so a stream named on the
architecture overview (`stream_map`) can be found in the block, where it is
made (`Program.stream_writers`).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.eval.program import Program
from backend.rendering import build_block_graph_data

_SNAP = PROJECT_ROOT / "backend" / "arch" / "snapshots" / "deepseek-ai__DeepSeek-V4.1-Flash.json"


def _walk(nodes):
    for node in nodes:
        yield node
        yield from _walk(node.get("children") or ())


def test_every_written_stream_leaves_its_block_from_its_writer():
    resolved = shared_loader().match(json.loads(_SNAP.read_text())["config"],
                                     model_id="deepseek-ai/DeepSeek-V4.1-Flash")
    program = Program.of(resolved)
    graphs, _ = build_block_graph_data(resolved, batch_size=1, seq_len=1)
    checked = 0
    # A block graph is drawn per template name, from its first invocation
    # (two invocations may bind it at different dims: DSv4.1's borrowing
    # CSA layers at compress ratio 2 and 1).
    drawn = {}
    for template in program.blocks:
        drawn.setdefault(template.meta.template_name, template)
    for template in drawn.values():
        writers = program.stream_writers(template)
        graph = graphs[template.meta.template_name]
        exits = {n["id"]: n for n in graph["nodes"] if n.get("op") == "stream_out"}
        assert sorted(exits) == sorted(f"{name}>" for name in writers)
        marked = {name: node["id"] for node in _walk(graph["nodes"])
                  for name in node.get("stream_writes") or ()}
        for name, nodes in writers.items():
            assert marked[name] in {node.aggregation_id for node in nodes}
            fed = [e for e in graph["edges"] if e["dst"] == f"{name}>"]
            assert len(fed) == len(nodes), (template.name, name)
            checked += 1
    # Layer 20's block alone writes five: index_k, candidates, csa_topk,
    # compress_kv, hc_pre_mix.
    assert checked >= 5


def _program(model: str):
    snap = PROJECT_ROOT / "backend" / "arch" / "snapshots" / f"{model.replace('/', '__')}.json"
    return Program.of(shared_loader().match(json.loads(snap.read_text())["config"],
                                            model_id=model))


def test_a_formal_passed_on_is_read_only_where_the_callee_consumes_it():
    """DeepSeek-V4's block hands `input_ids` to whichever FFN its slot binds:
    the hash-routed MoE reads it, the score-routed one declares it and
    ignores it, so only the hash-routed layers read the stack's
    `input_ids` (`BlockTemplate.reads_formal` follows the call into the
    bound component)."""
    program = _program("deepseek-ai/DeepSeek-V4-Flash")
    (layers,) = [e for e in program.ir.spine if e.name == "layers"]
    readers = {t.meta.template_name.rsplit("ffn-", 1)[-1]: t.reads_formal("input_ids")
               for t in program.blocks
               if tuple(t.meta.layer_path) == tuple(layers.layer_path)}
    assert readers == {"dsv4_moe_hash": True, "dsv4_moe_sigmoid": False}


def test_a_read_of_another_stacks_layer_says_whose(monkeypatch):
    """A layer whose inner-stream read resolves to another stack's layer
    gets that stack named, not a bare index the reader's own stack would
    misread as one of its layers. No snapshot reads across stacks today, so
    one read is planted on DeepSeek-V4.1."""
    program = _program("deepseek-ai/DeepSeek-V4.1-Flash")
    (layers,) = [e for e in program.ir.spine if e.name == "layers"]
    (mtp,) = [e for e in program.ir.spine if e.name == "mtp"]
    owner = tuple(layers.layer_path)
    other = tuple(mtp.children[0].layer_path)
    monkeypatch.setitem(program.ir.stream_reads, (owner, 5, "planted"), (other, 1))
    assert program.inner_stream_layers(owner)["planted"][5] == (False, "mtp layer 1")
