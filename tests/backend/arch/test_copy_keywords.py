"""A node's copies say how they are arranged: `chain:` or `repeat:`.

`chain: N` — copy i reads copy i−1 and a reader of the node sees the LAST.
`repeat: N` — every copy reads the node's `in:` and a reader sees ALL of them.

The two cost the same (N copies of a GEMM either way) and differ only in the
bytes on the edge leaving the node, so no measurement separated them while
both were spelled `repeat:` (`docs/design.md` §1.2). These tests
hold the three things that keep them apart: the count is read through one
accessor whichever keyword states it, a node may not claim both, and the
corpus's own sites stay on the side the census put them on — every remaining
`repeat:` read by a reducer or by its own bank, every `chain:` read as one
tensor.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import get_default_loader
from backend.arch.ops import OP_REGISTRY
from backend.arch.schema.graph import GraphNode
from tools.scoreboard import graph_node_sites


# --- the keywords -----------------------------------------------------------

def test_the_count_is_read_through_one_accessor():
    """`copies` is the count whichever keyword declares it, so a consumer of
    the count cannot be written in a way that misses a chain."""
    assert GraphNode(op="silu", inputs=["x"], repeat="n_experts").copies == "n_experts"
    assert GraphNode(op="blk", inputs=["x"], chain="n_vis_layers").copies == "n_vis_layers"
    assert GraphNode(op="silu", inputs=["x"]).copies is None


def test_a_node_may_not_claim_both_arrangements():
    with pytest.raises(ValueError, match="both `repeat:` and `chain:`"):
        GraphNode(op="blk", inputs=["x"], repeat="n_vis_layers",
                  chain="n_vis_layers")


def test_active_is_a_banks_word_only():
    """`active:` is the subset of a bank's copies a token runs; the copies of
    a chain all run, so the pair is refused."""
    with pytest.raises(ValueError, match="only valid with `repeat`"):
        GraphNode(op="blk", inputs=["x"], chain="n_vis_layers",
                  active="n_experts_active")


def test_an_arrangement_needs_copies_to_arrange():
    with pytest.raises(ValueError, match="only valid with `repeat`"):
        GraphNode(op="silu", inputs=["x"], active="n_experts_active")


# --- the engine reads the keyword -------------------------------------------

_SPEC = """
match: {{ model_type: [_copy_kw] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size }}
blocks:
  inner:
    graph:
      w: {{ op: linear, in: x, dims: {{ in: d_model, out: d_model }},
           weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
    outputs: [w]
  prod:
    graph:
      body: {{ op: inner, in: x, {keyword}: "2" }}
    outputs: [body]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids],
      boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }} }}
  - {{ name: layers, inputs: [embed], repeat: n_layers, template: prod,
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: {{ in: d_model, out: n_vocab }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def _analyze(keyword: str):
    """The same two-copy node spelled `repeat:` and spelled `chain:`."""
    import os
    import tempfile

    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.arch.loader import SpecLoader
    from backend.arch.spec_tree import iter_tree

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(_SPEC.format(keyword=keyword))
        path = f.name
    loader = SpecLoader()
    try:
        spec = loader.load_model_spec(path)
    finally:
        os.unlink(path)
    resolved = loader.resolve(spec, {
        "torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 3,
        "vocab_size": 16, "model_type": "_copy_kw"})
    tree = iter_tree(resolved)
    nodes = [n for _meta, roots in tree.blocks for n in roots
             if n.name == "body"]
    analysis = (Deployment(Program.of(resolved))
                .at(batch=1, input_tokens=16).analyze())
    return nodes, analysis


def test_a_chain_carries_the_copy_count_a_bank_always_had():
    """The count multiplies the node and everything under it whichever
    keyword declares it — a chain of 2 is 2 copies of work, as `repeat: 2`
    is. Without this the six towers would quietly lose their layer count."""
    chained, _ = _analyze("chain")
    banked, _ = _analyze("repeat")
    assert [n.repeat for n in chained] == [n.repeat for n in banked] == [2]
    assert [n.chained for n in chained] == [True]
    assert [n.chained for n in banked] == [False]


def test_the_keyword_is_carried_and_not_yet_priced():
    """`SpecIRNode.chained` is the declaration materialized for a consumer
    that asks what a reader of the node moves. No consumer prices on it yet,
    so the two spellings of the same two copies cost the same — the step
    that makes a bank's reader move every copy moves numbers and is a
    different change (`docs/known_limitations.md` §5.2)."""
    _, chained = _analyze("chain")
    _, banked = _analyze("repeat")
    assert chained.total().flops == banked.total().flops
    assert chained.total().params == banked.total().params


# --- the corpus's sites -----------------------------------------------------

#: One graph per authored name: the library, each spec's resolved components
#: (drafts included) and each spec's inline `blocks:`.
def _graphs():
    loader = get_default_loader()
    for name in sorted(loader.component_names()):
        body = loader.component(name)
        if body is not None and body.graph:
            yield name, body
    for spec_name in sorted(loader.all_specs()):
        spec = loader.get(spec_name)
        for table in (getattr(spec, "components", None) or {},
                      getattr(spec, "blocks", None) or {}):
            for name, body in table.items():
                if getattr(body, "graph", None):
                    yield f"{spec_name}/{name}", body


def _consumers(graph, node_name):
    """Every node of `graph` that reads `node_name` — by its own name, by a
    port of it, or (packed leaves) as the head of a dotted ref."""
    for name, node in graph.items():
        for ref in node.inputs or ():
            if (ref == node_name or ref.startswith(f"{node_name}.")
                    or ref.partition(".")[0] == node_name):
                yield name, node


def test_every_repeat_site_is_read_as_a_bank():
    """A `repeat:`ing node's readers are the shape a fan-in has: either the
    same bank (the `*_packed` leaves, where the count multiplies both sides
    and the factor is 1), or a node that reduces the copies it reads —
    declared as an op whose inputs include the bank's output and which is a
    dispatch boundary (`combine`) or has an identity for an absent operand
    (`add`, the shared-expert residual).

    A reader that is neither would be reading one tensor out of a bank, which
    is a chain wearing the wrong keyword — the defect this keyword pair
    exists to make visible.
    """
    offenders = []
    for where, body in _graphs():
        graph = body.graph
        for name, node in graph.items():
            if node.repeat is None:
                continue
            readers = list(_consumers(graph, name))
            if not readers:
                offenders.append(f"{where}/{name}: a bank nobody reads")
            for reader_name, reader in readers:
                if reader.copies == node.repeat:
                    continue                  # same bank, factor 1
                spec = OP_REGISTRY.get(reader.op)
                if spec is not None and (
                        spec.is_dispatch_boundary
                        or spec.absent_operand_identity is not None):
                    continue                  # a reduction over the copies
                offenders.append(
                    f"{where}/{name} declares `repeat: {node.repeat}` but "
                    f"{reader_name!r} ({reader.op}) reads it without being "
                    f"its bank or a reduction over it — if its forward reads "
                    f"only the last copy, this node is a `chain:`")
    assert offenders == []


def test_every_chain_site_hands_on_one_tensor():
    """A `chain:`ing node's reader is not another member of the same bank:
    the chain's copies collapse to one live tensor, so a reader carrying the
    same count would be claiming a per-copy pairing that does not exist."""
    offenders = []
    for where, body in _graphs():
        graph = body.graph
        for name, node in graph.items():
            if node.chain is None:
                continue
            for reader_name, reader in _consumers(graph, name):
                if reader.copies == node.chain:
                    offenders.append(
                        f"{where}/{name} chains {node.chain} copies, and "
                        f"{reader_name!r} carries the same count — a chain "
                        f"publishes its last copy, not one per reader copy")
    assert offenders == []


# --- the census -------------------------------------------------------------

#: The chains in the corpus: eight
#: vision towers (Qwen3.5's joined as one library component in 8f0aac44a), the audio encoder's transformer stack, and the fold over
#: the speech banks (`op: accumulate`, `reads_own_output`). MiMo-V2.6 inherits
#: the MiMo-V2.5 family (5c9216133), so the two audio nodes are declared once.
CHAIN_SITES = {
    "backend/arch/components/library/dsv41_vision_tower.yaml::vision.blocks",
    "backend/arch/components/library/hunyuan_vl_vision_tower.yaml::vit.layers",
    "backend/arch/components/library/kimi_k25_vision_tower.yaml::vision_tower.encoder.blocks",
    "backend/arch/components/library/kimi_k3_vision_tower.yaml::vision_tower.encoder.blocks",
    "backend/arch/components/library/minimax_m3_vision_tower.yaml::vision_tower.vision_model.encoder.layers",
    "backend/arch/components/library/qwen35_vision_tower.yaml::blocks",
    "backend/arch/components/library/step3p7_perception_encoder_tower.yaml::vision_model.transformer.resblocks",
    "backend/arch/components/drafts/perception_encoder_rope2d_tower.yaml::vision_model.transformer.resblocks",
    "backend/arch/components/library/mimo_v2_5_audio_encoder.yaml::audio_encoder.input_local_transformer.layers",
    "backend/arch/components/library/mimo_v2_5_audio_encoder.yaml::speech_bank_sum",
}


def test_the_chains_are_the_ones_the_census_named():
    """Nothing else converted along with them, and none of them went back:
    a site changing sides is a number moving, so it is this test's business
    to name them."""
    assert set(graph_node_sites("chain")) == CHAIN_SITES


def test_no_repeat_site_is_a_chain_site():
    """The scoreboard counts what is left (`repeat_sites`, target 0). The two
    keywords partition the copy sites — no node is in both lists."""
    repeats = set(graph_node_sites("repeat"))
    assert repeats & CHAIN_SITES == set()
    assert repeats, "the counter would read 0 for the wrong reason"
