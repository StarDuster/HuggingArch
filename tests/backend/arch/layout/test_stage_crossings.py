"""What crosses a pipeline cut, and how it is priced: toys T1-T5 and the
corpus cases.

A hop (layer 1) is one `send_recv` of a value across one cut, sent once
however many sinks read it past that cut, in the state the value is handed on
in: each rank's local tensor, whole (`close.stage_crossing_records`). Each
sink's own input transition is the record the walk already made for it
(layer 2), on the sink's stage. A stage pays the hops across its exit cut.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from math import prod
from pathlib import Path

import pytest

from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.eval.occurrence import Occurrence
from backend.arch.layout.close import stage_crossing_records
from backend.arch.layout.lower import lower_diff
from backend.arch.layout.materializer import materialize_comm_events
from backend.arch.layout.parallel_config import ParallelConfig
from backend.arch.layout.stages import StagePlacement
from backend.arch.loader import SpecLoader
from tests.snapshots import resolve_snapshot, shared_loader, shared_walk

# Four layers: layer 0 runs `writer`, which publishes a row-parallel
# projection (a pending sum) as the inner stream `pstream`; layers 1-3 run
# `reader`, which adds it. After the stack, `col` is column-parallel (its
# output is cut over tp), and two model-level adds read the embedding: one
# beside the replicated hidden state, one beside `col`.
SPEC = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_layers: "4"
  n_heads: "4"
  d_head: "2"
blocks:
  writer:
    stream_out: {{ pstream: rp }}
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      rp: {{ op: linear, in: n, dims: {{ in: "n_heads * d_head", out: d_model }},
             weight_axes: [{{ replicated: true }}, {{ world: attention, parts: n_heads, fit: exact }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
  reader:
    reads: [pstream]
    graph:
      u: {{ op: add, in: [x, pstream], dim: d_model }}
    outputs: [u]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: layers
    repeat: n_layers
    inputs: [embed]
    streams:
      hidden: {{ carries: true }}
      pstream: {{ shape: [B, s_ctx, d_model], src: "toy" }}
    per_layer:
      assign:
        - {{ type: w, layers: ["0"] }}
        - {{ type: r, rest: true, reads_from: {{ pstream: "0" }} }}
      map:
        w: {{ template: writer }}
        r: {{ template: reader }}
  - {{ name: col, op: linear, inputs: [layers], dims: {{ in: d_model, out: "n_heads * d_head" }},
      weight_axes: [{{ world: attention, parts: n_heads, fit: exact }}, {{ replicated: true }}] }}
  - {{ name: side_r, op: add, inputs: [layers, embed], dim: d_model }}
  - {{ name: side_b, op: add, inputs: [col, embed], dim: d_model }}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [side_r],
      dims: {{ in: d_model, out: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def _toy(model_type="xs_toy"):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{model_type}.yaml").write_text(
            SPEC.format(model_type=model_type))
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": model_type, "torch_dtype": "bfloat16", "hidden_size": 8},
            model_id=model_type)


def _analyze(program, **parallel):
    deployment = Deployment(program).at(batch=1, input_tokens=4)
    return shared_walk(replace(deployment, config=ParallelConfig(**parallel)))


def _walked(pp, tp=4):
    analysis = _analyze(Program.of(_toy()), pp=pp, tp_attention=tp)
    return analysis, StagePlacement.of(analysis, pp)


def _named(analysis, name):
    (node,) = [n for n in analysis.program.nodes() if n.name == name]
    return node


def _hops(placement, analysis, name):
    uid = _named(analysis, name).uid
    return [c for c in placement.crossings if c.producer[0] == uid]


def test_t1_one_send_serves_a_fan_out():
    """The embedding is read past cut 0 by `side_r`, which needs it
    replicated as it arrives, and by `side_b`, which needs it cut over tp
    beside `col`: one send, nothing for the first sink, a free slice for the
    second."""
    analysis, placement = _walked(pp=2)
    (hop,) = _hops(placement, analysis, "embed")
    assert hop.cut == 0
    side_r, side_b = _named(analysis, "side_r"), _named(analysis, "side_b")
    assert sorted((s.uid, i) for s, i in hop.sinks) == [
        (side_r.uid, 1), (side_b.uid, 1)]
    handed = placement.handed_state(analysis, hop.producer)
    assert analysis.input_states[(side_r.uid, 1)] == handed
    required = analysis.input_states[(side_b.uid, 1)]
    assert required != handed
    assert lower_diff(analysis.rule, handed, required) == ()
    assert dict((s.uid, r) for s, r in placement.repairs(analysis, hop)) == {
        side_r.uid: (), side_b.uid: ()}


def test_t2_a_pending_sum_crosses_as_it_is_and_reduces_where_it_is_read():
    """A row-parallel output read past a cut is sent as each rank's local
    partial (pp reduces nothing); the `all_reduce` stays the reader's input
    transition, on the destination stage, with the walk's count at
    pp = 1."""
    analysis, placement = _walked(pp=2)
    (hop,) = _hops(placement, analysis, "rp")
    handed = placement.handed_state(analysis, hop.producer)
    assert handed.value is not None and handed.value.reducer == "sum"
    reader = _named(analysis, "u")
    repairs = dict(((s.uid, s.iteration), r)
                   for s, r in placement.repairs(analysis, hop))
    assert set(repairs) == {(reader.uid, 2), (reader.uid, 3)}
    for records in repairs.values():
        assert [r.collectives for r in records] == [
            (("tp_attention", "all_reduce"),)]
    assert {placement.stage(s) for s, _ in hop.sinks} == {1}

    def reductions(a):
        return [(r.node, r.input_index, r.collectives)
                for r in a.redistributes if r.crossing is None]
    single = _analyze(Program.of(_toy()), tp_attention=4)
    assert reductions(analysis) == reductions(single)


def test_t3_a_hop_is_a_plain_send_then_each_sink_repairs():
    """At tp 4 the hop sends each rank's local tensor as the walk hands it
    on: one `send_recv` row on the sending stage, a quarter of nothing
    sliced away and nothing restored on the receiving stage. Each sink then
    makes its own transition from the handed state, there."""
    analysis, placement = _walked(pp=2)
    (hop,) = _hops(placement, analysis, "embed")
    sends = [e for e in materialize_comm_events(analysis)
             if e.op == "send_recv"
             and e.source.node_uid == hop.producer[0]]
    assert [dict(e.meta)["cut"] for e in sends] == [0]
    extents = [int(value_of(x)) for x in
               analysis.shapes.values[hop.producer[0]].out]
    assert sends[0].payload_shape == tuple(extents)
    on_receiver = materialize_comm_events(
        analysis, stage=placement.view(1))
    assert not [e for e in on_receiver if e.op == "send_recv"]
    handed = placement.handed_state(analysis, hop.producer)
    side_b = _named(analysis, "side_b")
    assert lower_diff(analysis.rule, handed,
                      analysis.input_states[(side_b.uid, 1)]) == ()


def test_t4_a_value_read_on_several_later_stages_is_relayed_once_per_cut():
    """Written on stage 0 (layer 0), read by layers 1-3 on stages 1 and 2:
    one send on cut 0 and one on cut 1, never two on one cut."""
    analysis, placement = _walked(pp=3)
    hops = _hops(placement, analysis, "rp")
    assert sorted(h.cut for h in hops) == [0, 1]
    by_cut = {h.cut: sorted(s.iteration for s, _ in h.sinks) for h in hops}
    assert by_cut == {0: [1, 2, 3], 1: [3]}


def test_t5_no_stage_relays_what_no_stage_needs_yet():
    """Four layers on eight stages leave stages 0-2 and 7 empty: the
    embedding is pulled to layer 0's stage and the model-level ops follow
    layer 3, so no hop crosses cuts 0-2 or 6."""
    analysis, placement = _walked(pp=8, tp=1)
    assert [len(r) for r in placement.ranges] == [0, 0, 0, 1, 1, 1, 1, 0]
    cuts = {c.cut for c in placement.crossings}
    assert cuts == {3, 4, 5}
    assert placement.stage(Occurrence(_named(analysis, "embed").uid,
                                      None)) == 3
    assert placement.stage(Occurrence(_named(analysis, "head").uid,
                                      None)) == 6



def test_t5_a_corpus_model_on_more_stages_than_layers():
    """Llama-3.2-1B, 16 layers at pp = 32: layers 0-15 sit on stages 15-30.
    The embedding runs on stage 15, the norm and the head on stage 30, and
    stages 0-14 and 31 send nothing."""
    program = Program.of(resolve_snapshot("meta-llama/Llama-3.2-1B",
                                          shared_loader()))
    analysis = _analyze(program, pp=32)
    placement = StagePlacement.of(analysis, 32)
    by_name = {n.name: n.uid for n in analysis.program.nodes()}
    assert placement.stage(Occurrence(by_name["embed_tokens"], None)) == 15
    assert placement.stage(Occurrence(by_name["lm_head"], None)) == 30
    cuts = {r.cut
            for r in stage_crossing_records(analysis.rule, analysis)}
    assert cuts == set(range(15, 30))


@pytest.mark.parametrize("timed, cuts", [(False, []), (True, [0])])
def test_an_mtp_head_reading_the_embedding_crosses_only_when_timed(timed,
                                                                   cuts):
    """DeepSeek-V4's MTP reads `('layers', 'embed')`: placed after its
    latest input, on the last stage, it reads the embedding across cut 0.
    That hop serves the MTP stack, so it is in a timing scope only when the
    MTP is timed."""
    program = Program.of(resolve_snapshot("deepseek-ai/DeepSeek-V4-Flash",
                                          shared_loader()))
    analysis = _analyze(program, pp=2, tp_attention=4)
    placement = StagePlacement.of(analysis, 2)
    scope = placement.backbone | ({"mtp"} if timed else set())
    embed = {n.uid for n in analysis.program.nodes() if n.name == "embed"}
    assert [c.cut for c in placement.crossings
            if c.producer[0] in embed and c.serves & scope] == cuts
    priced = [dict(e.meta)["cut"]
              for e in materialize_comm_events(analysis, timing_scope=scope)
              if e.op == "send_recv" and e.source.node_uid in embed]
    assert priced == cuts


def test_an_encoder_element_read_past_a_cut_is_handed_over_not_relayed():
    """Qwen3-VL at pp = 32: layers 0-1 on stage 0, and layer 2 adds a
    deepstack element. The element is the vision encoder's, handed over
    from the encode phase to the stage that reads it: no stage of the
    language model writes it, so no cut relays it. Cut 0 carries the hidden
    state alone."""
    program = Program.of(resolve_snapshot("Qwen/Qwen3-VL-235B-A22B-Instruct",
                                          shared_loader()))
    analysis = _analyze(program, pp=32)
    placement = StagePlacement.of(analysis, 32)
    assert [producer for producer, _state
            in analysis.appended["deepstack_feat"]] == [None] * 3
    (hop,) = [c for c in placement.crossings if c.cut == 0]
    assert (hop.source_iteration, hop.serves & placement.backbone) == (
        1, placement.backbone)
    sends = [(dict(e.meta)["cut"], e.source.node_uid)
             for e in materialize_comm_events(analysis)
             if e.op == "send_recv" and dict(e.meta)["cut"] == 0]
    assert sends == [(0, hop.producer[0])]


# -- the hidden state and the corpus' streams, as rows --------------------

# ph_toy: [B, S, D] = [1, 4, 8] at BF16, eight backbone layers.
WHOLE = 1 * 4 * 8 * 2


def _toy_analysis(**parallel):
    from tests.backend.arch.test_appended_streams import _resolved
    # The appended stream's readers (layers 0 and 1) sit on stage 0 up to
    # pp = 4: only the hidden state crosses a cut.
    return _analyze(Program.of(_resolved("ph_toy", n_layers=8)), **parallel)


def _cut_events(analysis, **kw):
    """The rows a pipeline cut prices, as (op, cut, event)."""
    return [(e.op, dict(e.meta)["cut"], e)
            for e in materialize_comm_events(analysis, **kw)
            if "cut" in dict(e.meta)]


@pytest.mark.parametrize("pp", [2, 4])
def test_the_hidden_state_crosses_every_cut_whole(pp):
    events = _cut_events(_toy_analysis(pp=pp))
    assert [(op, cut) for op, cut, _ in events] == [
        ("send_recv", cut) for cut in range(pp - 1)]
    for _op, _cut, send in events:
        assert send.payload_shape == (1, 4, 8)
        assert (send.count, send.world, send.group_size) == (1, "pp", pp)
        assert send.bytes_per_rank == WHOLE


@pytest.mark.parametrize("tp, fabric", [(4, "intra"), (8, "inter")])
def test_a_cut_leaves_the_node_when_one_stage_fills_it(tp, fabric):
    """A `pp` group holds one rank per stage, `per_stage_size` ranks apart
    on the mesh (`resolver.pipeline_family_for`), and the send is priced on
    that spacing like any other world's collective. On 8-GPU nodes pp2 x tp4
    keeps both stages on one node; pp2 x tp8 puts each stage on its own node,
    so the hop crosses the inter-node fabric."""
    from backend.inference.comm_cost import collective_phases

    ((_, _, send),) = _cut_events(_toy_analysis(pp=2, tp_attention=tp))
    assert send.base_stride == tp
    (hop,) = collective_phases("send_recv", send.group_size,
                               send.payload_bytes, gpus_per_node=8,
                               base_stride=send.base_stride)
    assert hop["fabric"] == fabric


def test_a_value_already_cut_over_the_world_is_sent_as_it_is():
    """Under sequence parallelism the residual stream is already cut over the
    attention TP world: its local shard is sent."""
    events = _cut_events(
        _toy_analysis(pp=2, tp_attention=4, sequence_parallel=True))
    assert [(op, e.bytes_per_rank) for op, _, e in events] == [
        ("send_recv", WHOLE // 4)]


def test_each_stage_pays_its_own_exit_cut():
    analysis = _toy_analysis(pp=2, tp_attention=4)
    placement = StagePlacement.of(analysis)
    assert [op for op, _, _ in _cut_events(
        analysis, stage=placement.view(0))] == ["send_recv"]
    assert _cut_events(analysis, stage=placement.view(1)) == []


def _topk_rows(model_id, pp):
    """`(op, cut, width, bytes per rank, element count)` of the rows that
    move the DSA top-k across a cut."""
    analysis = _analyze(Program.of(resolve_snapshot(model_id, shared_loader())),
                        pp=pp, tp_attention=4)
    return sorted(
        (e.op, dict(e.meta)["cut"], e.payload_dtype_bytes,
         e.bytes_per_rank, prod(e.payload_shape))
        for e in materialize_comm_events(analysis)
        if "cut" in dict(e.meta)
        and analysis.program.by_uid(e.source.node_uid).op == "dsa_topk_mask")


def test_an_inner_stream_crosses_a_cut_only_before_its_next_reader():
    """GLM-5.2's shared-indexer layers read the `dsa_topk` a full layer
    wrote. Its 78 layers are full at 0-2 and 6, 10, ..., 74. At pp = 3 the
    cut after layer 25 is followed by full layer 26 (its source is not
    before the cut: nothing crosses) and the cut after 51 by shared layer 52
    (the top-k crosses, int32, whole)."""
    (send,) = _topk_rows("zai-org/GLM-5.2", 3)
    op, cut, width, per_rank, numel = send
    assert (op, cut, width) == ("send_recv", 1, 4.0)
    assert per_rank == numel * 4


def test_an_inner_stream_no_later_layer_reads_crosses_nothing():
    """GLM-5.1 computes its top-k on every layer: no reader, no transfer."""
    assert _topk_rows("zai-org/GLM-5.1", 2) == []


def test_an_encoder_element_is_no_row_of_any_stage():
    """Qwen3-VL at pp = 32: the deepstack element layer 2 adds is handed
    over from the encode phase, so no stage's per-op rows price a send of
    it: stage 0 sends the hidden state (iteration 1 of the decoder) across
    cut 0 and nothing else, stage 1 the hidden state across cut 1."""
    from backend.inference import pipeline
    from backend.inference.parallel_config import ParallelConfig as Serving

    resolved = resolve_snapshot("Qwen/Qwen3-VL-235B-A22B-Instruct",
                                shared_loader())

    def sends(stage):
        rows = pipeline.analyze(resolved, parallel=Serving(pp=32),
                                batch_size=1, avg_input_tokens=64, avg_output_tokens=0,
                                stage=stage, breakdown=False).per_op_timing
        return sorted((r["source_occurrence"]["iteration"], r["cut"])
                      for r in rows if r.get("comm_op") == "send_recv")

    assert sends(0) == [(1, 0)]
    assert sends(1) == [(4, 1)]
