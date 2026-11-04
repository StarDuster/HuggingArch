"""Three degenerate axes the layout walk used to refuse, each a legal tensor.

- A padded cut gathered back: a vocab-parallel head whose vocabulary does
  not divide by the world gathers the padded extent, and the padding is a
  local trim (vLLM `LogitsProcessor._get_logits`:
  `logits = logits[..., : self.org_vocab_size]` after `_gather_logits`).
  HunyuanOCR's 120818-row vocabulary at TP 8 gathered 120824 and raised.
- An empty concat operand: HunyuanOCR's PatchMerger appends one image_begin /
  image_end row per image and one newline per merged row, all zero when the
  walk holds one position (a decode step); the transfer table refused a
  zero extent.
- An encoder's own sequence under context parallelism: a broadcast whose
  width names `s_ctx` inside the vision tower took a CP cut, though the
  encoder view has no text sequence world; and the tower's concat of
  replicated pieces (a Segmented record) had no lowering to the text
  model's CP cut, though every rank holds every position.
"""
from __future__ import annotations

from tests.snapshots import resolve_snapshot, shared_loader

from backend.arch.layout import layout as L
from backend.arch.layout.groups import ProcessGroupFamily
from backend.arch.layout.lower import axis_collective
from backend.arch.layout.materialize_comm import PostTransform

HUNYUAN = "tencent/HunyuanOCR"


def _fam(name, *groups):
    return ProcessGroupFamily(id=name, groups=tuple(tuple(g) for g in groups))


class TestPaddedGather:
    def test_a_padded_cut_gathers_its_padding(self):
        cut = L.sharded(_fam("tp", range(8)), pad_to=1)
        assert cut.padded_global_extent(120818) == 120824
        assert cut.gathers_padding(120824, 120818)
        assert not cut.gathers_padding(120818, 120818)
        assert not cut.gathers_padding(120832, 120818)

    def test_an_unpadded_cut_never_gathers_padding(self):
        cut = L.sharded(_fam("tp", range(8)))
        assert not cut.gathers_padding(120824, 120818)

    def test_the_head_gathers_then_trims(self):
        from backend.arch.layout.materializer import materialize_comm_events
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.step_eval import evaluate_step
        analysis = evaluate_step(
            resolve_snapshot(HUNYUAN, shared_loader()),
            ParallelConfig(tp_attention=8, tp_dense_ffn=8),
            phase="decode", batch=1, input_tokens=128, output_tokens=0).analysis
        trims = [e for e in materialize_comm_events(analysis)
                 if e.post_transform == PostTransform.PADDING_TRIM]
        assert trims, "the vocab-parallel head's gather drops its padding"
        for event in trims:
            assert event.op == "all_gather"
            assert event.collective_out_shape[-1] == 120824
            assert event.flow_out_shape[-1] == 120818


class TestFullyReplicatedSegments:
    def test_a_replicated_segment_record_is_cut_for_free(self):
        cp = _fam("cp", range(2))
        whole = L.Replicated(scope=frozenset(range(2)))
        record = L.segmented(((0, 1, whole), (1, 32, whole), (33, 5, whole)))
        assert axis_collective(None, record, L.sharded(cp), None) is None


_BEGIN_END_TOWER = """\
match: { model_type: [_begin_end_tower], model_id: [_begin_end_tower] }
dims: { d_model: config.hidden_size, n_layers: "1" }
blocks:
  blk:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [n]
model_inputs:
  input_ids: { kind: tokens }
  pixel_values: { kind: image, axes: [n_img, img_h, img_w] }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: d_model, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - name: vis
    group: true
    aux_kind: vision
    layers:
      - { name: vis.patchify, op: patchify, inputs: [pixel_values],
          dims: { patch: "1", temporal: "1", channels: d_model, align: "1" },
          segment: item, source_anchor: "a test processor" }
      - { name: vis.proj, op: linear, inputs: [vis.patchify], dims: { in: d_model, out: d_model },
          weight_axes: [{ replicated: true }, { replicated: true }] }
      - { name: vis.begin, op: parameter_vector, dim: d_model, weight_axes: [{ replicated: true }] }
      - { name: vis.begin_row, op: broadcast, inputs: [vis.begin], axis: 0, dims: { width: "1" } }
      - { name: vis.begin_expand, op: broadcast, inputs: [vis.begin_row], axis: 0,
          dims: { width: "B * n_img" } }
      - { name: vis.out, op: concat, inputs: [vis.begin_expand, vis.proj], axis: 1 }
  - { name: merge, op: scatter_rows_replace, inputs: [embed, vis.out],
      prompt_slots: { domain: pixel_values, defines: true, source_anchor: "a test merge" } }
  - { name: layers, template: blk, repeat: n_layers, inputs: [merge],
      streams: { hidden: { carries: true } } }
  - { name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: { in: d_model, out: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


class TestAReplicatedConcatLeavesTheEncodeStepAsItIs:
    """An encoder whose output is a concat of replicated pieces (a learned
    begin row before its rows, as HunyuanOCR's merger writes) ends its
    encode step holding every position on every rank: the model-output
    endpoint makes no transition of it under TP."""

    def _encode(self, tp):
        import tempfile
        from pathlib import Path
        from backend.arch.eval.dim import value_of
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.loader import SpecLoader
        from backend.inference.parallel_config import ParallelConfig
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for sub in ("components/library", "components/adapters",
                        "components/drafts", "models"):
                (base / sub).mkdir(parents=True)
            (base / "models" / "_begin_end_tower.yaml").write_text(
                _BEGIN_END_TOWER)
            resolved = SpecLoader(base_dir=str(base)).match(
                {"model_type": "_begin_end_tower", "torch_dtype": "bfloat16",
                 "hidden_size": 8},
                model_id="_begin_end_tower")
        analysis = Deployment(
            Program.of(resolved),
            config=ParallelConfig(tp_attention=tp, tp_dense_ffn=tp),
        ).at(batch=1, media={"pixel_values": {
            "n_img": 1, "img_h": 4, "img_w": 4}}).during(
                "encode", stack="vis").analyze()
        (out,) = [n for n in analysis.program.nodes() if n.name == "vis.out"]
        from backend.arch.layout.stages import StagePlacement
        layout = StagePlacement.handed_state(analysis, (out.uid, "out")).layout
        # The concat's record: every segment whole on every rank.
        assert isinstance(layout[1], L.Segmented)
        assert L.axis_fully_replicated(layout[1])
        assert [value_of(e) for e in analysis.shapes.values[out.uid].out] \
            == [1, 17, 8]
        return analysis

    def test_no_model_output_transition(self):
        from backend.arch.eval.transition import RedistributeSite
        for tp in (2, 4):
            analysis = self._encode(tp)
            assert not [r for r in analysis.redistributes
                        if r.site == RedistributeSite.MODEL_OUTPUT], tp


class TestEncoderDegenerateExtents:
    def _analysis(self, phase, **parallel):
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        return analyze(resolve_snapshot(HUNYUAN, shared_loader()),
                       parallel=ParallelConfig(**parallel), batch_size=1,
                       avg_input_tokens=128, avg_output_tokens=0, phase=phase, breakdown=False)

    def test_a_decode_step_under_tp_has_empty_merger_rows(self):
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.step_eval import evaluate_step
        _ = evaluate_step(
            resolve_snapshot(HUNYUAN, shared_loader()),
            ParallelConfig(tp_attention=2, tp_dense_ffn=2),
            phase="decode", batch=1, input_tokens=128, output_tokens=0).analysis

    def test_context_parallelism_does_not_cut_the_merger_newlines(self):
        res = self._analysis("prefill", cp=2)
        states = res.analysis.states.values
        cut = [uid for uid in res.program.encoder_uids if uid in states
               and any(getattr(getattr(axis, "group", None), "id", None) == "cp"
                       for axis in states[uid].layout)]
        assert not cut
