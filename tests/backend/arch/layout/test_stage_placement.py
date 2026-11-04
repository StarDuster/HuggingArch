"""Where every occurrence runs at `pp` stages (PP B1, shadow).

The backbone's iteration space is cut by the canonical partition; every other
unit is placed by its edges: pulled to its earliest consumer when all its
inputs are free or pulled, else after its latest input. Which stacks are the
backbone is the declared `Stack.is_aux`, never the kind a stack is named as.
"""
from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from backend.arch.eval.occurrence import Occurrence
from backend.arch.eval.program import BlockStage, Program
from backend.arch.eval.transforms import Deployment, Workload
from backend.arch.layout.parallel_config import ParallelConfig
from backend.arch.layout.stages import StagePlacement
from backend.arch.loader import SpecLoader, SpecLoadError
from tests.snapshots import resolve_snapshot, shared_loader, shared_walk

# Two repeated backbone layers (`enc` then `dec`), a vision tower whose
# output the embedding merge reads (as a handoff: the tower runs in the
# encode phase, not on the language model's stages), a side branch off the embedding that
# rejoins after the norm, an output endpoint reading only the embedding
# (`probe`), an unconsumed buffer (`spare`), and an MTP-like head
# reading the norm.
STRUCT = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_layers: "3"
  n_vis_layers: "2"
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
model_inputs:
  input_ids: {{ kind: tokens }}
  features: {{ kind: image, axes: [n_img, img_h, img_w] }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: vis
    group: true
    aux_kind: {vision}
    checkpoint: {{ prefix: vis.blocks, index: local }}
    layers:
      - {{ name: vis.patchify, op: patchify, inputs: [features],
          dims: {{ patch: "1", temporal: "1", channels: d_model, align: "1" }},
          segment: item, source_anchor: "a test processor" }}
      - {{ name: vis.patch, op: linear, inputs: [vis.patchify], dims: {{ in: d_model, out: d_model }},
          weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      - name: vis.blocks
        template: blk
        repeat: n_vis_layers
        inputs: [vis.patch]
        streams:
          hidden: {{ carries: true }}
  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, vis.blocks],
      prompt_slots: {{ domain: features, defines: true, source_anchor: "a test merge" }} }}
  - name: enc
    repeat: n_layers
    inputs: [merge]
    streams:
      hidden: {{ carries: true }}
    template: blk
  - name: dec
    repeat: n_layers
    inputs: [enc]
    streams:
      hidden: {{ carries: true }}
    template: blk
  - {{ name: norm, op: rmsnorm, dim: d_model, inputs: [dec],
      weight_axes: [{{ replicated: true }}] }}
  - {{ name: side, op: rmsnorm, dim: d_model, inputs: [embed],
      weight_axes: [{{ replicated: true }}] }}
  - {{ name: join, op: add, inputs: [norm, side], dim: d_model }}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [join],
      dims: {{ in: d_model, out: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: probe, op: linear, boundary_role: output_head, inputs: [embed],
      dims: {{ in: d_model, out: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: spare, op: rotary_inv_freq_buffer, dim: "d_model // 2",
      weight_axes: [{{ replicated: true }}] }}
  - name: mtp
    group: true
    aux_kind: {mtp}
    inputs: [norm]
    checkpoint: {{ prefix: mtp.blocks, index: local }}
    layers:
      - name: mtp.blocks
        template: blk
        repeat: "1"
        inputs: [norm]
        streams:
          hidden: {{ carries: true }}
      - {{ name: mtp_logits, op: linear, boundary_role: output_head,
          inputs: [mtp.blocks], dims: {{ in: d_model, out: d_model }},
          weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def _load(text, model_type):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{model_type}.yaml").write_text(text)
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": model_type, "torch_dtype": "bfloat16", "hidden_size": 8},
            model_id=model_type)


def _struct(model_type="sp_struct", vision="vision", mtp="mtp"):
    return Program.of(_load(STRUCT.format(model_type=model_type,
                                          vision=vision, mtp=mtp),
                            model_type))


def _walked(program):
    return shared_walk(Deployment(program).at(batch=1, input_tokens=16))


def _placed(program, pp):
    """The placement at `pp` stages of the edges one walk of `program`
    bound."""
    return StagePlacement.of(_walked(program), pp)


def _exit(stage):
    """The node a stage hands on: a block's last declared output root."""
    return stage.template.output_roots()[-1]


def _uid(program, name):
    (node,) = [n for n in program.nodes() if n.name == name]
    return node.uid


def _stage_of(placement, name):
    program = placement.program
    uid = _uid(program, name)
    stage, _root = program.evaluating_stage(uid)
    return placement.stage(Occurrence(uid, program.stage_iterations(stage)[0]))


def _iteration_stages(placement, layer):
    stage = next(s for stack in placement.program.stacks for s in stack.stages
                 if isinstance(s, BlockStage) and s.name == layer)
    exit_uid = _exit(stage).uid
    return [placement.stage(Occurrence(exit_uid, i))
            for i in stage.template.layer_indices]


def test_two_backbone_stacks_are_one_iteration_space():
    """`enc` and `dec` are concatenated in walk order and cut as six
    iterations: at pp = 2 the cut falls between them, and `dec` reads `enc`'s
    last exit across it."""
    placement = _placed(_struct(), 2)
    assert placement.n_iterations == 6
    assert _iteration_stages(placement, "enc") == [0, 0, 0]
    assert _iteration_stages(placement, "dec") == [1, 1, 1]
    program = placement.program
    enc = next(s for stack in program.stacks for s in stack.stages
               if s.name == "enc")
    embed = _uid(program, "embed")
    backbone = [c for c in placement.crossings
                if c.serves & placement.backbone
                and c.producer[0] != embed]
    assert [(c.cut, c.producer[0], c.source_iteration) for c in backbone] == [
        (0, _exit(enc).uid, 2)]


def test_a_side_branch_runs_where_it_rejoins_and_reads_the_embedding_across():
    """`side` reads only the embedding, so it is pulled to its one consumer,
    `join`, which runs after the norm on the last stage. The embedding stays
    with layer 0 and its value is shipped to `side`, not recomputed."""
    placement = _placed(_struct(), 2)
    assert _stage_of(placement, "side") == 1
    assert _stage_of(placement, "embed") == 0
    embed = _uid(placement.program, "embed")
    assert [c.cut for c in placement.crossings if c.producer[0] == embed] == [0]


def test_the_merge_reading_an_encoder_is_pulled_to_its_first_reader():
    """Six iterations on eight stages leave stage 0 empty: the merge and the
    embedding are pulled to layer 0's stage, not stage 0, because their
    inputs are free — the encoder's output is a handoff from the encode
    phase, like a model input — and their consumer is there. The encoder is
    no unit of the language model's stages: nothing places it."""
    placement = _placed(_struct(), 8)
    assert [len(r) for r in placement.ranges] == [0, 1, 1, 1, 1, 1, 1, 0]
    assert _stage_of(placement, "merge") == 1
    assert _stage_of(placement, "embed") == 1
    assert ("stack", "vis") not in placement.pulled
    with pytest.raises(ValueError, match="no node a stage evaluates"):
        _stage_of(placement, "vis.patch")


def test_an_unconsumed_free_node_runs_first_and_a_free_output_runs_last():
    """`spare` (a buffer, like Llama-2's `rotary_inv_freq`) reads nothing
    and nothing reads it: it is billed once, on stage 0. `probe` reads only
    the embedding but is a model output, where logits are sampled: the last
    stage."""
    placement = _placed(_struct(), 4)
    assert _stage_of(placement, "spare") == 0
    assert _stage_of(placement, "probe") == 3


def test_a_head_stack_runs_after_its_latest_input():
    placement = _placed(_struct(), 4)
    assert _stage_of(placement, "mtp_logits") == 3
    assert ("stack", "mtp") not in placement.pulled


def test_non_contiguous_layer_indices_are_placed_per_iteration():
    """gpt-oss alternates two templates (even and odd layers). At pp = 2
    each iteration is placed by the partition, and the cut hands on the exit
    of iteration 11, the odd template's."""
    placement = _placed(Program.of(resolve_snapshot("openai/gpt-oss-20b",
                                                    shared_loader())), 2)
    program = placement.program
    stages = [s for stack in program.stacks for s in stack.stages
              if isinstance(s, BlockStage) and not stack.is_aux]
    for stage in stages:
        exit_uid = _exit(stage).uid
        for i in stage.template.layer_indices:
            assert placement.stage(Occurrence(exit_uid, i)) == (i >= 12)
    (hop,) = [c for c in placement.crossings
              if c.serves & placement.backbone]
    odd = next(s for s in stages if 11 in s.template.layer_indices)
    assert (hop.cut, hop.producer[0], hop.source_iteration) == (
        0, _exit(odd).uid, 11)


CYCLE = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_layers: "2"
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      o: {{ op: add, in: [x, n], dim: d_model }}
    outputs: [o]
model_inputs:
  input_ids: {{ kind: tokens }}
  pixel_values: {{ kind: image, axes: [n_img, img_h, img_w] }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: vis
    group: true
    aux_kind: vision
    layers:
      - {{ name: vis.patchify, op: patchify, inputs: [pixel_values],
          dims: {{ patch: "1", temporal: "1", channels: d_model, align: "1" }},
          segment: item, source_anchor: "a test processor" }}
      - {{ name: vis.a, op: linear, inputs: [vis.patchify], dims: {{ in: d_model, out: d_model }},
          weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      - {{ name: vis.b, op: rmsnorm, dim: d_model, inputs: [layers],
          weight_axes: [{{ replicated: true }}] }}
  - {{ name: merge, op: scatter_rows_replace, inputs: [embed, vis.a],
      prompt_slots: {{ domain: pixel_values, defines: true, source_anchor: "a test merge" }} }}
  - name: layers
    repeat: n_layers
    inputs: [merge]
    streams:
      hidden: {{ carries: true }}
    template: blk
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: {{ in: d_model, out: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def test_a_unit_that_feeds_the_loop_and_reads_it_is_refused_at_load():
    """The tower is one unit: it would feed layer 0 (through the merge) and
    read the stack's output. No stage order runs it whole. A stage may name
    only layers declared before it, so the spec is refused when it is
    loaded, at any pp (`StagePlacement` keeps a cycle check behind that)."""
    with pytest.raises(SpecLoadError, match="references 'layers'"):
        _load(CYCLE.format(model_type="sp_cycle"), "sp_cycle")


def _snapshot(placement):
    """Every occurrence's stage and every hop, by uid: comparable across two
    loads of one structure."""
    program = placement.program
    stages = {(uid, i): placement.stage(Occurrence(uid, i))
              for uid, iterations in program.occurrence_domains()
              for i in iterations}
    hops = [(c.cut, c.producer, c.source_iteration, c.sinks)
            for c in placement.crossings]
    return stages, hops, placement.pulled


@pytest.mark.parametrize("pp", [1, 2, 4, 8])
def test_renaming_an_aux_kind_cannot_move_placement(pp):
    """`Stack.is_aux` is declared by the `group:` that opens a stack;
    placement reads that boolean and no kind string. Declaring the tower
    `audio` instead of `vision`, the head `speculative` instead of `mtp`, or
    renaming both to strings outside the vocabulary leaves every occurrence
    and every hop where it was."""
    reference = _snapshot(_placed(_struct("sp_kind_a"), pp))
    swapped = _struct("sp_kind_a", vision="audio", mtp="speculative")
    assert _snapshot(_placed(swapped, pp)) == reference
    renamed = _load(STRUCT.format(model_type="sp_kind_a", vision="vision",
                                  mtp="mtp"), "sp_kind_a")
    for layer in renamed.spec.model:
        if layer.group:
            object.__setattr__(layer, "aux_kind", f"renamed_{layer.name}")
    program = Program.of(renamed)
    assert {s.aux_kind for s in program.stacks if s.is_aux} == {
        "renamed_vis", "renamed_mtp"}
    assert _snapshot(_placed(program, pp)) == reference


def _program(model_id):
    return Program.of(resolve_snapshot(model_id, shared_loader()))


@pytest.mark.parametrize("model_id, pp", [
    ("deepseek-ai/DeepSeek-V3", 4),
    ("Qwen/Qwen3-VL-235B-A22B-Instruct", 32),
    ("moonshotai/Kimi-K3", 4),
    ("deepseek-ai/DeepSeek-V4-Flash", 2),
])
def test_rows_and_cache_are_conserved_within_each_scope(model_id, pp):
    """Summed over stages, each scope's rows and cache count every node of
    that scope exactly as pp = 1 does."""
    placement = _placed(_program(model_id), pp)
    stacks = frozenset(s.name for s in placement.program.stacks)
    for scope in (placement.backbone, stacks, stacks - placement.backbone):
        whole = StagePlacement(placement.graph, 1).view(0)
        for project in ("rows", "cache"):
            total = Counter()
            for s in range(pp):
                total.update(getattr(placement.view(s), project)(scope).values)
            expected = getattr(whole, project)(scope).values
            assert {u: n for u, n in total.items() if n} == {
                u: n for u, n in expected.items() if n}


@pytest.mark.parametrize("model_id, shared", [
    ("Qwen/Qwen3-8B", False),
    ("meta-llama/Llama-3.2-3B", True),
])
def test_weights_count_at_least_once_and_twice_only_when_read_twice(
        model_id, shared):
    """A stage holds what runs on it and the tables it reads. Summed over
    stages that is the model's weights, plus one copy of every table read on
    two stages: Llama-3.2-3B's head reads the embedding through
    `tied_weight` on the last stage."""
    placement = _placed(_program(model_id), 2)
    scope = placement.backbone
    whole = StagePlacement(placement.graph, 1).view(0).weights(scope).values
    total = Counter()
    for s in range(2):
        total.update(placement.view(s).weights(scope).values)
    assert all(total[u] >= n for u, n in whole.items())
    extra = {placement.program.by_uid(u).name for u, n in whole.items()
             if total[u] > n}
    assert extra == ({"embed_tokens"} if shared else set())


def test_an_mtp_head_counts_on_its_stage_only():
    """DeepSeek-V3 at pp = 4 (15 | 15 | 16 | 15 layers): the MTP unit reads
    the norm and runs on stage 3. Timed, its rows count there and nowhere
    else; under the backbone scope they count nowhere."""
    placement = _placed(_program("deepseek-ai/DeepSeek-V3"), 4)
    assert [len(r) for r in placement.ranges] == [15, 15, 16, 15]
    mtp = frozenset({"mtp"})
    counted = [sum(placement.view(s).rows(mtp).values.values())
               for s in range(4)]
    assert counted[:3] == [0, 0, 0] and counted[3] > 0
    assert sum(sum(placement.view(s).rows(placement.backbone).values[u]
                   for u in placement.view(s).rows(mtp).values
                   if placement.view(3).rows(mtp).values[u])
               for s in range(4)) == 0


def test_a_draft_cache_is_kept_on_its_stage_only_and_only_when_asked():
    """DeepSeek-V3 at pp = 2 (31 | 30 layers). The backbone scope keeps no
    MTP cache. A draft's scope adds the MTP layer's, on stage 1 where its
    unit runs, and the stages still sum to the whole model's in each scope."""
    from backend.arch.eval.cache_view import cache_view_from_analysis
    program = _program("deepseek-ai/DeepSeek-V3")
    analysis = shared_walk(Deployment(program).at(batch=1, input_tokens=16))
    placement = StagePlacement.of(analysis, 2)
    draft = placement.backbone | {"mtp"}

    def layers(view, stack):
        return sum(c.count for c in view.entries if c.stack == stack)

    backbone = cache_view_from_analysis(analysis)
    assert backbone == cache_view_from_analysis(
        analysis, cache_scope=placement.backbone)
    assert layers(backbone, "mtp") == 0
    drafted = cache_view_from_analysis(analysis, cache_scope=draft)
    assert (layers(drafted, ""), layers(drafted, "mtp")) == (61, 1)
    assert drafted.within(placement.backbone) == backbone
    staged = [drafted.on(placement.view(s)) for s in range(2)]
    assert [layers(v, "mtp") for v in staged] == [0, 1]
    assert [layers(v, "") for v in staged] == [31, 30]


def test_intra_stage_records_are_the_walks_at_every_pp():
    """pp only adds hops: every other transition record is the one pp = 1
    makes, and summed over stages each is counted as often as at pp = 1."""
    program = _program("Qwen/Qwen3-8B")

    def walk(pp):
        deployment = Deployment(program).at(batch=1, input_tokens=16)
        return shared_walk(replace(deployment, config=ParallelConfig(
            pp=pp, tp_attention=4, tp_dense_ffn=4)))

    def inner(analysis):
        return Counter((r.node, r.input_index, r.collectives, r.site)
                       for r in analysis.redistributes
                       if r.crossing is None)
    assert inner(walk(2)) == inner(walk(1))
    placement = StagePlacement.of(walk(2))
    for record in inner(walk(1)):
        assert sum(placement.view(s).rows(placement.backbone).values[
            record[0]] for s in range(2)) == \
            StagePlacement(placement.graph, 1).view(0).rows(
                placement.backbone).values[record[0]]


@pytest.mark.parametrize("change", ["gap", "overlap"])
def test_a_gap_or_overlap_in_the_iteration_space_is_refused_at_pp_1(change):
    """The partition cuts each repeated layer's iterations, so its
    templates must cover them exactly once. That is checked when a program
    is walked, at any pp — here at pp = 1, with no placement built."""
    from dataclasses import replace as replaced

    from backend.arch.eval.evaluator import Evaluator
    from backend.arch.eval.program import BlockTemplate
    from backend.arch.layout.propagation import LayoutPropagation

    resolved = resolve_snapshot("openai/gpt-oss-20b", shared_loader())
    program = Program.of(resolved).lowered()
    # gpt-oss alternates two templates over its 24 layers.
    backbone = next(s for s in program.stacks if not s.is_aux)
    first = next(s for s in backbone.stages if isinstance(s, BlockStage))
    indices = list(first.template.layer_indices)
    indices = indices[1:] if change == "gap" else [*indices, indices[0] + 1]
    meta = replaced(first.template.meta, layer_indices=tuple(indices))
    broken = replace(first, template=BlockTemplate(meta, first.template.roots))
    program.__dict__["stacks"] = tuple(
        replace(stack, stages=tuple(broken if s is first else s
                                    for s in stack.stages))
        if stack is backbone else stack
        for stack in program.stacks)
    with pytest.raises(ValueError, match=change):
        Evaluator(program, rule=LayoutPropagation.of(ParallelConfig(),
                                                     "prefill"),
                  workload=Workload())
