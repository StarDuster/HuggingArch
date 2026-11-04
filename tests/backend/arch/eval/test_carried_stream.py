"""A stack's main stream is a named stream it declares, carried across the
block boundary.

`streams: { hidden: { carries: true } }` on a repeated stack says:
the stack starts from `inputs[0]`, every iteration binds its block's formal 0
to the stream's current value, and the block's output becomes the new value,
settled at each iteration exit. There is no default stream and no reserved
name; a stack that declares nothing carries nothing.

A stack's variants are evaluated once each and multiplied by their counts.
That equals walking every iteration only when every occurrence of a variant
enters with the state it was evaluated at, so a variant that occurs more than
once must exit with the state it entered with, and a variant that occurs once
and changes the state must not sit between two occurrences of another.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment

HIDDEN = "streams: { hidden: { carries: true } }"

# `norm` keeps the carried state (a replicated activation stays replicated);
# `colpar` is a column-parallel projection whose output is split across the
# tensor-parallel group, so under TP it hands on a different state than it
# received; `act` is elementwise and keeps whatever state it received.
BLOCKS = """\
blocks:
  norm:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      o: { op: add, in: [x, n], dim: d_model }
    outputs: [o]
  colpar:
    graph:
      p: { op: linear, in: x, dims: { in: d_model, out: "n_heads * d_head" },
           weight_axes: [{ world: attention, parts: n_heads, fit: exact }, { replicated: true }] }
    outputs: [p]
  act:
    graph:
      g: { op: gelu, in: x, dim: d_model }
    outputs: [g]
  small:
    graph:
      n: { op: rmsnorm, in: x, dim: d_q_lora, weight_axes: [{ replicated: true }] }
    outputs: [n]
"""

HEAD = """\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims:
  d_model: config.hidden_size
  d_q_lora: "4"
  n_heads: "2"
  d_head: "4"
  n_layers: "{n_layers}"
"""

EMBED = """\
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: d_model, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""

TAIL = """\
  - { name: out, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: d_model }, inputs: [layers],
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def _per_layer(assign: str, *, hidden: bool = True) -> str:
    streams = f"    {HIDDEN}\n" if hidden else ""
    return (
        "  - name: layers\n"
        "    inputs: [embed]\n"
        "    repeat: n_layers\n"
        f"{streams}"
        "    per_layer:\n"
        "      assign:\n"
        f"{assign}"
        "      map:\n"
        "        a: { template: a }\n"
        "        b: { template: b }\n")


def _assign(a: str, b: str) -> str:
    return (f"        - {{ type: a, layers: {a} }}\n"
            f"        - {{ type: b, layers: {b} }}\n")


def _spec(name: str, *, model: str, blocks: str = BLOCKS,
          n_layers: int = 3) -> str:
    return HEAD.format(name=name, n_layers=n_layers) + blocks + EMBED + model


def _variants(a: str, b: str) -> str:
    """BLOCKS plus `a` / `b` aliases naming the variants a test assigns."""
    return BLOCKS + (
        f"  a:\n    graph:\n      v: {{ op: {a}, in: x }}\n    outputs: [v]\n"
        f"  b:\n    graph:\n      v: {{ op: {b}, in: x }}\n    outputs: [v]\n")


def _load(name: str, text: str):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(text)
        loader = SpecLoader(base_dir=str(base))
        return loader.match({"model_type": name, "torch_dtype": "bfloat16", "hidden_size": 8},
                            model_id=name)


def _analyze(resolved, tp: int = 2):
    program = Program.of(resolved)
    analysis = (Deployment(program).on(tp=tp)
                .at(batch=1, input_tokens=4).analyze())
    return program, analysis


def _node(program, template: str, name: str):
    return next(n for t in program.blocks if t.name == template
                for n in t.nodes() if n.name == name)


def _chain(name: str, a: str, b: str, layers_a: str, layers_b: str,
           n_layers: int = 3):
    return _load(name, _spec(
        name, blocks=_variants(a, b), n_layers=n_layers,
        model=_per_layer(_assign(layers_a, layers_b)) + TAIL))


class TestDeclaration:
    def test_two_carried_streams_on_one_stack_are_refused(self):
        model = _per_layer(_assign('["0"]', '["1", "2"]'), hidden=False)
        model = model.replace(
            "    repeat: n_layers\n",
            "    repeat: n_layers\n"
            "    streams: { h1: { carries: true }, h2: { carries: true } }\n")
        with pytest.raises(Exception, match="carries: true` streams"):
            _load("carried_two", _spec(
                "carried_two", blocks=_variants("norm", "norm"),
                model=model + TAIL))

    # A reader names what it reads in an `in:` (a read no node names is
    # refused by the component's own interface check, before this one).
    @pytest.mark.parametrize(("interface", "reads_from"), [
        ("reads: [hidden]", "hidden"), ("stream_out: { hidden: v }", "x")])
    def test_a_component_naming_a_carried_stream_is_refused(
            self, interface, reads_from):
        blocks = _variants("norm", "norm").replace(
            "  a:\n    graph:\n      v: { op: norm, in: x }",
            f"  a:\n    {interface}\n    graph:\n"
            f"      v: {{ op: norm, in: {reads_from} }}")
        with pytest.raises(Exception, match="that stack carries"):
            _load("carried_named", _spec(
                "carried_named", blocks=blocks,
                model=_per_layer(_assign('["0"]', '["1", "2"]')) + TAIL))

    def test_a_formal_named_like_a_read_stream_is_refused(self):
        blocks = _variants("norm", "norm").replace(
            "  a:\n    graph:", "  a:\n    reads: [x]\n    graph:")
        with pytest.raises(Exception, match="also name formal"):
            _load("carried_formal_clash", _spec(
                "carried_formal_clash", blocks=blocks,
                model=_per_layer(_assign('["0"]', '["1", "2"]')) + TAIL))

    def test_a_model_level_op_binds_its_inputs_one_to_one(self):
        """`op:` layers had no arity check: a component called with fewer
        inputs than formals had its first formal filled from a default
        stream. There is none; the count must match."""
        blocks = BLOCKS + (
            "  pair:\n    inputs: [x, y]\n    graph:\n"
            "      s: { op: add, in: [x, y], dim: d_model }\n"
            "    outputs: [s]\n")
        model = (
            "  - { name: layers, inputs: [embed], repeat: n_layers,"
            " template: norm, " + HIDDEN + " }\n"
            "  - { name: both, op: pair, inputs: [layers] }\n") + TAIL
        with pytest.raises(Exception, match="one to one"):
            _load("carried_op_arity", _spec(
                "carried_op_arity", blocks=blocks, model=model))


    def test_a_model_level_primitive_binds_its_inputs_one_to_one(self):
        """`add` reads two tensors; a layer naming one is refused, not left
        to find its second operand somewhere."""
        model = (
            "  - { name: layers, inputs: [embed], repeat: n_layers,"
            " template: norm, " + HIDDEN + " }\n"
            "  - { name: both, op: add, dim: d_model, inputs: [layers] }\n"
            + TAIL)
        with pytest.raises(Exception, match="one to one"):
            _load("carried_prim_arity", _spec("carried_prim_arity",
                                              model=model))


class TestScope:
    def test_two_stacks_named_hidden_hold_two_values(self):
        """The backbone and an MTP head both carry `hidden`; the head starts
        from its own `inputs[0]` (a narrower projection), not from the
        backbone's value of a stream with the same name."""
        model = (
            "  - { name: layers, inputs: [embed], repeat: n_layers,"
            " template: norm, " + HIDDEN + " }\n"
            "  - { name: down, op: linear, inputs: [layers],"
            " dims: { in: d_model, out: d_q_lora},"
            " weight_axes: [{ replicated: true }, { replicated: true }] }\n"
            "  - name: mtp\n    group: true\n    aux_kind: mtp\n"
            "    checkpoint: { prefix: mtp, index: local }\n"
            "    inputs: [down]\n    repeat: \"2\"\n"
            f"    {HIDDEN}\n"
            "    layers:\n"
            "      - { name: mtp, template: small, inputs: [down] }\n"
            + TAIL)
        program, analysis = _analyze(_load("carried_two_stacks", _spec(
            "carried_two_stacks", model=model)))
        norm = _node(program, "small", "n")
        assert analysis.shapes.values[norm.uid].ins[0][-1] == 4


class TestNameScopes:
    """Two tables decide every clash: each stack's carried name (local),
    and each model-wide name's one declaring stack."""

    # Stack `side` declares a model-wide stream `hidden`, produced by one
    # variant and read by the other, and carries a stream of its own (`h`).
    SIDE = BLOCKS + """\
  prod:
    stream_out: { hidden: mk }
    graph:
      mk: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [mk]
  cons:
    reads: [hidden]
    graph:
      u: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      g: { op: mul, in: [u, hidden], dim: d_model }
    outputs: [g]
"""

    @staticmethod
    def _side(carried: str) -> str:
        inner = ("" if carried == "hidden"
                 else " hidden: { src: \"threads hidden\" }")
        return (
            "  - { name: layers, inputs: [embed], repeat: n_layers,"
            " template: norm, " + HIDDEN + " }\n"
            "  - name: side\n    inputs: [layers]\n    repeat: n_layers\n"
            f"    streams: {{ {carried}: {{ carries: true }},"
            f"{inner} }}\n"
            "    per_layer:\n      assign:\n"
            "        - { type: p, range: [\"0\", \"0\"] }\n"
            "        - { type: c, rest: true, reads_from: { hidden: \"0\" } }\n"
            "      map:\n        p: { template: prod }\n"
            "        c: { template: cons }\n"
            "  - { name: out, op: linear, boundary_role: output_head,"
            " dims: { in: d_model, out: d_model }, inputs: [side],"
            " weight_axes: [{ replicated: true }, { replicated: true }] }\n")

    def test_another_stack_may_name_an_inner_stream_hidden(self):
        """`layers` carries `hidden`; `side` carries `h` and declares an
        inner `hidden` of its own, whose reader gets the producer's tensor.
        A carried name is not reserved beyond its stack."""
        program, analysis = _analyze(_load("scope_side_ok", _spec(
            "scope_side_ok", blocks=self.SIDE, model=self._side("h"))), tp=1)
        mul = _node(program, "cons", "g")
        assert [list(i) for i in analysis.shapes.values[mul.uid].ins] == \
            [[1, 4, 8], [1, 4, 8]]

    def test_a_component_reading_its_own_stack_s_carried_name_is_refused(
            self):
        """`side` carries `hidden` itself (and declares no model-wide one):
        its `prod` / `cons` now name their own stack's carried stream."""
        with pytest.raises(Exception, match="bound in stack 'side'"):
            _load("scope_side_bad", _spec(
                "scope_side_bad", blocks=self.SIDE,
                model=self._side("hidden")))

    def test_a_nested_component_is_bound_in_the_stack_too(self):
        """Reaching the component through a block's node is binding it."""
        blocks = BLOCKS + (
            "  deep:\n    reads: [hidden]\n    graph:\n"
            "      g: { op: mul, in: [x, hidden], dim: d_model }\n"
            "    outputs: [g]\n"
            "  outer:\n    graph:\n      d: { op: deep, in: x }\n"
            "    outputs: [d]\n")
        model = ("  - { name: layers, inputs: [embed], repeat: n_layers,"
                 " template: outer, " + HIDDEN + " }\n" + TAIL)
        with pytest.raises(Exception, match="Component 'deep', bound in"):
            _load("scope_nested", _spec("scope_nested", blocks=blocks,
                                        model=model))

    def test_an_inner_stream_name_is_its_stacks(self):
        """`a` and `layers` both declare an inner stream `s`: each name is
        its stack's, and the reader in `layers` says whose write it reads
        (`{stack: a, layer: …}`)."""
        from backend.arch.eval.occurrence import Occurrence
        blocks = BLOCKS + (
            "  w:\n    stream_out: { s: n }\n    graph:\n"
            "      n: { op: rmsnorm, in: x, dim: d_model,"
            " weight_axes: [{ replicated: true }] }\n    outputs: [n]\n"
            "  r:\n    reads: [s]\n    graph:\n"
            "      g: { op: mul, in: [x, s], dim: d_model }\n"
            "    outputs: [g]\n")
        model = (
            "  - { name: a, inputs: [embed], repeat: n_layers, template: w,"
            " streams: { hidden: { carries: true }, s: {} } }\n"
            "  - { name: layers, inputs: [a], repeat: n_layers, template: r,"
            " streams: { hidden: { carries: true }, s: {} },"
            " reads_from: { s: { stack: a, layer: \"n_layers - 1\" } } }\n"
            + TAIL)
        resolved = _load("scope_two_stacks", _spec(
            "scope_two_stacks", blocks=blocks, model=model))
        program = Program.of(resolved)
        (reader,) = [t for t in program.blocks if t.name == "r"]
        (writer,) = [t for t in program.blocks if t.name == "w"]
        owner = reader.meta.iteration_owner
        last = max(writer.layer_indices)
        (node,) = [n for n in writer.nodes() if n.name == "n"]
        assert {program.stream_origin(owner, i, "s")
                for i in reader.layer_indices} == {Occurrence(node.uid, last)}

    @pytest.mark.parametrize("stream", ["down", "head_blk"])
    def test_a_model_wide_stream_named_like_a_stage_is_refused(self, stream):
        """A top-level stage (`down`) or one nested in a group
        (`head_blk`) is published by that name; a stream may not share it."""
        model = (
            "  - { name: layers, inputs: [embed], repeat: n_layers,"
            " template: norm, streams: { hidden: { carries: true"
            f" }}, {stream}: {{}} }} }}\n"
            "  - { name: down, op: linear, inputs: [layers],"
            " dims: { in: d_model, out: d_model },"
            " weight_axes: [{ replicated: true }, { replicated: true }] }\n"
            "  - name: mtp\n    group: true\n    aux_kind: mtp\n"
            "    inputs: [down]\n"
            "    layers:\n"
            "      - { name: head_blk, op: rmsnorm, dim: d_model,"
            " inputs: [down], weight_axes: [{ replicated: true }] }\n"
            + TAIL)
        with pytest.raises(Exception, match="share a name with a model layer"):
            _load(f"scope_stage_{stream}", _spec(
                f"scope_stage_{stream}", model=model))


class TestChain:
    def test_a_variant_that_changes_the_state_hands_it_to_the_next(self):
        """`colpar` occurs once, at iteration 0, and exits split across the
        TP group; `act` covers iterations 1-2 and enters with that split
        state — the chain's physical meaning — and keeps it, so it is a
        fixed point on its own entry."""
        program, analysis = _analyze(_chain(
            "carried_handoff", "colpar", "act", '["0"]', '["1", "2"]'))
        entry = analysis.states.values[
            next(n for n in program.nodes() if n.name == "embed").uid]
        handed = analysis.states.values[_node(program, "a", "v").uid]
        gelu = _leaf(program, _node(program, "b", "v"))
        assert handed != entry
        assert analysis.input_states[(gelu.uid, 0)] == handed

    def test_a_variant_changing_the_state_inside_another_s_span_is_refused(
            self):
        """`norm` at iterations 0 and 2, `colpar` at 1: folding `norm` x2 on
        the entry state would evaluate its occurrence at 2 on a state it
        never enters with."""
        with pytest.raises(NotImplementedError, match="lies inside"):
            _analyze(_chain("carried_straddled", "norm", "colpar",
                            '["0", "2"]', '["1"]'))

    def test_the_same_schedule_passes_when_nothing_changes(self):
        program, analysis = _analyze(_chain(
            "carried_straddled_ok", "norm", "norm", '["0", "2"]', '["1"]'))
        entry = analysis.states.values[
            next(n for n in program.nodes() if n.name == "embed").uid]
        for template in ("a", "b"):
            add = _node(program, template, "v")
            assert analysis.input_states[(_leaf(program, add).uid, 0)] \
                == entry

    def test_an_interleaved_variant_must_keep_the_state(self):
        """`a, b, a, b`: `colpar` occurs twice, not contiguously; it must be
        a fixed point on its entry, and it is not."""
        with pytest.raises(NotImplementedError, match="fixed point"):
            _analyze(_chain("carried_interleaved", "norm", "colpar",
                            '["0", "2"]', '["1", "3"]', n_layers=4))


def _mtp(inner_a: str, inner_b: str, outer: str) -> str:
    return (
        "  - { name: layers, inputs: [embed], repeat: n_layers,"
        " template: norm, " + HIDDEN + " }\n"
        "  - name: mtp\n    group: true\n    aux_kind: mtp\n"
        "    checkpoint: { prefix: mtp, index: local }\n"
        f"    inputs: [layers]\n    repeat: \"{outer}\"\n"
        "    layers:\n"
        "      - name: mtp.layers\n        inputs: [layers]\n"
        "        repeat: \"2\"\n"
        f"        {HIDDEN}\n"
        "        per_layer:\n          assign:\n"
        "            - { type: a, layers: [\"0\"] }\n"
        "            - { type: b, layers: [\"1\"] }\n"
        "          map:\n            a: { template: a }\n"
        "            b: { template: b }\n"
        + TAIL)


class TestOuterRepeat:
    """Nemotron-3-Ultra's MTP: an outer group `repeat:` around a per-layer
    stack. The chain runs once per outer iteration, so every variant must be
    a fixed point and the edge from the chain's end back to its start is a
    consistency check, not a redistribution."""

    def test_states_that_all_agree_pass(self):
        program, analysis = _analyze(_load("carried_outer_ok", _spec(
            "carried_outer_ok", blocks=_variants("norm", "norm"),
            model=_mtp("norm", "norm", "2"))))
        assert {t.outer_repeat for t in program.blocks
                if t.name in ("a", "b")} == {2}
        # The edge from the chain's end back to its start moves nothing.
        head = {n.uid for n in program.nodes() if program.stack_of(n) == "mtp"}
        assert not [r for r in analysis.redistributes if r.node in head]

    def test_a_variant_that_changes_the_state_is_refused(self):
        with pytest.raises(NotImplementedError, match="fixed point"):
            _analyze(_load("carried_outer_bad", _spec(
                "carried_outer_bad", blocks=_variants("norm", "colpar"),
                model=_mtp("norm", "colpar", "2"))))

    def test_one_outer_iteration_hands_the_state_on(self):
        """With the outer repeat at 1 the same variants are single
        occurrences and the chain passes `colpar`'s split exit on."""
        _analyze(_load("carried_outer_one", _spec(
            "carried_outer_one", blocks=_variants("norm", "colpar"),
            model=_mtp("norm", "colpar", "1"))))


def _leaf(program, node):
    """The primitive a template's single-node body evaluates."""
    while node.children:
        node = node.children[0]
    return node


class TestAuxGroupStages:
    def test_each_template_child_of_a_group_is_its_own_stage(self):
        """Qwen3.5-style MTP: a one-shot front end named like its group
        (`mtp`), then the repeated `mtp.layers`, then `mtp.norm` reading
        them. The front end is one stage and `mtp.layers` another; a child
        sharing its group's name claims no sibling's template."""
        model = (
            "  - { name: layers, inputs: [embed], repeat: n_layers,"
            " template: norm, " + HIDDEN + " }\n"
            "  - name: mtp\n    group: true\n    aux_kind: mtp\n"
            "    checkpoint: { prefix: mtp.layers, index: local }\n"
            "    inputs: [layers]\n"
            "    layers:\n"
            "      - { name: mtp, template: norm, inputs: [layers] }\n"
            "      - { name: mtp.layers, repeat: \"2\", template: act,"
            " inputs: [mtp], " + HIDDEN + " }\n"
            "      - { name: mtp.norm, op: rmsnorm, dim: d_model,"
            " inputs: [mtp.layers], weight_axes: [{ replicated: true }] }\n"
            + TAIL)
        program, _analysis = _analyze(_load("aux_group_stages", _spec(
            "aux_group_stages", model=model)))
        (stack,) = [s for s in program.stacks if s.name == "mtp"]
        from backend.arch.eval.program import BlockStage
        assert [(st.name,
                 st.template.name if isinstance(st, BlockStage) else None)
                for st in stack.stages] == [
            ("mtp", "norm"), ("mtp.layers", "act"), ("mtp.norm", None)]
