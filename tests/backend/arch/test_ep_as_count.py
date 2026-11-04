"""An expert bank's experts are an axis of its tensors: its rows conserve
when the bank and its denominator scale together, and the router, sized by
the destinations it scores, scales.

**The bank.** The expert count is in a bank row's work twice, with opposite
signs:

* the bank's tensors lead with its experts: its weights are
  `[n_expert_copies, out, in]` (`shape:`) and the activation `permute` writes
  is `[n_expert_copies, M, d]`;
* each expert's rows are `M = B·s_ctx·n_experts_active·dp_attention /
  n_route_targets`, the destinations the top-k draws from — the copy count
  again, plus LongCat's zero-compute reserve where there is one.

So every bank row's FLOPs are homogeneous of degree +1 in the copies and -1
in the targets, and scaling both by one factor leaves the row's work alone.
That is checked per ROW, exactly, not on a total: a walk that computed a
bank as a constant also conserves under joint scaling, so joint conservation
alone cannot tell a bank that cancels its count from one that dropped it.

**The router** is sized by the destinations it scores (`n_route_targets`, at
87 component positions), so its work moves when they do — linearly on the
corpus, and affinely in general: Inkling's gate scores
`n_route_targets + n_shared_experts` lanes, so scaling the targets alone
moves it by less than the factor. No row outside the bank reads the expert
axis.

**A pool whose two counts are one number.** A DSpark draft stage rebinds
`n_experts` to one config fact, `n_dspark_experts`, and both its copies and
its denominator are valued in it: scaling that one symbol cannot separate
the two degrees, and joint conservation holds vacuously. Such a row is held
by its own operand and by the IDENTITY half of its tracked cost: the
operand's axis 0 is the expert axis by identity, valued in the pool's
count, its rows axis is valued over the pool's denominator, and the cost's
identity half is still separately +1 in `n_expert_copies` and -1 in
`n_route_targets` — the names the cost was computed from, whatever they
came to.

The scaling happens in the symbolic value domain, so no magnitude is
hand-closed here. It holds because `M` is exact
(`Evaluator._tokens_per_expert` keeps it a `Rational`).

**Grouped GEMM is not a structural op.** A grouped kernel that runs the
rank's experts in one launch is a KERNEL-layer realization: a routed linear
resolves to the `grouped_gemm` measured-corpus signature. No registered op
declares the expert axis in its own weight template.

**Named gap: gated globally, ungated per rank.** On a rank the bank's axis
is `E_local` while `M`'s denominator is the whole pool, so `E_local ×
tokens·k/E` is count-free only where `E_local == E / ep` exactly — which the
expert axis's `fit: exact` placement refuses otherwise.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import pytest
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.arch.eval.dim import expr_of, value_of
from backend.arch.eval.storage import is_expert_axis
from backend.inference import benchmark_identity as _bid
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step
from tests.snapshots import (
    SNAPSHOT_DIR,
    any_snapshot,
    resolve_snapshot,
    shared_walk)

COPIES, TARGETS = "n_expert_copies", "n_route_targets"
_LAMBDA = sympy.Symbol("lam", positive=True)


def degree(expr: Any, symbols: FrozenSet[Any]) -> Optional[int]:
    """The power of λ the expression picks up when every one of `symbols`
    is multiplied by λ at once: an exact homogeneity degree in -2..2, else
    None (not homogeneous in them)."""
    expr = sympy.expand(sympy.sympify(expr))
    scaled = sympy.expand(expr.subs({s: _LAMBDA * s for s in symbols},
                                    simultaneous=True))
    for d in (0, 1, -1, 2, -2):
        if sympy.simplify(scaled - _LAMBDA ** d * expr) == 0:
            return d
    return None


def _symbols(value: Any) -> FrozenSet[Any]:
    return frozenset(getattr(sympy.sympify(expr_of(value)), "free_symbols", ()))


@dataclass
class Row:
    uid: int
    path: str
    flops: Any                 # the tracked cost: `expr` identity, `value` value
    params: Any
    copies: FrozenSet[Any]     # the value-half symbols of its frame's copies
    targets: FrozenSet[Any]    # ... and of its frame's denominator
    pool: int                  # uid of the frame that binds them
    operand: Optional[tuple] = None
    frame: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutedRows:
    """The rows of one symbolic walk, split as the identity reads them."""
    bank: List[Row]
    boundary: List[Row]
    other: List[Row]

    @classmethod
    def of(cls, analysis: Any) -> "RoutedRows":
        program = analysis.program
        routed = program.routed_uids
        boundary_uids = program.dispatch_members - routed
        out = cls([], [], [])
        for node in program.nodes():
            if node.children or node.uid not in analysis.cost.values:
                continue
            cost = analysis.cost.values[node.uid]
            # The frame is the nearest enclosing CALL's (a leaf's own `dims:`
            # names its parameters, not the frame).
            frame_node, frame = node.parent, {}
            while frame_node is not None:
                frame = analysis.node_dims.values.get(frame_node.uid) or {}
                if COPIES in frame:
                    break
                frame_node = frame_node.parent
            shape = analysis.shapes.values.get(node.uid)
            row = Row(node.uid, node.display_path, cost.flops, cost.params,
                      _symbols(frame.get(COPIES, 0)),
                      _symbols(frame.get(TARGETS, 0)),
                      frame_node.uid if frame_node is not None else -1,
                      tuple(shape.ins[0]) if shape and shape.ins else None,
                      dict(frame))
            if node.uid in routed:
                out.bank.append(row)
            elif node.uid in boundary_uids:
                out.boundary.append(row)
            else:
                out.other.append(row)
        return out


def violations(analysis: Any) -> List[str]:
    """Every way one routed walk fails the identity, as readable lines.
    Empty is the identity holding."""
    rows = RoutedRows.of(analysis)
    bad: List[str] = []
    if not rows.bank:
        return ["a routed model has no bank rows"]
    all_copies = frozenset().union(*(r.copies for r in rows.bank))
    all_targets = frozenset().union(*(r.targets for r in rows.bank))
    for row in rows.bank:
        flops = expr_of(row.flops)
        if sympy.sympify(flops) == 0 and sympy.sympify(expr_of(row.params)) == 0:
            continue
        # (2') the factors exist in the row's own frame.
        if not row.copies or not row.targets:
            bad.append(f"{row.path}: its frame binds no symbol for the copies "
                       f"or the denominator ({sorted(map(str, row.copies))}, "
                       f"{sorted(map(str, row.targets))})")
            continue
        if sympy.sympify(flops) != 0:
            # (1) joint conservation.
            if degree(flops, row.copies | row.targets) != 0:
                bad.append(f"{row.path}: FLOPs move when its copies and "
                           f"denominator scale together")
            # (2) the exact factor, where the two are separable.
            if not row.copies & row.targets:
                d_e, d_t = degree(flops, row.copies), degree(flops, row.targets)
                if (d_e, d_t) != (1, -1):
                    bad.append(f"{row.path}: FLOPs are degree {d_e} in the "
                               f"copies and {d_t} in the denominator, not "
                               f"+1 / -1")
            # (2') the operand carries both factors.
            lead, rows_axis = (row.operand or (None, None))[:2]
            if lead is None or not is_expert_axis(lead):
                bad.append(f"{row.path}: its operand does not lead with the "
                           f"expert axis")
            else:
                copies = sympy.sympify(expr_of(row.frame[COPIES]))
                if sympy.simplify(sympy.sympify(expr_of(lead)) - copies) != 0:
                    bad.append(f"{row.path}: its expert axis is "
                               f"{expr_of(lead)}, not its frame's copies "
                               f"{copies}")
                # A frame that rebinds the pool's count carries it as its
                # copies (the model's own frame names the copies by itself).
                # The identity half keeps both names, so the two are
                # compared on the value half.
                count = sympy.sympify(expr_of(row.frame["n_experts"]))
                if count != sympy.Symbol("n_experts") and \
                        value_of(row.frame[COPIES]) != value_of(row.frame["n_experts"]):
                    bad.append(f"{row.path}: its frame's copies {copies} are "
                               f"not the frame's expert count {count}")
                if degree(expr_of(rows_axis), row.targets) != -1:
                    bad.append(f"{row.path}: its rows axis "
                               f"{expr_of(rows_axis)} is not over its "
                               f"denominator")
            # (2') the law, on identity: the names the cost was computed from.
            ident = expr_of(row.flops)
            if (degree(ident, frozenset({sympy.Symbol(COPIES)})),
                    degree(ident, frozenset({sympy.Symbol(TARGETS)}))) != (1, -1):
                bad.append(f"{row.path}: the identity half of its cost "
                           f"({ident}) is not +1 in {COPIES} and -1 in "
                           f"{TARGETS}")
        # (3) storage.
        params = expr_of(row.params)
        if sympy.sympify(params) != 0 and (degree(params, row.copies) != 1 or (
                not row.copies & row.targets
                and degree(params, row.targets) != 0)):
            bad.append(f"{row.path}: params are not +1 in the copies and "
                       f"0 in the denominator")
    pools: Dict[int, List[Row]] = {}
    for row in rows.bank:
        pools.setdefault(row.pool, []).append(row)
    for members in pools.values():
        stored = sympy.expand(sum(sympy.sympify(expr_of(r.params)) for r in members))
        scaled = stored.subs({s: 3 * s for s in members[0].copies},
                             simultaneous=True)
        if sympy.expand(scaled - stored) == 0:
            bad.append(f"pool at {members[0].path}: its bank's parameters do "
                       f"not grow with its own copies")
    # (4) the dispatch boundary conserves when the copies and the denominator
    # scale together. A row on the token side (permute) reads neither; one
    # billed for the routed rows it reduces (combine: `n_expert_copies * M *
    # d_model`, ae6dca52d) reads both, +1 / -1, and the count cancels. What
    # fails is a row that leaves the expert count uncancelled.
    bad.extend(f"{row.path}: a dispatch boundary row moves when its copies "
               f"and denominator scale together"
               for row in rows.boundary
               if degree(expr_of(row.flops), all_copies | all_targets) != 0)
    # (5) every other row: no expert axis, affine in the destinations, and
    # something (the router) sized by them.
    only_copies = all_copies - all_targets
    sized = False
    for row in rows.other:
        flops = sympy.expand(sympy.sympify(expr_of(row.flops)))
        if only_copies and degree(flops, only_copies) != 0:
            bad.append(f"{row.path}: reads the expert axis outside the bank")
        bad.extend(f"{row.path}: not affine in {t}" for t in all_targets
                   if sympy.simplify(sympy.diff(flops, t, 2)) != 0)
        if sympy.simplify(sympy.diff(flops, sympy.Symbol(TARGETS))) != 0:
            sized = True
    if not sized:
        bad.append(f"no row outside the bank is sized by the model's "
                   f"destinations `{TARGETS}` — the router lost its width")
    return bad


def _routed_step(name: str) -> Any:
    """One prefill step of `name`, or None when the model routes nothing."""
    step = evaluate_step(resolve_snapshot(name), ParallelConfig(),
                         phase="prefill", batch=1, input_tokens=512, output_tokens=0)
    return step if step.analysis.moe is not None else None


def _walk(deployment: Any) -> Any:
    """The walk over the workload's counts, unread: each value's identity
    half (`Dim.expr`) is its law in the model's names."""
    from dataclasses import replace
    return shared_walk(replace(deployment, workload=deployment.workload.symbolic()[0]))


def _symbolic(step: Any) -> Any:
    return _walk(step.as_deployment())


# -- a synthetic MoE whose every factor is separable --------------------------

_COMPONENTS = Path(__file__).resolve().parents[3] / "backend" / "arch" / "components"
TOY_CONFIG = {
    "model_type": "ep_identity_toy",
    "torch_dtype": "bfloat16", "hidden_size": 8,
    "moe_intermediate_size": 4,
    "num_experts": 8, "num_experts_per_tok": 2,
    "draft_num_experts": 4, "vocab_size": 32,
}
# A backbone `moe_ffn` with two zero-compute destinations (copies and
# denominator separable even in value), a pool that rebinds its own count,
# and an Inkling gate (`n_route_targets + n_shared_experts` lanes: affine,
# not linear, in the destinations).
TOY_MODEL = """\
match: { model_type: [ep_identity_toy], model_id: [ep_identity_toy] }
dims:
  d_model: config.hidden_size
  d_ffn_expert: config.moe_intermediate_size
  n_shared_experts: "2"
  n_zero_experts: "2"
  n_vocab: config.vocab_size
  n_experts: config.num_experts
  n_experts_active: config.num_experts_per_tok
  n_expert_groups: "1"
  n_groups_active: "1"
  n_routed_scale: "0"
  n_topk_renorm: "0"
  n_draft_experts: config.draft_num_experts
dim_symbols:
  n_draft_experts: { kind: whole, group: mtp_speculative, meaning: "a second pool's size" }
blocks:
  moe_blk:
    inputs: [x]
    graph:
      ffn: { op: moe_ffn, in: x }
      o: { op: add, in: [x, ffn], dim: d_model }
    outputs: [o]
  pool_blk:
    inputs: [x]
    graph:
      ffn: { op: moe_ffn, in: x, dims: { n_experts: n_draft_experts } }
      o: { op: add, in: [x, ffn], dim: d_model }
    outputs: [o]
  inkling_blk:
    inputs: [x]
    graph:
      ffn: { op: inkling_moe, in: x }
      o: { op: add, in: [x, ffn], dim: d_model }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: moe_layers, repeat: "1", template: moe_blk, inputs: [embed_tokens],
      streams: { hidden: { carries: true } } }
  - { name: pool_layers, repeat: "1", template: pool_blk, inputs: [moe_layers],
      streams: { hidden: { carries: true } } }
  - { name: inkling_layers, repeat: "1", template: inkling_blk, inputs: [pool_layers],
      streams: { hidden: { carries: true } } }
  - { name: norm, op: rmsnorm, dim: d_model, inputs: [inkling_layers],
      weight_axes: [{ replicated: true }] }
  - { name: lm_head, op: linear, boundary_role: output_head, inputs: [norm],
      dims: { in: d_model, out: n_vocab },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def toy_resolved(model: str = TOY_MODEL, config: Optional[dict] = None) -> Any:
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        shutil.copytree(_COMPONENTS / "library", base / "components" / "library")
        for sub in ("components/adapters", "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "ep_identity_toy.yaml").write_text(model)
        return SpecLoader(base_dir=str(base)).match(
            config or TOY_CONFIG, model_id="ep_identity_toy")


def toy_symbolic(model: str = TOY_MODEL) -> Any:
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    return _walk(Deployment(Program.of(toy_resolved(model))).at(
        batch=1, input_tokens=8))


class TestNoOpDeclaresTheExpertAxis(unittest.TestCase):
    def test_no_registered_op_writes_the_expert_axis_into_its_weight(self):
        import backend.arch.dims as dim_registry
        from backend.arch.ops import OP_REGISTRY

        def expert(label: Any) -> bool:
            declared = dim_registry.get(str(label))
            return declared is not None and declared.kind == "expert_copies"

        grouped = sorted(
            name for name, spec in OP_REGISTRY.items()
            if any(expert(label) for label in (spec.weight_shape or ()))
            or any(expert(label) for decl in spec.tensors
                   for label in decl.shape))
        self.assertEqual(grouped, [], (
            "these ops declare the expert axis in their own weight template, "
            "a grouped op in the spec; the expert axis belongs to the node's "
            "`shape:`, and grouping to the kernel layer"))

    def test_expert_gemm_still_maps_to_the_grouped_kernel_signature(self):
        # Where a grouped kernel legitimately lives: the kernel layer, keyed
        # on the node's `[E, M, d]` operand, after fusion.
        self.assertEqual(
            _bid.resolve_op("linear", routed=True), "grouped_gemm",
            "a routed linear no longer resolves to the grouped_gemm kernel "
            "signature — the measured corpus join would break.")
        self.assertNotEqual(
            _bid.resolve_op("linear", routed=False), "grouped_gemm")


class TestBankRowsConserveAndTheRouterScales(unittest.TestCase):
    """The identity on the synthetic spec and on the first routed snapshot,
    so a default run carries the lock; the corpus-wide form below proves
    every routed model states it."""

    def test_on_a_spec_where_every_factor_separates(self):
        bad = violations(toy_symbolic())
        self.assertEqual(bad, [], "\n".join(bad))

    def test_the_inkling_gate_is_affine_not_linear_in_the_destinations(self):
        """The gate scores `n_route_targets + n_shared_experts` lanes: its
        work moves with the destinations, by less than their factor."""
        analysis = toy_symbolic()
        (gate,) = [n for n in analysis.program.nodes()
                   if n.name == "gate" and n.op == "linear"
                   and n.parent is not None and n.parent.op == "inkling_moe"]
        flops = expr_of(analysis.cost.values[gate.uid].flops)
        targets = sympy.Symbol(TARGETS)
        self.assertNotEqual(sympy.diff(flops, targets), 0)
        self.assertIsNone(degree(flops, frozenset({targets})))

    def test_on_the_first_routed_snapshot(self):
        any_snapshot()
        for path in sorted(SNAPSHOT_DIR.glob("*.json")):
            step = _routed_step(path.name)
            if step is not None:
                bad = violations(_symbolic(step))
                self.assertEqual(bad, [], f"{path.name}:\n" + "\n".join(bad))
                return
        raise unittest.SkipTest("no routed model in the checked-out corpus")


class TestTheInklingGateIsWiderThanItsDestinations(unittest.TestCase):
    """Inkling's gate emits `n_experts + n_shared_experts` logits and runs
    its top-k over the routed ones alone (vllm/models/inkling/nvidia/moe.py:
    208-216). The width is written in its own component as
    `n_route_targets + n_shared_experts`; no snapshot loads Inkling, so this
    is its only gate."""

    def test_the_gate_is_routed_plus_shared_and_the_slice_routed(self):
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment

        model = (TOY_MODEL.replace('  n_zero_experts: "2"\n', "")
                 .replace('n_shared_experts: "2"', 'n_shared_experts: "2"'))
        config = {**TOY_CONFIG, "num_experts": 7}
        resolved = toy_resolved(model, config)
        deployment = Deployment(Program.of(resolved)).at(batch=1, input_tokens=8)
        numeric, symbolic = shared_walk(deployment), _walk(deployment)

        def node(analysis, name):
            (found,) = [n for n in analysis.program.nodes()
                        if n.name == name and n.parent is not None
                        and n.parent.op == "inkling_moe"]
            return found

        gate, routed = node(numeric, "gate"), node(numeric, "gate_routed_slice")
        self.assertEqual(int(value_of(numeric.shapes.values[gate.uid].out[-1])), 9)
        self.assertEqual(int(value_of(numeric.shapes.values[routed.uid].ins[0][-1])), 9)
        self.assertEqual(int(value_of(numeric.shapes.values[routed.uid].out[-1])), 7)
        width = expr_of(symbolic.shapes.values[node(symbolic, "gate").uid].out[-1])
        self.assertEqual(sympy.simplify(width - sympy.Symbol(TARGETS)
                                        - sympy.Symbol("n_shared_experts")), 0)


@pytest.mark.regression
class TestEveryRoutedModelStatesTheIdentity(unittest.TestCase):
    """The same identity over the whole corpus. Marked `regression` because
    it walks every snapshot. A routed model that cannot state it is a routed
    model whose expert work is counted wrong, and the fix is the engine, not
    an entry excusing it."""

    def test_every_routed_model(self):
        any_snapshot()
        offenders, routed = [], 0
        for path in sorted(SNAPSHOT_DIR.glob("*.json")):
            step = _routed_step(path.name)
            if step is None:
                continue
            routed += 1
            offenders += [f"{path.name}: {line}"
                          for line in violations(_symbolic(step))]
        self.assertEqual(offenders, [], "\n".join(offenders))
        self.assertGreater(routed, 0, "no routed model in the corpus")


class TestARepeatOverTheExpertAxisIsRefused:
    """A bank counted by `repeat:` would carry its experts as copies AND
    have no share of them a token runs; the walk refuses it, inside the
    zone or over an `expert_copies` symbol anywhere.

    The refusal is about what a SPEC declares, so both cases are declared
    in a spec rather than produced by editing a real model's graph. They
    used to rewrite `repeat` / `repeat_raw` on a node of a resolved Qwen —
    which tested the edit rather than the declaration, and corrupted the
    program for every later holder: running this class before
    `test_program_moe`'s corpus sweep turned the sweep red, and the reverse
    order was green.

    The two cases are the two arms of the refusal (`Evaluator._copies`):
    a `repeat:` inside a scope that already has a routed share, and a
    `repeat:` whose value names an `expert_copies` symbol anywhere.
    """

    #: An expert bank written out here rather than called from the library,
    #: because the `repeat:` has to sit ON one of the bank's own leaves and
    #: a call site cannot override a field inside the component it calls.
    #: The weights lead with `n_expert_copies`, as every routed bank's do.
    BANK = """\
  repeated_bank:
    inputs: [x]
    graph:
      gate_proj: { op: linear, in: x, dims: { in: d_model, out: d_ffn }, repeat: "2",
                   shape: [n_expert_copies, d_ffn, d_model],
                   weight_axes: [{ world: moe_copies, parts: n_expert_copies, fit: exact },
                                 { world: moe_expert, parts: d_ffn, fit: exact },
                                 { replicated: true }] }
      up_proj:   { op: linear, in: x, dims: { in: d_model, out: d_ffn },
                   shape: [n_expert_copies, d_ffn, d_model],
                   weight_axes: [{ world: moe_copies, parts: n_expert_copies, fit: exact },
                                 { world: moe_expert, parts: d_ffn, fit: exact },
                                 { replicated: true }] }
      act:       { op: silu, in: gate_proj, dim: d_ffn }
      gate_mul:  { op: mul, in: [act, up_proj], dim: d_ffn }
      down_proj: { op: linear, in: gate_mul, dims: { in: d_ffn, out: d_model },
                   shape: [n_expert_copies, d_model, d_ffn],
                   weight_axes: [{ world: moe_copies, parts: n_expert_copies, fit: exact },
                                 { replicated: true },
                                 { world: moe_expert, parts: d_ffn, fit: exact }] }
    outputs: [down_proj]
  bank_blk:
    inputs: [x]
    graph:
      gate:    { op: router_softmax_topk, in: x }
      permute: { op: permute, in: [x, gate] }
      experts: { op: repeated_bank, role: ffn_moe, in: permute,
                 dims: { d_model: d_model, d_ffn: d_ffn_expert } }
      combine: { op: combine, in: [experts, gate] }
      o:       { op: add, in: [x, combine], dim: d_model }
    outputs: [o]
"""

    @staticmethod
    def _walk(model: str) -> Any:
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from tests.snapshots import own_walk
        return own_walk(
            Deployment(Program.of(toy_resolved(model))).at(
                batch=1, input_tokens=8),
            why="a toy spec of this test's own, shared with nobody")

    def test_inside_the_zone(self):
        """A `repeat:` on a leaf of a routed bank: the bank's scope already
        carries the share a token runs, so a copy count has none."""
        model = TOY_MODEL.replace("  pool_blk:\n", self.BANK + "  pool_blk:\n")
        model = model.replace(
            '  - { name: moe_layers, repeat: "1", template: moe_blk,',
            '  - { name: moe_layers, repeat: "1", template: bank_blk,')
        with pytest.raises(ValueError, match="over a routed expert bank"):
            self._walk(model)

    def test_over_the_copies_symbol(self):
        """A `repeat:` whose VALUE names the expert axis, on a node that is
        nowhere near a bank — refused on the symbol alone."""
        model = TOY_MODEL.replace(
            "  - { name: norm, op: rmsnorm, dim: d_model, inputs: [inkling_layers],",
            "  - { name: norm, op: rmsnorm, dim: d_model, inputs: [inkling_layers],"
            ' repeat: "n_expert_copies",')
        with pytest.raises(ValueError, match="over a routed expert bank"):
            self._walk(model)
