"""Two gates over the symbolic domain, both counted ratchets.

A `Dim` holds an expression and a number at once, and the whole point of the
walk is that a consumer can read a row again at another batch or another
context without walking again. Two things have to hold for that, and until
now neither was a mechanism:

**No second road from "cannot decide" to an answer.** Ask an extent a
magnitude question with a Python operator and only its number half answers
(`Dim.__gt__` compares the values, `Dim.__bool__` tests one, `int(dim)` takes
it). Where that number half is itself a symbol there is no answer — sympy
raises `TypeError` on `bool()` of an undecidable relation — and the ways of
turning the non-answer back into a value are a closed set of seven shapes,
which are the suffixes of the names below:

  `order-decision`     `a > b` deciding a branch
  `equality-decision`  `a == b`, which compares the value halves
  `zero-test`          `if extent:` / `not extent` / `bool(extent)`
  `builtin-minmax`     `min(a, b)` picking one and returning it
  `or-default`         `extent or 0`, a number behind a falsy extent
  `except-TypeError`   catching the refusal and answering in its place
  `bare-division`      `a / b` / `a // b`, which answer in BOTH domains and
                       answer differently

Each decides something and writes nothing down. One of them survived a
review (`traffic._wider` caught the `TypeError` of `b > a` and kept `a`,
billing the first of two incomparable cache spans) and was found by reading.
`test_no_new_place_decides_an_undecidable_magnitude` is that reading, run
every time.

Division is the one shape that does not refuse. `128 / 128` on two Python
ints is the float `1.0`; the same division of two laws is sympy's exact `1`.
`//` floors a number where `Dim.__floordiv__`'s expression half divides
exactly. So the operator returns either way, one answer per binding, of two
kinds — and the two readers of that one row then disagree about what it says.
Measured at `layout.physical.pairs_share`, where a decode step's live pair
count published as the string "4224.00000000000" on the walk's own reader and
the integer 4224 on the reader that evaluates the same law at a point; five
tests in `tests/backend/inference/test_sweep_parity.py` were red on it for a
long time, reading as a phase quirk rather than as a division. Only one of
the two phases was wrong, because prefill's counts are laws in `s_input`
where sympy keeps them exact.

The fix is a named verb, and the choice of verb IS the declaration of intent
(all in `eval/dim.py`): `exact_div` for a division the caller declares exact,
which RAISES when the declaration is wrong; `ceil_div` / `floor_div` for a
rounding the law intends; `ratio(part, whole, empty=…)` for a proportion,
which makes the caller say what an empty denominator means; and `exact` on
both halves for a ratio of two counts that must stay a `Rational`. The scan
reads those declarations rather than a list of names: a division through
anything declaring a SymPy return (`exact`, `algebra`, `ratio_symbol` —
`_algebra_sympy_carriers`) is sympy's in either domain, so it is not this
shape and does not appear below.

A `bare-division` admitted below needs the reason stated the other way round
from the six comparison shapes: there, the question is whether an expression
can REACH the operator. Here an expression reaching it is the SAFE case —
sympy divides exactly. What a division's reason has to say is why its two
value halves cannot both be plain Python ints (a declared `float` operand
that `_exact_values` has already made exact, an `int()` cast into the display
domain, a sibling `%` refusal that has already gated the remainder).

**A gate must not stop gating where the domain is symbolic.** `ops_registry`
reads the extents its `input_contract`s check through a helper declared
`-> Optional[int]` — "the number, or None where the domain is symbolic" — and
every caller then skips its check. The contract reports clean either way.
`test_no_new_gate_skips_itself_on_a_symbol` pins the eleven that exist;
`ALLOWED_SKIPS` says what each one drops.

WHY A LIST AND NOT A RULE
-------------------------
Neither question is decidable by reading syntax. There are no types here, so
`tools.scoreboard` approximates "could this be an extent" with an
intraprocedural taint whose sources are read off declarations — the public
surface of `eval/dim.py` and `eval/values.py`, parsed from those modules, and
`Extent` / `Dim` annotations — and subtracts anything the same function
measures, indexes, iterates or appends to, because a container is not a
magnitude. That classifier has false positives and false negatives, so the
count alone would be a poor gate: it would go red on a rename.

So the sites are pinned BY NAME, the way `test_eval_boundary` pins eval's
defensive accesses. A name is `<file>::<qualified function>::<shape>`, with
an occurrence number where one function holds several of a shape — stable
while the lines around it move. A new name is a new road and fails here; a
name that disappears also fails, with a different message, because a stale
allowance is where the next regression hides.

WHAT THE SCAN CANNOT SEE, STATED RATHER THAN GUESSED
----------------------------------------------------
* The taint is flow-INSENSITIVE: one name bound twice in a function carries
  both bindings' taint for the whole of it. That is how
  `cache.py::_of_component::skip` below got its name — the skip it stands for
  is real (a symbolic cached shape leaves `inner` / `rows` untracked, and the
  scale grouping that reads them is what does not run), but it happens at the
  consumer of `_row_split`, not at the `is not None` the scan names.
* Taint does not cross a function boundary except through a declared
  annotation, so an extent reaching a comparison through an undeclared field
  or a `**kwargs` is invisible here.
* `getattr(x, "...", <number>)` is a seventh shape of the same kind, and it
  is deliberately NOT counted here: every `getattr` / `hasattr` in
  `backend/` is already pinned by the `defensive_access` counter (116, exact)
  and every one in `eval/` additionally per-module by
  `test_eval_boundary.ACCESS`. A third count of the same construct would be
  the second table CLAUDE.md §1 refuses. A new extent-shaped `getattr`
  default is red today — in those gates.
* Scope is `backend/arch/eval` and `backend/arch/layout` for the first gate.
  `ops_registry.py`'s `_history_contract` decides `value_of(history[2]) == 0`
  with no guard at all, which is this shape outside that scope; it is written
  up beside `ALLOWED_SKIPS` rather than counted twice.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools import scoreboard as sb

#: Every magnitude decision taken on a possible expression today, by name.
#: This list may only get SHORTER. The way to shorten it is to ask `dim` the
#: question instead — `equal` / `larger` / `is_whole` / `assume` answer over
#: every admitted point, record an assumption, or refuse — and delete the
#: line here in the same commit.
#:
#: A name is admitted the other way only where no expression can reach the
#: operator, and then a comment beside it says why the scan sees one and the
#: code does not; a reason that is not checkable by reading the declarations
#: it cites is a fix someone skipped.
ALLOWED_DECISIONS = {
    "backend/arch/eval/access.py::_formula_display::equality-decision",
    "backend/arch/eval/access.py::_formula_display::zero-test",
    "backend/arch/eval/analysis.py::Analysis._bytes_by::order-decision",
    "backend/arch/eval/analysis.py::Analysis._moe_pool_stacks::or-default",
    "backend/arch/eval/analysis.py::Analysis._padded_rows::order-decision",
    "backend/arch/eval/analysis.py::Analysis.dominant_cache_dtype::order-decision",
    "backend/arch/eval/analysis.py::Analysis.dominant_weight_scale_dtype::order-decision",
    "backend/arch/eval/analysis.py::Analysis.media_rows::equality-decision",
    # `numerator // source_count` on three declared ints — `values: Dict[int,
    # int]`, `value = int(value_of(raw))`, and two layer-multiplicity counts —
    # with the integrality already refused by name two lines above (`stage
    # count {value} * {kept} / {source_count} is not integral`). So neither
    # half can be an expression and the remainder is gated where the message
    # can name the node. The scan names it because `value_of` is on the
    # algebra's surface and taint follows a call's argument.
    "backend/arch/eval/analysis.py::Analysis.recount::bare-division",
    "backend/arch/eval/analysis.py::Analysis.recount::zero-test",
    "backend/arch/eval/analysis.py::Analysis.recount::zero-test#2",
    # `position_bytes * positions(cache_len) / positions_over`. The two
    # halves cannot both be ints: `position_bytes` is a `float` (2.0 per BF16
    # position, 0.5 per FP4), so `Dim.__mul__` runs it through
    # `_exact_values` and the value half is already a `Rational` — or a law,
    # or a `sp.Integer` — before the division, and `positions_over` is a
    # declared positive `int` (`CachedTensor.__post_init__` refuses <= 0).
    # Measured 1827 times over 17 model/mesh cases: never a Python float.
    # The scan names it because `positions(cache_len, ...)` takes an `Extent`.
    "backend/arch/eval/cache.py::CachedTensor.bytes_at::bare-division",
    "backend/arch/eval/cache.py::_consumer_law.agreed::equality-decision",
    "backend/arch/eval/cache.py::_row_split::except-TypeError",
    "backend/arch/eval/cache.py::positions_at::except-TypeError",
    # Three divisions in the Python-FLOAT display domain, which is where
    # these three functions are declared to answer: `-> Dict[Optional[int],
    # Tuple[float, float]]` twice and a `result_bytes` float on a row dict.
    # Each enters it by an explicit cast — `float(value_of(position_bytes))`,
    # `int(value_of(...))`, `sum(float(...))` — and a cast REFUSES an
    # expression rather than dividing one, so no second domain reaches the
    # operator and there is nothing for the two to disagree about. The scan
    # names them because `value_of` is on the algebra's surface and taint
    # follows a call's argument; neither `int` nor `float` stops it.
    "backend/arch/eval/cache_view.py::CacheView._growth_and_floor_by_block::bare-division",
    "backend/arch/eval/cache_view.py::CacheView._stored_term::bare-division",
    "backend/arch/eval/cache_view.py::CacheView.storage_rows::bare-division",
    "backend/arch/eval/cache_view.py::CacheView._data_formula._leaf_sym::zero-test",
    "backend/arch/eval/cache_view.py::CacheView._data_formula._leaf_val::or-default",
    "backend/arch/eval/cache_view.py::Rate.of::order-decision",
    "backend/arch/eval/cache_view.py::Rate.of::zero-test",
    "backend/arch/eval/cache_view.py::cache_view_from_analysis::order-decision",
    "backend/arch/eval/capacity.py::Weights.add::zero-test",
    "backend/arch/eval/capacity.py::Weights.add::zero-test#2",
    "backend/arch/eval/evaluator.py::Evaluator._eval_primitive::equality-decision",
    "backend/arch/eval/evaluator.py::Evaluator._eval_primitive::equality-decision#2",
    "backend/arch/eval/evaluator.py::Evaluator._eval_primitive::zero-test",
    "backend/arch/eval/evaluator.py::Evaluator._joined::equality-decision",
    "backend/arch/eval/evaluator.py::Evaluator._refuse_view_selecting_an_iteration::zero-test",
    # `expr_of(k) * expr_of(g) * expr_of(batch) * expr_of(seq) / expr_of(e)`
    # builds the EXPRESSION half of `M`, under two guards the scan cannot
    # read: `all(isinstance(v, int) for v in values)` and then
    # `any(isinstance(v, Dim) for v in (batch, seq, k, g, e))`. At least one
    # operand is therefore a `Dim`'s expression — a sympy Symbol — so the
    # quotient is sympy's, and the value half beside it is the exact
    # `sp.Rational(bv * sv * kv * gv, ev)` this docstring is about.
    "backend/arch/eval/evaluator.py::Evaluator._tokens_per_expert::bare-division",
    "backend/arch/eval/evaluator.py::Evaluator._tokens_per_expert::order-decision",
    "backend/arch/eval/evaluator.py::Evaluator._tokens_per_expert::zero-test",
    # `v // degree` where `values = [int(value_of(node_dims[node][dim])) ...]`
    # and `degree = analysis.rule.world_degree(world)`: two plain ints, and
    # the line above has already returned False unless `degree` divides every
    # `v` (`any(v % degree for v in values)`). The `int()` refuses an
    # expression rather than dividing one.
    "backend/arch/eval/fusion.py::GraphCondition.admits_match::bare-division",
    "backend/arch/eval/flops.py::_any_input_voted::order-decision",
    "backend/arch/eval/flops.py::_times_batch::equality-decision",
    "backend/arch/eval/flops.py::bind_labels::equality-decision",
    "backend/arch/eval/flops.py::bind_labels::equality-decision#2",
    "backend/arch/eval/flops.py::bind_labels::order-decision",
    "backend/arch/eval/flops.py::bind_labels::order-decision#2",
    "backend/arch/eval/flops.py::bind_labels::zero-test",
    "backend/arch/eval/flops.py::bind_labels::zero-test#2",
    "backend/arch/eval/flops.py::bind_labels::zero-test#3",
    "backend/arch/eval/flops.py::flops_from_shape::zero-test",
    "backend/arch/eval/flops.py::flops_over_operands::order-decision",
    "backend/arch/eval/flops.py::shape_derived_dims::zero-test",
    "backend/arch/eval/program.py::Program._cut_align::order-decision",
    "backend/arch/eval/projection/result.py::_exact_num::equality-decision",
    "backend/arch/eval/projection/result.py::_exact_num::zero-test",
    "backend/arch/eval/traffic.py::TableRead.scaled::equality-decision",
    "backend/arch/eval/traffic.py::Traffic.scaled::equality-decision",
    # Four truth tests on two Python flags, and nothing goes unchecked for
    # them: the magnitude question around all four is `where`'s — whether a
    # branch's fresh rows fall short of the rows it carries is asked of the
    # values as the cases they are (`where(cache_reads, more, …)` /
    # `where(reads_it, spans_more, …)`) — and these only choose WHICH
    # condition `where` is asked. `gathering` (#1, #4) is a membership in
    # the `gathered` mapping's KEYS beside an `is not None`;
    # `reads_cache_branch` (#2, #3) is `bool()` of a branch's `seeds`,
    # declared `FrozenSet[int]`. So no expression reaches any of them. The
    # scan names them because taint follows the assignment's right side and
    # not the `and` / `bool()` that settles it, and because a container
    # reached through a local name is no longer visibly a container.
    "backend/arch/eval/traffic.py::_activation_reads::zero-test",
    "backend/arch/eval/traffic.py::_activation_reads::zero-test#2",
    "backend/arch/eval/traffic.py::_activation_reads::zero-test#3",
    "backend/arch/eval/traffic.py::_activation_reads::zero-test#4",
    # `not branch.gathered` reads a field declared `bool` on
    # `_EdgeBranchRead`, set from `_activation_reads`'s own flag: whether
    # the evaluator says this reader gathers the branch, which no workload
    # changes. The branch's magnitudes (`elements`, `rows`) are multiplied
    # below, never decided. The scan names it because `branch` is bound by
    # iterating a chain the taint reached, and taint does not stop at a
    # `bool` field.
    "backend/arch/eval/traffic.py::_settle_gathered_live_reads::zero-test",
    "backend/arch/eval/traffic.py::_table_read_fractions::equality-decision",
    "backend/arch/layout/attention.py::attn_res_combine::equality-decision",
    "backend/arch/layout/attention.py::attn_res_combine::equality-decision#2",
    "backend/arch/layout/attention.py::position_table_add::equality-decision",
    # `not _interior_accepts(per_shard, degree, …)` decides whether a Ulysses
    # degree divides the heads one tensor-parallel shard holds, and nothing
    # goes unchecked for it: both operands are declared `int` — `per_shard`
    # through `_heads_on_one_tp_shard` over `_head_axis`'s
    # `int(value_of(extent))`, which refuses an expression rather than
    # deciding, and `degree` is the world's configured size — so no
    # expression reaches the comparison. The scan names it because
    # `Dim(label, extent)` pairs the head axis's identity with that number in
    # the argument, and taint follows a call's arguments.
    "backend/arch/layout/attention_plan.py::head_repartition_kv_cache_region::zero-test",
    # `logical_elements // packing`, both through `_positive_int(...) -> int`
    # (which goes through `operator.index` and raises on anything else, an
    # expression included), with the divisibility refused by name three lines
    # above: "logical elements {n} are not divisible by saved packing factor
    # {p}". That refusal is also what lets `formats.weight_bytes_for` declare
    # the same division exact.
    "backend/arch/layout/copy_checkpoint.py::_checkpoint_payload_fact::bare-division",
    "backend/arch/layout/copy_checkpoint.py::_checkpoint_payload_fact::equality-decision",
    "backend/arch/layout/copy_checkpoint.py::_checkpoint_payload_fact::zero-test",
    "backend/arch/layout/copy_checkpoint.py::_checkpoint_payload_fact::zero-test#2",
    "backend/arch/layout/geometry.py::spatial_merge::order-decision",
    "backend/arch/layout/linear.py::linear_external_weight::equality-decision",
    "backend/arch/layout/linear.py::rope_emb::equality-decision",
    "backend/arch/layout/moe.py::hc_split_sinkhorn::equality-decision",
    "backend/arch/layout/moe.py::hc_split_sinkhorn::equality-decision#2",
    "backend/arch/layout/moe.py::hc_split_sinkhorn::equality-decision#3",
    # `global_rows / preserved_cells`, where `global_rows` is
    # `per_expert * experts` and `per_expert`'s value half is `exact(mean)` —
    # a SymPy number, by the two lines above it, because the routed mean is
    # deliberately kept fractional. A SymPy value divided by the int product
    # of the preserved cuts is exact in either domain (measured 76 times, all
    # `Dim[Expr]`). The scan names it because a `Dim(...)` constructor is not
    # itself a declared SymPy carrier, so the `exact` inside is not followed.
    "backend/arch/layout/moe.py::permute::bare-division",
    "backend/arch/layout/moe.py::permute::zero-test",
    "backend/arch/layout/persistent.py::_count::order-decision",
    "backend/arch/layout/persistent.py::_integer::except-TypeError",
    "backend/arch/layout/persistent.py::active_parameters_by_stack::equality-decision",
    "backend/arch/layout/persistent.py::active_parameters_by_stack::equality-decision#2",
    "backend/arch/layout/physical.py::_validate_kernel_shape::equality-decision",
    "backend/arch/layout/physical_cache.py::_project_state::order-decision",
    "backend/arch/layout/physical_cache.py::_project_state::order-decision#2",
    "backend/arch/layout/physical_cache.py::_project_state::zero-test",
    "backend/arch/layout/physical_cache.py::cache_geometry::equality-decision",
    "backend/arch/layout/physical_cost.py::_project_flops.env_guard.guard::order-decision",
    "backend/arch/layout/propagation.py::LayoutPropagation._finalize_cached_producer::order-decision",
    "backend/arch/layout/propagation.py::LayoutPropagation._finalize_cached_producer::zero-test",
    "backend/arch/layout/recurrent.py::conv1d::equality-decision",
    "backend/arch/layout/recurrent.py::conv1d::equality-decision#2",
    "backend/arch/layout/recurrent.py::conv1d::order-decision",
    "backend/arch/layout/recurrent.py::conv1d::zero-test",
    "backend/arch/layout/recurrent.py::gated_delta_rule::equality-decision",
    "backend/arch/layout/recurrent.py::gated_delta_rule::equality-decision#2",
    "backend/arch/layout/recurrent.py::gated_delta_rule::equality-decision#3",
    "backend/arch/layout/recurrent.py::gated_delta_rule::equality-decision#4",
    # Five `//` on declared ints, in a method whose own signature says so
    # (`term_extent: int`) and whose result is a `Tuple[Tuple[str, int], ...]`.
    # `parts_extent` and every `symbol_extents` value come through
    # `_positive_int(...) -> int`, which goes through `operator.index` and
    # raises on an expression rather than dividing one; `family.group_size`,
    # `num_shards` and `degree` are process-group sizes. Each division is also
    # already gated: `term_extent % parts_extent`, `runtime_local %
    # part_width` and `selected_numerator % parts_extent` each raise above
    # their own quotient, and `group_size // num_shards` divides by
    # construction (`num_shards` is `degree`, which line 651 has just checked
    # equals `group_size`, or a `gcd` with it). The scan names them because
    # `value_of` reaches `_positive_int`'s argument and taint follows a call's
    # arguments.
    "backend/arch/layout/weight_endpoints.py::PlacementContext._placement_layout::bare-division",
    "backend/arch/layout/weight_endpoints.py::PlacementContext._placement_layout::bare-division#2",
    "backend/arch/layout/weight_endpoints.py::PlacementContext._placement_layout::bare-division#3",
    "backend/arch/layout/weight_endpoints.py::PlacementContext._placement_layout::bare-division#4",
    "backend/arch/layout/weight_endpoints.py::PlacementContext._placement_layout::bare-division#5",
    "backend/arch/layout/weight_endpoints.py::PlacementContext._placement_layout::zero-test",
    "backend/arch/layout/weight_endpoints.py::PlacementContext._placement_layout::zero-test#2",
    "backend/arch/layout/weight_endpoints.py::PlacementContext._placement_layout::zero-test#3",
    "backend/arch/layout/weight_endpoints.py::PlacementContext._placement_layout::zero-test#4",
    "backend/arch/layout/weight_endpoints.py::local_symbol_extents::equality-decision",
}

#: Each gate that skips its own check where the extent is an expression,
#: with the check it drops — the inventory, and the gate on the count. The
#: fix for each is the same: carry the question instead of dropping it —
#: `dim.require_multiple` for a divisibility, `dim.require_equal` for an
#: agreement, `dim.require(sp.Le(…))` / `dim.require(sp.Ge(…))` for a
#: bound (rows that fit, a video of at least one frame).
#:
#: One more of the same failure is outside the scan: `ops_registry`'s
#: `_history_contract` tests `value_of(history[2]) == 0` with no `None`
#: guard, so a symbolic history length compares unequal to 0 and the check
#: passes. The scan looks for the guard and cannot see it.
ALLOWED_SKIPS = {
    "backend/arch/eval/cache.py::_of_component::skip":
        "a symbolic cached shape leaves `_row_split` -> (None, None), so the quant-scale grouping (inner channel extent x rows) goes untracked; the scan names this `is not None` because the taint is flow-insensitive",
    "backend/arch/ops_registry.py::_group_items_contract::skip":
        "`None not in (items, per, groups)` -> items == groups x per is not checked",
    "backend/arch/ops_registry.py::_pad_rows_contract::skip":
        "`rows is not None` -> the rows-fit-the-padding check does not run",
    "backend/arch/ops_registry.py::_pad_rows_contract::skip#2":
        "`padded is not None` -> same check, the other operand",
    "backend/arch/ops_registry.py::_patchify_contract::skip":
        "`align is None` -> return: the image side is never checked to be a multiple of the processor's resize factor",
    "backend/arch/ops_registry.py::_patchify_contract::skip#2":
        "`frames is not None` -> the \"a video holds at least one frame\" check does not run",
    "backend/arch/ops_registry.py::_patchify_contract::skip#3":
        "`extent is None` -> continue: this side skips the multiple-of-align check",
    "backend/arch/ops_registry.py::_scatter_rows_contract::skip":
        "`None not in (x[axis], v[axis])` -> x and v are not checked to agree on B and D",
    "backend/arch/ops_registry.py::_scatter_rows_contract::skip#2":
        "`None not in (x[1], *rows)` -> the scattered rows are not checked to fit the sequence",
    "backend/arch/ops_registry.py::_unpad_contract::skip":
        "`None not in (x[0], ref[0])` -> x and the unpadded rows are not checked to agree on items",
    "backend/arch/ops_registry.py::_unpad_contract::skip#2":
        "`None not in (x[1], ref[1])` -> valid rows are not checked to fit the padded rows",
}


class TestNoSecondRoadFromUndecidableToAnswer(unittest.TestCase):
    maxDiff = None

    def test_no_new_place_decides_an_undecidable_magnitude(self) -> None:
        now = set(sb.undecided_decision_sites())
        new = sorted(now - set(ALLOWED_DECISIONS))
        self.assertEqual(
            new, [],
            "a new place decides a magnitude question with a Python operator "
            "on something that may still be an expression. Ask `dim` instead "
            "— `equal` / `require_equal` for sameness, `larger` / `smaller` "
            "for a span, `is_whole` / `floored` for an index, `where` for a "
            "value that differs between points, `require` for a condition the "
            "point must meet — each of which answers with a formula, records "
            "the condition, or refuses. If the site genuinely cannot "
            "receive an expression, add it here with that reason: "
            f"{new}")

    def test_the_allowance_is_not_stale(self) -> None:
        now = set(sb.undecided_decision_sites())
        gone = sorted(set(ALLOWED_DECISIONS) - now)
        self.assertEqual(
            gone, [],
            "these decisions are gone — good. Delete them from "
            "ALLOWED_DECISIONS in the same commit (and lower the "
            "`undecided_decisions` baseline with `python -m tools.scoreboard "
            f"--rebaseline --write`), or the slack hides the next one: {gone}")


class TestTheSymbolicDomainKeepsGating(unittest.TestCase):
    maxDiff = None

    def test_no_new_gate_skips_itself_on_a_symbol(self) -> None:
        now = set(sb.symbol_skipping_gate_sites())
        new = sorted(now - set(ALLOWED_SKIPS))
        self.assertEqual(
            new, [],
            "a new check skips itself where the extent is an expression. A "
            "gate that stops gating is worse than no gate: the walk reports "
            "the contract clean. Carry the question instead of dropping it "
            "(`dim.require_multiple`, `dim.require_equal`, or "
            "`dim.require(sp.Le/Ge(...))` for a bound, record it and "
            f"refuse at a point that breaks it): {new}")

    def test_the_skip_inventory_is_not_stale(self) -> None:
        now = set(sb.symbol_skipping_gate_sites())
        gone = sorted(set(ALLOWED_SKIPS) - now)
        self.assertEqual(
            gone, [],
            "these gates no longer skip — good. Delete them from "
            "ALLOWED_SKIPS and lower the `symbol_skipping_gates` baseline in "
            f"the same commit: {gone}")

    def test_every_skip_says_what_it_drops(self) -> None:
        """An exemption is a judgement, not a comment: each skip names the
        check that does not run, so the inventory and the gate cannot
        drift."""
        unexplained = sorted(k for k, why in ALLOWED_SKIPS.items()
                             if not why or len(why) < 20)
        self.assertEqual(unexplained, [])


if __name__ == "__main__":
    unittest.main()
