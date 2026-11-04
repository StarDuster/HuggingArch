"""One walk with the batch and the length symbolic is every numeric walk.

The batch `B` and the prompt length are bound to positive integer symbols, and
the deployment is walked once. Every number a reader takes off that walk is an
expression in those two symbols: the global shapes and costs, the layout
states and collectives, and the per-rank projections (FLOPs, access and
kernel shapes, cost, HBM traffic, cache). Substituted at a point, each one
must be exactly the number the deployment at that point gives
(`Deployment.analyze`, which walks every count as a symbol and reads the walk
there): a walk over some of the counts and the walk over all of them agree. A
point the deployment refuses must be refused by `Analysis.admit` with the
same error.

No tolerance. A float is compared as the exact dyadic rational it is.

The walk needs no point: it decides nothing about the workload
(`test_walk_decides_nothing`), and the only refusals at a point are the
conditions its formulas hold under (`Analysis.admit`), each the error a
numeric walk there raises.
"""
from __future__ import annotations

import dataclasses
import enum
import sys
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.dim import Dim, Undecided, value_of
from backend.arch.eval.layer import Layer
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.parallel_config import ParallelConfig
from backend.arch.layout.physical import (physical_access_shapes,
                                          physical_kernel_shapes)
from backend.arch.layout.physical_cache import cache_geometry
from backend.arch.layout.physical_cost import (physical_cost_plan,
                                               physical_flops,
                                               physical_traffic)
from tests.snapshots import own_walk, resolve_snapshot, shared_program


def _walked(deployment):
    """A walk of this file's own, never the module's shared one.

    Every analysis here is read through `_readers`, which opens
    `Analysis.assumed()` to derive the physical layers under the walk's
    conditions -- and that scope WRITES the assumption ledger back to the
    analysis when it closes. A shared analysis would therefore reach the
    next test in this module carrying this test's assumptions, so the
    reason is stated once here rather than at each of the twelve calls."""
    return own_walk(deployment, why=(
        "_readers opens Analysis.assumed(), which writes the assumption "
        "ledger back to the analysis when the scope closes"))


def _symbol(name):
    """A workload input as the walk is given it: a positive whole number."""
    return sp.Symbol(name, positive=True, integer=True)


#: One symbol per INPUT of `Workload` a sweep varies, each in its own right.
#: They are never shared: `Phase.resolve` turns them into the engine's
#: sequence symbols through different laws (`s_query` is `input_tokens -
#: s_prefix_cached` in prefill and `s_verify` in decode; `s_kv` is the cache
#: plus `s_verify - 1`), and two inputs behind one symbol would hide a law
#: that reads the wrong one.
BATCH = _symbol("B")
LENGTH = _symbol("s_ctx")
PREFIX = _symbol("s_prefix")
VERIFY = _symbol("s_verify")
OUTPUT = _symbol("s_out")

BATCHES = (1, 8, 64)
CONTEXTS = (1024, 4096, 65536, 131072)

#: Fields that point back into the program or the walk's machinery rather
#: than carrying a number of the walk.
_STRUCTURE = frozenset({"program", "node", "rule", "binding", "precision",
                        "walked", "input_references", "group", "groups",
                        "scope", "refusal"})


def _number(value, point):
    if isinstance(value, sp.Basic):
        if value.free_symbols:
            value = sp.sympify(value.xreplace(
                {s: sp.Integer(point[s.name]) for s in value.free_symbols
                 if s.name in point}))
        if value.free_symbols:
            return ("unbound", str(value))
        if value in (sp.true, sp.false):
            # A condition read at a point is that point's truth.
            return bool(value)
        return sp.Rational(value) if value.is_Number else value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, Fraction, float)):
        return sp.Rational(value)
    return value


def _read(obj, point, depth=0):
    """`obj` with every extent read at `point`, as plain comparable data."""
    if depth > 12:
        return "..."
    if isinstance(obj, Dim):
        return _number(obj.value, point)
    if isinstance(obj, (sp.Basic, int, float, Fraction)) \
            and not isinstance(obj, bool):
        return _number(obj, point)
    if obj is None or isinstance(obj, (str, bool, enum.Enum)):
        return obj
    if isinstance(obj, Layer):
        return {uid: _read(v, point, depth + 1)
                for uid, v in obj.values.items()}
    if dataclasses.is_dataclass(obj):
        return (type(obj).__name__,
                {f.name: _read(getattr(obj, f.name), point, depth + 1)
                 for f in dataclasses.fields(obj)
                 if f.name not in _STRUCTURE})
    if isinstance(obj, dict):
        return {repr(_read(k, point, depth + 1)): _read(v, point, depth + 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_read(x, point, depth + 1) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(repr(_read(x, point, depth + 1)) for x in obj)
    return repr(obj)


def _readers(analysis):
    """Every layer a reader takes numbers from, the walk's and per rank."""
    out = {
        "shapes": analysis.shapes, "cost": analysis.cost,
        "kernel_shapes": analysis.kernel_shapes,
        "instances": analysis.instances,
        "active_share": analysis.active_share,
        "states": analysis.states,
        "redistributes": analysis.redistributes,
        # The per-node dims, which is where the narrowed sequence symbols
        # live: `s_attend` (what one query attends, `Min(s_kv, s_window)`)
        # and the `s_kv` a windowed or top-k law narrowed to.
        "node_dims": analysis.node_dims,
        "cache_positions": analysis.cache_positions,
    }
    with analysis.assumed():
        out["physical_flops"] = physical_flops(analysis)
        out["physical_access_shapes"] = physical_access_shapes(analysis)
        out["physical_kernel_shapes"] = physical_kernel_shapes(analysis)
        out["physical_cost"] = physical_cost_plan(analysis).cost
        out["physical_traffic"] = physical_traffic(analysis)
        out["cache"] = cache_mod.derive(
            analysis, geometry=cache_geometry(analysis))
        out["kv_access"] = cache_mod.kv_access(analysis)
    return out


def _deployment(program, mesh, workload, phase="prefill", cache_len=None):
    """A deployment at `workload` (the `Workload` inputs, symbolic or not)."""
    deployment = Deployment(program).at(**workload)
    if phase == "decode":
        deployment = deployment.during("decode", cache_len=cache_len)
    if mesh:
        deployment = replace(deployment,
                             config=ParallelConfig.from_dict(mesh))
    return deployment


def _at(value, point):
    """`value` with the point's symbols bound — a workload input read at one
    point of the sweep."""
    if isinstance(value, sp.Basic):
        bound = sp.sympify(value.xreplace(
            {s: sp.Integer(point[s.name]) for s in value.free_symbols
             if s.name in point}))
        return int(bound) if bound.is_Integer else bound
    return value


def _grid(axes):
    """Every point of `{symbol name: values}`, in a stable order."""
    names = sorted(axes)
    points = [{}]
    for name in names:
        points = [{**p, name: v} for p in points for v in axes[name]]
    return points


def _differences(left, right, path, out):
    if len(out) >= 5:
        return
    if type(left) is not type(right):
        out.append((path, left, right))
    elif isinstance(left, dict):
        for key in left.keys() | right.keys():
            if key not in left or key not in right:
                out.append(([*path, key], left.get(key), right.get(key)))
            else:
                _differences(left[key], right[key], [*path, key], out)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            out.append((path, left, right))
        else:
            for i, (a, b) in enumerate(zip(left, right, strict=True)):
                _differences(a, b, [*path, i], out)
    elif left != right:
        out.append((path, left, right))


class _Equivalence(unittest.TestCase):
    def one_walk(self, model, mesh, phase="prefill", *, workload=None,
                 cache_len=None):
        """The symbolic walk."""
        program = shared_program(model)
        workload = {"batch": BATCH, "input_tokens": LENGTH} \
            if workload is None else workload
        return program, _deployment(program, mesh, workload, phase, cache_len)

    def assert_one_walk_is_every_walk(
            self, model, mesh, phase="prefill", *, workload=None,
            cache_len=None, axes=None, batches=BATCHES, contexts=CONTEXTS):
        """One symbolic walk, read at every point of `axes`.

        `workload` says which `Workload` inputs are symbolic and `axes` what
        each symbol runs over; the default is the batch and the prompt
        length over `batches x contexts`.
        """
        if workload is None:
            workload = {"batch": BATCH, "input_tokens": LENGTH}
            if phase == "decode":
                cache_len = LENGTH
            axes = {"B": batches, "s_ctx": contexts}
        program, deployment = self.one_walk(
            model, mesh, phase, workload=workload, cache_len=cache_len)
        analysis = _walked(deployment)
        law = _readers(analysis)
        for point in _grid(axes):
            with self.subTest(model=model, mesh=mesh, phase=phase,
                              point=point):
                self.assert_point(program, mesh, phase, workload, cache_len,
                                  analysis, law, point)
        return analysis

    def assert_point(self, program, mesh, phase, workload, cache_len,
                     analysis, law, point):
        """Compare at `point`, or hold the refusal to the numeric walk's."""
        own = {name: _at(value, point) for name, value in workload.items()}
        try:
            walked = _readers(_walked(_deployment(
                program, mesh, own, phase, _at(cache_len, point))))
            refused = None
        except ValueError as exc:
            refused = exc
        try:
            analysis.admit(point)
            admitted = True
        except ValueError as exc:
            admitted = False
            self.assertIsNotNone(
                refused, f"{point} was refused ({exc}) and a numeric walk "
                f"there succeeds")
            self.assertIs(type(exc), type(refused))
            self.assertEqual(str(exc), str(refused))
        if not admitted:
            return
        self.assertIsNone(refused,
                          f"{point} was admitted and the numeric walk "
                          f"refuses it: {refused}")
        for name, layer in law.items():
            found = []
            _differences(_read(layer, point), _read(walked[name], point),
                         [name], found)
            self.assertEqual(found, [], f"{name} at {point}")


class TestTheCacheReadIsAFormula(_Equivalence):
    """A decode step's cache read is billed where the reader spans more than
    its producer made — `1 < s_cache`, false at a one-position cache. That
    answer differs between points, so it is written as the cases, and one
    walk with no point to settle at answers both sides of it.
    """

    CASE = ("meta-llama/Llama-3.2-1B", {}, "decode")

    def test_no_point_is_needed_and_none_is_taken(self):
        model, mesh, phase = self.CASE
        program = shared_program(model)
        analysis = _walked(_deployment(
            program, mesh, {"batch": BATCH, "input_tokens": LENGTH}, phase,
            LENGTH))
        _readers(analysis)
        self.assertEqual(analysis.assumptions, ())

    def test_one_walk_reads_both_sides(self):
        model, mesh, phase = self.CASE
        self.assert_one_walk_is_every_walk(model, mesh, phase,
                                           batches=(1, 8),
                                           contexts=(1, 2, 1024))

class TestAPointKeepsItsOwnReadings(unittest.TestCase):
    """What readings at one point have read belongs to that point
    (`analysis.Point`): a walk read twice there is read once, and another
    point — another number, or the same numbers over names — reads its
    own."""

    def test_a_point_reads_once_and_only_its_own(self):
        from backend.arch.eval.analysis import Point
        analysis = _walked(_deployment(
            shared_program("meta-llama/Llama-3.2-1B"), {},
            {"batch": BATCH, "input_tokens": 64}, "prefill"))
        one = Point({"B": 1})
        self.assertEqual(one.read(BATCH * 3), 3)
        self.assertIs(analysis.at(one), analysis.at(one))
        self.assertEqual(Point({"B": 2}).read(BATCH * 3), 6)
        self.assertIsNot(analysis.at(Point({"B": 1})), analysis.at(one))


class TestAPointIsInTheCountsDomain(unittest.TestCase):
    """The walk's laws hold where its counts are declared: the batch and the
    verify width at least one, the prompt, the output and the cached prefix
    zero or more. A point outside that domain has no walk, and a law read
    there would print a number no step has (a negative batch's FLOPs)."""

    #: Every count outside its field's declared domain, by what it breaks.
    OUTSIDE = (({"batch": -2}, "counts at least one"),
               ({"batch": 0}, "counts at least one"),
               ({"input_tokens": -1}, "counts zero or more"),
               ({"output_tokens": -1}, "counts zero or more"),
               ({"s_prefix_cached": -1}, "counts zero or more"),
               ({"s_verify": 0}, "counts at least one"),
               ({"batch": 2.5}, "counts in whole numbers"),
               ({"input_tokens": 1.5}, "counts in whole numbers"),
               ({"ep_load_factor": 0.5}, "is at least 1"))

    def test_a_count_below_its_domain_is_refused(self):
        from backend.arch.eval.transforms import Workload
        for counts, _reason in self.OUTSIDE:
            with self.subTest(**counts), self.assertRaises(ValueError):
                Workload(**counts).symbolic()

    def test_a_count_outside_its_domain_is_refused_at_the_door(self):
        """And refused by CONSTRUCTING it, which is the numeric half of the
        same declaration: `Phase.resolve` reads a workload nobody
        symbolised (`phases.phase_binding`), so a door only the symbolic
        path passes through is no door — a verify width of 0 used to reach
        the step and be rewritten to 1 by a `maximum` the declaration had
        already folded away for the symbolic walk."""
        from backend.arch.eval.transforms import Workload
        for counts, reason in self.OUTSIDE:
            with self.subTest(**counts):
                with self.assertRaises(ValueError) as caught:
                    Workload(**counts)
                said = str(caught.exception)
                self.assertIn(next(iter(counts)), said)   # names the field
                self.assertIn(reason, said)               # and its domain

    def test_a_step_at_a_negative_batch_or_prompt_is_refused(self):
        from backend.inference.parallel_config import ParallelConfig as PC
        from backend.inference.step_eval import evaluate_step
        spec = resolve_snapshot("meta-llama/Llama-3.2-1B")
        for batch, prompt in ((-2, 1), (1, -1)):
            with self.subTest(batch=batch, prompt=prompt), \
                    self.assertRaises(ValueError):
                _ = evaluate_step(spec, PC(), phase="prefill", batch=batch,
                                  input_tokens=prompt,
                                  output_tokens=0).analysis


class TestEverySweptWorkloadInputGoesSymbolic(_Equivalence):
    """`Workload` has more than one swept input, and "sequence" is not one
    symbol.

    `Phase.resolve` turns the inputs into the engine's sequence symbols
    through different laws — prefill `s_query = input_tokens -
    s_prefix_cached`, decode `s_query = s_verify` and `s_kv = cache +
    s_verify - 1`, a cache the caller did not state being `input_tokens +
    output_tokens/2`. Each input is symbolised in its own right here, so a
    law that reads the wrong one has somewhere to show up: behind one shared
    symbol every one of those reads the same number.
    """

    def test_prefill_prefix_cached(self):
        """`s_query = input_tokens - s_prefix_cached`, `s_past = prefix`."""
        self.assert_one_walk_is_every_walk(
            "meta-llama/Llama-3.2-1B", {}, "prefill",
            workload={"batch": 1, "input_tokens": 4096,
                      "s_prefix_cached": PREFIX},
            axes={"s_prefix": (512, 1024, 2048)})

    def test_decode_output_tokens(self):
        """The cache a decode step reads when the caller states none:
        `input_tokens + output_tokens/2`, rounded."""
        self.assert_one_walk_is_every_walk(
            "meta-llama/Llama-3.2-1B", {}, "decode",
            workload={"batch": 1, "input_tokens": 4096,
                      "output_tokens": OUTPUT},
            cache_len=None, axes={"s_out": (128, 512, 2049)})

    def test_decode_speculative_verify_width(self):
        """`s_query = n_verify` and `s_kv = cache + n_verify - 1`: wrong
        here is invisible at `s_verify = 1` and wrong at every other
        width. DeepSeek-V3 ships the MTP stack that serves it."""
        self.assert_one_walk_is_every_walk(
            "deepseek-ai/DeepSeek-V3", {}, "decode",
            workload={"batch": 1, "input_tokens": 4096, "s_verify": VERIFY},
            cache_len=4096, axes={"s_verify": (2, 4, 8)})

    def test_the_batch_the_cache_and_the_verify_width_at_once(self):
        """Three inputs symbolic in one walk, read over their product."""
        self.assert_one_walk_is_every_walk(
            "deepseek-ai/DeepSeek-V3", {"tp_attention": 8, "ep_moe": 8},
            "decode",
            workload={"batch": BATCH, "input_tokens": LENGTH,
                      "s_verify": VERIFY},
            cache_len=LENGTH,
            axes={"B": (1, 8), "s_ctx": (1024, 65536), "s_verify": (2, 4)})


class TestAnUndecidableConditionIsNeverAnswered(unittest.TestCase):
    """A question about the workload has no answer in the walk: a value
    that differs between points is the cases it is, and a condition on the
    point is recorded with the site's own refusal."""

    def test_two_incomparable_spans_keep_both(self):
        """The widest of two cache spans needs no decision.

        `traffic._wider` used to catch the `TypeError` of an undecidable
        `b > a` and keep `a` — the same silent pick as the `kv_reads = 0`
        fallback, on the same path. Where neither span is wider everywhere,
        both are kept and the point resolves them.
        """
        from backend.arch.eval.traffic import _wider

        batch = _symbol("B")
        length = _symbol("s_ctx")
        wider = _wider(64 * batch, 8 * length)
        for point, expect in (({"B": 1, "s_ctx": 1}, 64),
                              ({"B": 1, "s_ctx": 64}, 512),
                              ({"B": 64, "s_ctx": 1}, 4096)):
            with self.subTest(point=point):
                self.assertEqual(_read(wider, point),
                                 max(64 * point["B"], 8 * point["s_ctx"]))
                self.assertEqual(_read(wider, point), expect)

    def test_a_validation_question_keeps_its_own_refusal(self):
        """A requirement is recorded, not decided: `require_equal` carries
        the call site's exception, so a point that breaks it is refused as
        a walk there refuses it. The same comparison asked as a question of
        two laws (`equal`) has no answer."""
        from backend.arch.eval.dim import assuming, equal, require_equal

        length = _symbol("s_ctx")

        class Wrong(ValueError):
            pass

        with assuming() as ledger:
            require_equal(length, 8, lambda at: Wrong(f"not 8: {at(length)}"))
            (held,) = ledger.held
        held.at({"s_ctx": 8})
        with self.assertRaises(Wrong):
            held.at({"s_ctx": 9})
        with self.assertRaises(Undecided):
            equal(length, 8)


class TestCountsThatMayBeZero(_Equivalence):
    """The tokens a request brings in and takes out are counts, and zero is
    one of them — `Workload` admits `input_tokens=0` and `output_tokens=0` —
    so their symbols are declared non-negative. That puts a `Max(1, ·)` and
    a rounding into the decode cache length (`max(1, nearest(input +
    output/2))`); one walk reads every point of it, the zeros included."""

    def test_decode_with_tokens_that_may_be_zero(self):
        """DeepSeek-V3 decode under dp_attention=8 x ep_moe=8, the prompt
        and the output each a count that may be zero: every point equals
        the numeric walk there."""
        prompt = sp.Symbol("s_input", integer=True, nonnegative=True)
        output = sp.Symbol("s_output", integer=True, nonnegative=True)
        self.assert_one_walk_is_every_walk(
            "deepseek-ai/DeepSeek-V3", {"dp_attention": 8, "ep_moe": 8},
            "decode",
            workload={"batch": BATCH, "input_tokens": prompt,
                      "output_tokens": output, "s_verify": VERIFY},
            axes={"B": (1, 64), "s_input": (0, 1024), "s_output": (0, 257),
                  "s_verify": (1, 4)})

    def test_prefill_with_a_prompt_and_a_prefix_that_may_be_zero(self):
        prompt = sp.Symbol("s_input", integer=True, nonnegative=True)
        prefix = sp.Symbol("s_prefix", integer=True, nonnegative=True)
        self.assert_one_walk_is_every_walk(
            "meta-llama/Llama-3.2-1B", {}, "prefill",
            workload={"batch": BATCH, "input_tokens": prompt,
                      "s_prefix_cached": prefix},
            axes={"B": (1, 8), "s_input": (0, 1024, 4096),
                  "s_prefix": (0, 1024)})


class _StructuralComparisons:
    """Every place backend code compares a symbolic extent by its shape —
    `extent == 0`, `extent != 1`, `if expression:` — while this is open.

    Each is a road from an undecidable condition to a bool that leaves no
    record: an expression is never the number 0 and is always truthy as a
    Python object, so the walk takes the non-empty road even at the points
    where the extent is zero. The decisions are `dim.is_zero`, `dim.equal`,
    `dim.less`, `dim.is_unit` (and a `Dim`'s own truth, which is
    `is_zero`'s); a comparison written any other way shows up here.

    Recorded where backend code asked, never inside SymPy (a SymPy routine
    compares its own expressions structurally, and that is SymPy's business)
    and never inside `dim` itself, which is where the decisions are made.
    """

    _BACKEND = str(PROJECT_ROOT / "backend") + "/"
    #: Where the decisions are made: it compares expressions as expressions
    #: (has `simplify` changed this one, is this the question already
    #: asked), which is what deciding over the points is built from.
    _DECISIONS = str(PROJECT_ROOT / "backend" / "arch" / "eval" / "dim.py")
    #: Reading a walk at a point (`Analysis.at`) looks each expression up
    #: among the ones it already read — an expression compared as an
    #: expression, by the reader, not by the walk.
    _READING = (str(PROJECT_ROOT / "backend" / "arch" / "eval" / "analysis.py"),
                frozenset({"_number", "_identity"}))

    def __init__(self):
        self.sites = {}
        self._inside = False

    @staticmethod
    def _counts(expr):
        return any(symbol.is_integer and symbol.is_nonnegative
                   for symbol in getattr(expr, "free_symbols", ()))

    def _note(self, kind):
        """Record the comparison `kind()` describes, where backend code
        asked it. Printing an expression compares inside SymPy, so the
        description is rendered with the hooks quiet."""
        frame = sys._getframe(2)
        where = frame.f_code.co_filename
        if not where.startswith(self._BACKEND) or where == self._DECISIONS:
            return
        if (where == self._READING[0]
                and frame.f_code.co_name in self._READING[1]):
            return
        self._inside = True
        try:
            key = (f"{where[len(self._BACKEND):]}:{frame.f_lineno} "
                   f"{frame.f_code.co_name}: {kind()}")
        finally:
            self._inside = False
        self.sites[key] = self.sites.get(key, 0) + 1

    def __enter__(self):
        from sympy.core.basic import Basic

        seen = self
        self._saved = (Basic.__eq__, Basic.__ne__, sp.Expr.__eq__,
                       sp.Expr.__ne__, Dim.__eq__, Dim.__ne__)
        basic_eq, basic_ne, _, _, dim_eq, dim_ne = self._saved

        def extent(other):
            return isinstance(other, (int, float, Fraction)) or (
                isinstance(other, sp.Basic)
                and (other.is_Number or seen._counts(other)))

        def asked(left, right):
            # Asking SymPy whether a symbol is a count compares inside
            # SymPy; only the outermost comparison is backend code's.
            if seen._inside:
                return False
            seen._inside = True
            try:
                return seen._counts(left) and extent(right)
            finally:
                seen._inside = False

        def eq(left, right):
            if asked(left, right):
                seen._note(lambda: f"{left} == {right}")
            return basic_eq(left, right)

        def ne(left, right):
            if asked(left, right):
                seen._note(lambda: f"{left} != {right}")
            return basic_ne(left, right)

        def counted(expr):
            if seen._inside:
                return False
            seen._inside = True
            try:
                return seen._counts(expr)
            finally:
                seen._inside = False

        def truth(expr):
            if counted(expr):
                seen._note(lambda: f"bool({expr})")
            return True

        def symbolic(extent):
            return isinstance(extent, Dim) and counted(extent.value)

        def dim_eq_(left, right):
            if symbolic(left) or symbolic(right):
                seen._note(lambda: f"{left!r} == {right!r}")
            return dim_eq(left, right)

        def dim_ne_(left, right):
            if symbolic(left) or symbolic(right):
                seen._note(lambda: f"{left!r} != {right!r}")
            return dim_ne(left, right)

        Basic.__eq__, Basic.__ne__ = eq, ne
        sp.Expr.__eq__, sp.Expr.__ne__ = eq, ne
        sp.Expr.__bool__ = truth
        Dim.__eq__, Dim.__ne__ = dim_eq_, dim_ne_
        return self

    def __exit__(self, *exc):
        from sympy.core.basic import Basic

        (Basic.__eq__, Basic.__ne__, sp.Expr.__eq__, sp.Expr.__ne__,
         Dim.__eq__, Dim.__ne__) = self._saved
        del sp.Expr.__bool__
        return False


class TestAnExtentIsNeverComparedByItsShape(_Equivalence):
    """A zero extent is a fact a walk decides, not a shape it reads.

    Two places this read the shape and got the point wrong, each found by
    walking with the prompt and the prefix allowed to be zero and the prompt
    fully cached (Qwen3.8-Flash-Next prefill at 8 tokens, all 8 cached): a
    concat of two pooled keys, both `floor(s_query/4)` long and so both
    empty, was counted as two non-empty operands (`_elements(dims) != 0`)
    and billed edge reads a numeric walk there does not have; and a
    compressor's active fraction divided its live pairs by an empty
    `s_query x s_kv` matrix (`cells != 0`) and came out NaN where the
    numeric walk says one. Both now ask `dim.is_zero`.
    """

    @pytest.mark.regression
    def test_a_prompt_that_is_all_cached_or_empty(self):
        prompt = sp.Symbol("s_input", integer=True, nonnegative=True)
        prefix = sp.Symbol("s_prefix", integer=True, nonnegative=True)
        workload = {"batch": BATCH, "input_tokens": prompt,
                    "s_prefix_cached": prefix}
        program = shared_program("Qwen/Qwen3.8-Flash-Next")
        analysis = _walked(_deployment(program, {}, workload, "prefill",
                               None))
        law = _readers(analysis)
        for point in ({"B": 1, "s_input": 4096, "s_prefix": 0},
                      {"B": 1, "s_input": 4096, "s_prefix": 1024},
                      {"B": 1, "s_input": 8, "s_prefix": 8},
                      {"B": 2, "s_input": 0, "s_prefix": 0}):
            with self.subTest(point=point):
                self.assert_point(program, {}, "prefill", workload, None,
                                  analysis, law, point)

    def test_the_truth_of_a_symbolic_extent_is_refused(self):
        """`if extent:` on a law in the workload's symbols has no answer,
        and neither has `is_zero` of one: whether a tensor is empty can
        differ between points, and the walk does not decide it — an empty
        tensor is a zero value of the law (`where`, `ratio`). A number
        answers as itself."""
        from backend.arch.eval.dim import Undecided, is_zero

        length = _symbol("s_ctx")
        pooled = Dim(sp.Symbol("n_pool"), sp.floor(length / 4))
        self.assertTrue(Dim(sp.Symbol("x"), 3))
        self.assertFalse(Dim(sp.Symbol("x"), 0))
        for extent in (Dim(sp.Symbol("x"), BATCH * length), pooled):
            with self.subTest(extent=repr(extent)):
                with self.assertRaises(Undecided):
                    bool(extent)
                with self.assertRaises(Undecided):
                    is_zero(extent)
        self.assertTrue(is_zero(Dim(sp.Symbol("x"), 0)))

    def test_no_backend_code_compares_a_symbolic_extent_by_its_shape(self):
        self.assert_no_structural_comparison(self._cases())

    @pytest.mark.regression
    def test_no_structural_comparison_across_the_families(self):
        """A compressed pool under a sequence cut — the one place a slice's
        emptiness is asked of a symbolic extent — and the hybrid, windowed,
        top-k and routed families."""
        prefill, prefill_at, decode, decode_at = self._workloads()
        length = _symbol("s_ctx")
        self.assert_no_structural_comparison((
            *self._cases(),
            ("deepseek-ai/DeepSeek-V4-Flash", {"cp": 8}, "prefill",
             {"batch": BATCH, "input_tokens": length},
             {"B": 1, "s_ctx": 4096}),
            ("deepseek-ai/DeepSeek-V3", {"dp_attention": 8, "ep_moe": 8},
             "decode", decode, decode_at),
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", {}, "prefill",
             prefill, prefill_at),
            ("google/gemma-2-9b", {}, "decode", decode, decode_at),
            ("moonshotai/Kimi-K3", {"dp_attention": 8, "ep_moe": 8},
             "prefill", prefill, prefill_at),
        ))

    @staticmethod
    def _workloads():
        """Prefill with a cached prefix and decode with an output and a
        verify width, every token count allowed to be zero."""
        prompt = sp.Symbol("s_input", integer=True, nonnegative=True)
        prefill = {"batch": BATCH, "input_tokens": prompt,
                   "s_prefix_cached": sp.Symbol("s_prefix", integer=True,
                                                nonnegative=True)}
        decode = {"batch": BATCH, "input_tokens": prompt,
                  "output_tokens": sp.Symbol("s_output", integer=True,
                                             nonnegative=True),
                  "s_verify": VERIFY}
        return (prefill, {"B": 1, "s_input": 4096, "s_prefix": 1024},
                decode, {"B": 1, "s_input": 4096, "s_output": 256,
                         "s_verify": 2})

    def _cases(self):
        prefill, prefill_at, decode, decode_at = self._workloads()
        return (
            ("meta-llama/Llama-3.2-1B", {"tp_attention": 8}, "prefill",
             prefill, prefill_at),
            ("meta-llama/Llama-3.2-1B", {"cp": 8}, "prefill",
             {"batch": BATCH, "input_tokens": prefill["input_tokens"]},
             {"B": 1, "s_input": 4096}),
            ("Qwen/Qwen3.8-Flash-Next", {}, "decode", decode, decode_at),
        )

    def assert_no_structural_comparison(self, cases):
        for model, mesh, phase, workload, _point in cases:
            program = shared_program(model)
            with self.subTest(model=model, mesh=mesh, phase=phase), \
                    _StructuralComparisons() as found:
                analysis = _walked(_deployment(program, mesh, workload, phase,
                                       None))
                _readers(analysis)
            self.assertEqual(found.sites, {})


class TestNoRouteInSkipsTheDeclarationDoor(unittest.TestCase):
    """A symbol a walk is asked to reason about must say it is a whole
    count, and there must be no way in that skips saying so.

    The door is on the ASSEMBLED binding (`transforms._declared`), not on
    each route, because the routes are several and one per route is one per
    route someone forgets: `at` writes the workload, `during` the cache
    length, `with_dims` a run-scoped overlay applied AFTER the phase, the
    encode branch binds an encoder's media axes and no workload at all, and
    the mesh adds layout scalars. The first version of this check guarded
    `Workload` only, and `with_dims(s_ctx=Symbol('S', integer=True))` walked
    straight past it into `physical_axis_partition_audit`'s `extent <= 0`.

    So this test is not "the door exists" — it is one case per route that
    can install a dim, and the door is placed where they all arrive.
    """

    UNDECLARED = sp.Symbol("Undeclared")
    #: Integral but unsigned: `Mod(x, n) = 0` would be read as `n t`, and
    #: without a lower bound on `x` there is no lower bound on `t`.
    UNSIGNED = sp.Symbol("Unsigned", integer=True)

    def program(self):
        return shared_program("meta-llama/Llama-3.2-1B")

    def test_every_route_that_binds_a_dim_refuses_an_undeclared_symbol(self):
        program = self.program()
        for symbol in (self.UNDECLARED, self.UNSIGNED):
            routes = {
                "at(batch=)": lambda s=symbol: Deployment(program).at(
                    batch=s, input_tokens=1024),
                "at(input_tokens=)": lambda s=symbol: Deployment(program).at(
                    batch=1, input_tokens=s),
                "at(output_tokens=)": lambda s=symbol: Deployment(
                    program).at(batch=1, input_tokens=1024,
                                output_tokens=s).during("decode"),
                "at(s_prefix_cached=)": lambda s=symbol: Deployment(
                    program).at(batch=1, input_tokens=1024,
                                s_prefix_cached=s),
                "at(s_verify=)": lambda s=symbol: Deployment(program).at(
                    batch=1, input_tokens=1024, s_verify=s).during("decode"),
                "during(cache_len=)": lambda s=symbol: Deployment(
                    program).at(batch=1, input_tokens=1024).during(
                        "decode", cache_len=s),
                "with_dims(a step symbol)": lambda s=symbol: Deployment(
                    program).at(batch=1, input_tokens=1024).with_dims(
                        s_ctx=s),
                "with_dims(a model dim)": lambda s=symbol: Deployment(
                    program).at(batch=1, input_tokens=1024).with_dims(
                        n_heads=s),
            }
            for name, route in routes.items():
                with self.subTest(route=name, symbol=str(symbol)):
                    with self.assertRaises(ValueError) as caught:
                        _walked(route())
                    # The same message everywhere, naming the declaration it
                    # wants, because a caller that hits one of these has the
                    # same thing to fix whichever door it was.
                    self.assertIn("declare", str(caught.exception))
                    self.assertIn("integer=True", str(caught.exception))

    def test_an_encode_walk_refuses_one_too(self):
        """The encode branch binds an encoder's media axes and never touches
        the workload, so it used to reach the walk with anything."""
        for model in ("Qwen/Qwen3.5-35B-A3B",):
            program = shared_program(model)
            media = {"pixel_values": {"n_img": 1, "img_h": 672,
                                      "img_w": 672}}
            encode = Deployment(program).at(
                batch=1, input_tokens=64, media=media).encoders()[0]
            with self.assertRaises(ValueError) as caught:
                _walked(encode.with_dims(img_h=self.UNDECLARED))
            self.assertIn("declare", str(caught.exception))
            # A media axis cannot be a symbol by any route: `media_values`
            # takes a whole number, which is why the patchify contract's
            # symbolic skip is not reachable through media.
            with self.assertRaises(ValueError) as caught:
                _walked(Deployment(program).at(
                    batch=1, input_tokens=64,
                    media={"pixel_values": {"n_img": self.UNSIGNED,
                                            "img_h": 672, "img_w": 672}},
                ).encoders()[0])
            self.assertIn("whole number", str(caught.exception))

    def test_a_declared_symbol_is_let_through_every_route(self):
        """The door refuses a declaration, not a symbol."""
        program = self.program()
        length = _symbol("s_ctx")
        for name, deployment in (
            ("at", Deployment(program).at(batch=1, input_tokens=length)),
            ("with_dims", Deployment(program).at(
                batch=1, input_tokens=1024).with_dims(s_ctx=length)),
            ("during", Deployment(program).at(
                batch=1, input_tokens=1024).during("decode",
                                                   cache_len=length)),
        ):
            with self.subTest(route=name):
                analysis = _walked(deployment)
                self.assertIsNotNone(analysis.binding)


class TestTheDecodeCacheRoundsHalvesUp(unittest.TestCase):
    """The cache a decode step reads, when the caller states none, is
    `input_tokens + output_tokens/2` — a MEAN occupancy over the run, which
    is a half-integer whenever `output_tokens` is odd. Rounding it to a
    whole position is a choice, and this is where it is stated.

    Half UP, for two reasons. It never under-counts the KV read, which is
    what a decode step mostly does, so the error has the conservative sign.
    And it is one expression: half-to-EVEN (Python's `round`) needs the
    parity of the result to decide, which under a symbolic workload is a
    branch on the workload — a plan class split, and two walks over a sweep
    — for a tie-break with no physical meaning. A convention that cannot be
    written as a law is the wrong convention here.

    Nothing in the repo observed this before: both goldens use an even
    `output_tokens` (the SOL surface 1024, capacity 256), and the only odd
    literals hand a stub dict to `disagg_handoff` or sit in a frontend
    fixture. So the number was free to differ between the two domains, and
    did — 4096.5 came out 4096 as a number and 4097 as a law. This is the
    gate that was missing.
    """

    def test_a_half_rounds_up_in_both_domains(self):
        from backend.arch.eval.transforms import Phase, Workload

        for prompt, output, expected in ((4096, 1, 4097), (4096, 2, 4097),
                                         (4096, 3, 4098), (1024, 1, 1025),
                                         (1023, 1, 1024)):
            with self.subTest(prompt=prompt, output=output):
                numeric = Phase(kind="decode").resolve(
                    Workload(input_tokens=prompt, output_tokens=output))
                self.assertEqual(int(value_of(numeric["s_kv"])), expected)
                # The same law, read at the same point.
                law = Phase(kind="decode").resolve(
                    Workload(input_tokens=prompt, output_tokens=OUTPUT))
                read = sp.sympify(value_of(law["s_kv"])).xreplace(
                    {OUTPUT: sp.Integer(output)})
                self.assertEqual(int(read), expected)


class TestTheSequenceSymbolsStayDistinct(_Equivalence):
    """Several distinct engine symbols come out of the workload's sequence
    inputs, and no two of them may collapse into one.

    They are different numbers in the same step — a decode step's query
    axis is the verify width, its K/V rows are the cache plus that width
    less one, and what one query attends is narrowed again by a window — so
    two of them holding one law would be a wrong answer wearing a right
    one's shape, not a simplification.
    """

    def test_a_decode_step_binds_three_different_laws(self):
        _program, deployment = self.one_walk(
            "deepseek-ai/DeepSeek-V3", {}, "decode",
            workload={"batch": 1, "input_tokens": 4096, "s_verify": VERIFY},
            cache_len=LENGTH)
        binding = _walked(deployment).binding
        laws = {name: sp.sympify(value_of(binding[name]))
                for name in ("s_query", "s_kv", "s_past")}
        self.assertEqual(laws["s_query"], VERIFY)
        self.assertEqual(laws["s_kv"], LENGTH + VERIFY - 1)
        self.assertEqual(laws["s_past"], LENGTH - 1)
        for one, other in ((("s_query", "s_kv")), ("s_query", "s_past"),
                           ("s_kv", "s_past")):
            self.assertNotEqual(
                sp.simplify(laws[one] - laws[other]), 0,
                f"{one} and {other} came out the same law {laws[one]}")

    def test_what_one_query_attends_is_the_window(self):
        """`s_attend = Min(s_kv, s_window)`, carried as the law it is.

        gemma-2-9b alternates windowed and full attention, so one walk has
        both: the windowed layers' `s_attend` caps at the window and the
        full layers' does not. Read off ONE symbolic walk at a length below
        the window and a length above it — a collapsed `Min` cannot answer
        both.
        """
        _program, deployment = self.one_walk("google/gemma-2-9b", {},
                                            "prefill")
        analysis = _walked(deployment)
        window = int(value_of(analysis.binding["s_window"]))
        attends = {str(value_of(dims["s_attend"])): dims["s_attend"]
                   for dims in analysis.node_dims.values.values()
                   if "s_attend" in dims}
        capped = [law for law in attends.values()
                  if sp.sympify(value_of(law)).has(sp.Min)]
        self.assertTrue(capped, f"no windowed s_attend among {list(attends)}")
        for law in capped:
            below = _read(law, {"B": 1, "s_ctx": window // 2})
            above = _read(law, {"B": 1, "s_ctx": window * 4})
            self.assertEqual(below, window // 2)
            self.assertEqual(above, window)
        # And the full layers' key axis is the step's own length, uncapped.
        uncapped = [law for law in attends.values()
                    if not sp.sympify(value_of(law)).has(sp.Min)]
        self.assertTrue(uncapped,
                        "gemma-2 has full-attention layers and none showed")
        for law in uncapped:
            self.assertEqual(_read(law, {"B": 1, "s_ctx": window * 4}),
                             window * 4)


class TestOneWalkIsEveryWalk(_Equivalence):
    """The default-run case: a small dense model, sequence-cut and not."""

    def test_llama_tensor_and_context_parallel(self):
        for mesh in ({"tp_attention": 8}, {"cp": 8}):
            self.assert_one_walk_is_every_walk(
                "meta-llama/Llama-3.2-1B", mesh, batches=(1, 8),
                contexts=(1024, 4096))

    def test_llama_decode(self):
        self.assert_one_walk_is_every_walk(
            "meta-llama/Llama-3.2-1B", {"tp_attention": 8}, "decode",
            batches=(1, 8), contexts=(1024, 4096))

    def test_an_indivisible_length_is_refused_as_the_walk_refuses_it(self):
        self.assert_one_walk_is_every_walk(
            "meta-llama/Llama-3.2-1B", {"cp": 8}, batches=(1,),
            contexts=(1024, 1000, 1001))


@pytest.mark.regression
class TestOneWalkIsEveryWalkAcrossTheCorpus(_Equivalence):
    """Per model; each case is one symbolic walk against twelve numeric."""

    CASES = (
        ("meta-llama/Llama-3.2-1B", {}, "prefill"),
        ("meta-llama/Llama-3.2-1B", {"tp_attention": 8}, "prefill"),
        ("meta-llama/Llama-3.2-1B", {"cp": 8}, "prefill"),
        ("meta-llama/Llama-3.2-1B", {}, "decode"),
        ("deepseek-ai/DeepSeek-V3", {"tp_attention": 8, "ep_moe": 8},
         "prefill"),
        ("deepseek-ai/DeepSeek-V3", {"cp": 8}, "prefill"),
        ("deepseek-ai/DeepSeek-V3", {"tp_attention": 8, "ep_moe": 8},
         "decode"),
        ("Qwen/Qwen3-30B-A3B", {"dp_attention": 8, "ep_moe": 8}, "prefill"),
        ("deepseek-ai/DeepSeek-V4-Flash", {"tp_attention": 8, "ep_moe": 8},
         "prefill"),
        ("moonshotai/Kimi-K3", {"tp_attention": 8, "ep_moe": 8}, "prefill"),
        ("moonshotai/Kimi-K3", {"dp_attention": 8, "ep_moe": 8}, "prefill"),
    )

    def test_every_case(self):
        for model, mesh, phase in self.CASES:
            self.assert_one_walk_is_every_walk(model, mesh, phase)


if __name__ == "__main__":
    unittest.main()
