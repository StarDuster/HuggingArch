"""`Dim` — an extent that still knows what it is.

Two halves, one walk
--------------------
An axis has a NAME and a VALUE. One walk runs over the workload's counts as
symbols (batch, query and cache lengths, media counts) with every model dim a
number, and each extent it emits carries both: `(B, n_kv_heads, s_ctx,
d_head)` by identity, `(B, 8, s_query, 128)` by value. The value half is the
law in the counts that every point reads; the identity half is the same
quantity written in the model's names, which is what a formula prints and
what a consumer asks when it needs to know which axis is which.

Without the identity half every such consumer would rebuild it, and they
would not agree. Sharding would read a symbol name out of the BINDING rather
than off the tensor, so a recurrent state written in `n_kv_heads` under
`shard_symbols: [n_heads]` would be invisible while the two happen to be
equal; a per-component `kv_tp_shardable` would let a component invoked with
`n_kv_heads: 1` still claim to be head-shardable; and a `(1, 128, 4, 512)`
whose 4 nobody declared would have no way to be questioned at all.

`expr` is the axis's identity — a SymPy
expression over dim symbols, `n_kv_heads*d_head` for a merged head axis — and
`value` is what it resolves to under the binding that produced it.

Cheap for a reason that is already in the code
----------------------------------------------
`eval_dim_expr` parses each spec expression once into a tree of dim
Symbols and caches it, then substitutes per call. The expression half of a `Dim`
is that cached tree; only the substitution runs twice, once against the symbols
and once against the values. Both are `xreplace` on a small tree.

Measured on the corpus: a walk holding model dims concrete and the workload
symbolic produces 3,918 extents for DeepSeek-V4-Pro of which 2,095 vary with the
workload — and SEVEN distinct expressions. Structure is repeated, not multiplied.

What a `Dim` deliberately does not hold
---------------------------------------
The per-rank value. Sharding is applied on READ, by the policy, to whatever the
binding hands out; a `Dim` carrying a pre-divided value would be the double
shard again — a value that arrives already divided and is divided a second time
by the next read. What is bound is the global extent; what is read is the rank's,
and that rule lives on `Binding`.

Comparisons of laws are not decided
-----------------------------------
`<`, `==` and `bool` answer from `value`, and where the value is a law in the
workload's counts there is nothing to answer: the comparison raises
`Undecided`. The walk is every point's, so a question whose answer differs
between points — whether a window caps a cache, whether a shape is degenerate
— is written into the formula as its cases (`where`, `ratio`), and each point
reads the case that holds there. What the counts' declared domain decides —
the batch is at least one, a cache of `s_verify - 1 + Max(1, …)` positions is
never empty — is decided where the law is written (`Domain.settled`), and the
formula keeps only the cases that differ between points.
"""
from __future__ import annotations


import math
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import (Any, Callable, Dict, FrozenSet, Iterable, Iterator, List,
                    Optional, Tuple)

import sympy as sp

from backend.arch.law_text import Cases, Quotient, read_law


def _parts(other: Any):
    """`(expr, value)` for anything that can meet a `Dim` in an operator."""
    if isinstance(other, Dim):
        return other.expr, other.value
    if isinstance(other, sp.Expr):
        return other, other
    if isinstance(other, bool):
        return sp.Integer(int(other)), int(other)
    if isinstance(other, int):
        return sp.Integer(other), other
    if isinstance(other, float):
        # Exact in both halves: see `_exact_values`.
        return sp.Rational(other), sp.Rational(other)
    if isinstance(other, Fraction):
        return sp.Rational(other.numerator, other.denominator), other
    return NotImplemented, NotImplemented


def _exact_values(a: Any, b: Any) -> Tuple[Any, Any]:
    """Two value halves about to meet, each read as the exact number it is.

    An extent is exact: a whole count, an exact fraction of one (a MoE
    dispatch leaves the expert bank `1/32` of a token), or an expression in
    the workload's symbols. A byte width arrives as a float — `2.0` per BF16
    element, `0.5` per FP4 — and a float times an exact value is a float,
    which rounds the extent on the way: Kimi-K3's `8/7` of a row at 2 bytes
    came out `2.2857142857142856` and the decimals compounded through every
    later product. `_tokens_per_expert` already says why this matters —
    rounding early gets multiplied later.

    A Python float IS a dyadic rational, so `Rational` loses nothing by
    holding it: `Rational(0.5)` is `1/2`, `Rational(2.0)` is `2`. The
    rounding happens once, where a number is published."""
    if isinstance(a, float):
        a = sp.Rational(a)
    if isinstance(b, float):
        b = sp.Rational(b)
    return a, b


def _exact_quotient(a: Any, b: Any) -> Any:
    """Two value halves divided, exactly, whichever domain they arrived in.

    `/` is the one operator whose answer DEPENDS on the domain: two Python
    ints divide to a float (`128 / 128` is `1.0`, `10 / 4` is `2.5`) where
    two laws divide to an exact `Rational`. So a quotient written with the
    operator says one thing while the walk is numeric and another while it
    is symbolic, and the readers of that one row then disagree about what it
    says — measured at `layout.physical.pairs_share`, where the live pair
    count published as the string "4224.00000000000" on one reader and the
    integer 4224 on another.

    A whole quotient stays a Python `int`, which is what the counts around
    it are; anything else is exact — the `Rational` the extent algebra
    composes with. The float case is `_exact_values`'s, for the same reason."""
    a, b = _exact_values(a, b)
    if isinstance(a, int) and isinstance(b, int):
        return a // b if not a % b else sp.Rational(a, b)
    return a / b


@dataclass(frozen=True)
class Dim:
    """One axis extent: what it is, and what it comes to.

    `expr` is over the model's dim symbols and never collapses; `value` is the
    number under the binding this was resolved against. An arithmetic result
    carries both, so a merged axis reports `n_kv_heads*d_head` and `512` at once
    and no consumer has to pick.
    """

    expr: Any
    value: Any

    # -- identity --------------------------------------------------------

    @property
    def free_symbols(self) -> FrozenSet[Any]:
        return getattr(self.expr, "free_symbols", frozenset())

    @property
    def names(self) -> FrozenSet[str]:
        """The dim symbols this axis is made of, as strings.

        The question every proxy was approximating: "is this the sequence axis"
        is `"s_ctx" in axis.names`, and "does this deployment shard it" is an
        intersection with the policy's keys.
        """
        return frozenset(str(s) for s in self.free_symbols)

    def is_(self, name: str) -> bool:
        """Whether this axis IS that dim, rather than merely containing it.

        A head axis is `n_heads`; a fused QKV axis is `n_heads*d_head*3` and
        contains it. The distinction decides whether an axis can be split.
        """
        return self.expr == sp.Symbol(name)

    # -- sympy interop ---------------------------------------------------

    def _sympy_(self):
        """`sympify` protocol, so a `Dim` may be handed to `sp.Min`, `ceiling`
        and friends. The result of such a call is a plain expression — sympy
        does not know about the value half — so a caller that needs both back
        rebuilds with `Dim.of`."""
        return self.expr

    # -- arithmetic ------------------------------------------------------

    def _binop(self, other: Any, fn) -> Any:
        e, v = _parts(other)
        if e is NotImplemented:
            return NotImplemented
        return Dim(fn(self.expr, e), fn(*_exact_values(self.value, v)))

    def _rbinop(self, other: Any, fn) -> Any:
        e, v = _parts(other)
        if e is NotImplemented:
            return NotImplemented
        return Dim(fn(e, self.expr), fn(*_exact_values(v, self.value)))

    def __add__(self, o): return self._binop(o, lambda a, b: a + b)
    def __radd__(self, o): return self._rbinop(o, lambda a, b: a + b)
    def __sub__(self, o): return self._binop(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._rbinop(o, lambda a, b: a - b)
    def __mul__(self, o): return self._binop(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._rbinop(o, lambda a, b: a * b)
    def __truediv__(self, o): return self._binop(o, lambda a, b: a / b)
    def __rtruediv__(self, o): return self._rbinop(o, lambda a, b: a / b)
    def __pow__(self, o): return self._binop(o, lambda a, b: a ** b)
    def __mod__(self, o): return self._binop(o, lambda a, b: a % b)

    def __floordiv__(self, o):
        # `//` on the value and exact division on the expression: a spec writing
        # `d_model // n_heads` means the axis `d_model/n_heads`, and floor is
        # how it grounds. Keeping
        # `floor()` in the expression would make every downstream comparison
        # structural instead of arithmetic.
        e, v = _parts(o)
        if e is NotImplemented:
            return NotImplemented
        return Dim(self.expr / e, self.value // v)

    def __rfloordiv__(self, o):
        e, v = _parts(o)
        if e is NotImplemented:
            return NotImplemented
        return Dim(e / self.expr, v // self.value)

    def __neg__(self): return Dim(-self.expr, -self.value)
    def __abs__(self): return Dim(abs(self.expr), abs(self.value))

    # -- resolving -------------------------------------------------------

    def __int__(self) -> int: return int(self.value)
    def __index__(self) -> int: return int(self.value)
    def __float__(self) -> float: return float(self.value)
    def __bool__(self) -> bool:
        """The truth of a number. An expression has none: Python would call
        any object true, so `if extent:` took the non-empty road even at the
        points where the extent is zero, and deciding it here would hide a
        modelling decision inside a truth test. Ask `is_zero` (a decision
        over the points) or `same_law` (an identity) where the question is
        asked."""
        if varies(self.value):
            raise Undecided(
                f"the truth of the extent {self!r} is not a number: ask "
                f"`dim.is_zero` or `dim.same_law` where the question is "
                f"asked")
        return bool(self.value)
    def __round__(self, ndigits=None):
        return round(self.value, ndigits)

    def __eq__(self, o) -> bool:
        _e, v = _parts(o)
        return NotImplemented if v is NotImplemented else bool(self.value == v)

    def __ne__(self, o) -> bool:
        eq = self.__eq__(o)
        return eq if eq is NotImplemented else not eq

    def __lt__(self, o): _e, v = _parts(o); return self.value < v
    def __le__(self, o): _e, v = _parts(o); return self.value <= v
    def __gt__(self, o): _e, v = _parts(o); return self.value > v
    def __ge__(self, o): _e, v = _parts(o); return self.value >= v

    def __hash__(self) -> int:
        """On the VALUE, matching `__eq__`. An extent is used as a dict key in
        exactly the places it stands for a size."""
        return hash(self.value)

    def __repr__(self) -> str:
        return f"{self.value}⟨{self.expr}⟩"

    def __format__(self, spec: str) -> str:
        """A format spec addresses the VALUE, completing the numeric protocol.

        `f"{dim}"` keeps the `value⟨expr⟩` form, because a bare interpolation
        is asking what the axis IS. `f"{dim:,.0f}"` is asking for the number,
        and reads the same as it would on the int the axis stands for — so a
        message that renders an extent does not have to know whether it was
        handed a `Dim` or an `int`. A symbolic value refuses, which is correct:
        there is no number to render.

        The value is exact (`_exact_values`), and an exact number is not a
        Python one: `format(Integer(864704792696), ",")` raises where the int
        it stands for renders. Rounding to a Python number is what a format
        spec asks for, so it happens here — once, at the boundary where the
        number becomes text.
        """
        return repr(self) if not spec else format(as_python(self.value), spec)

    # -- building --------------------------------------------------------

    @classmethod
    def symbol(cls, name: str, value: Any) -> "Dim":
        """A bound dim: its own symbol, and what the model binds it to."""
        return cls(sp.Symbol(name), value)

    @classmethod
    def of(cls, value: Any) -> "Dim":
        """A constant, or a `Dim` passed through. For the extents an op states
        outright — a singleton axis, a declared `2` — which have no identity to
        carry and should still compose."""
        if isinstance(value, Dim):
            return value
        e, v = _parts(value)
        if e is NotImplemented:
            raise TypeError(f"not an extent: {value!r}")
        return cls(e, v)


def apply(sym_fn, num_fn, *args: Any) -> Any:
    """Build an extent through a function rather than an operator.

    The operators above cover arithmetic, and arithmetic is most of it. What
    they cannot cover is a sympy FUNCTION — `ceiling`, `Min`, `Max` — because
    calling one on a `Dim` sympifies it through `_sympy_` and returns a plain
    expression with the value half dropped. That is not a hole to be defended
    against; it is the second way an extent gets built, and it needs the same
    two halves stated.

    It came up immediately: `M =
    ceil(B*s_ctx*n_experts_active*dp_attention/n_route_targets)`
    built with `sp.ceiling` came
    out an expression that no longer knew its own value, and 418 of 5,436 shapes
    — every node in a MoE dispatch zone — reported an expression where a number
    was expected. Whoever builds an extent by a route other than `+ - * /` says
    so here, with both readings.
    """
    if any(isinstance(a, Dim) for a in args):
        values = [value_of(a) for a in args]
        # In the symbolic value domain the VALUE half is itself an
        # expression — `math.ceil` of a symbol is a type error, and the
        # symbolic function is the right reading of both halves.
        if any(isinstance(v, sp.Basic) and v.free_symbols for v in values):
            return Dim(sym_fn(*[expr_of(a) for a in args]),
                       sym_fn(*values))
        return Dim(sym_fn(*[expr_of(a) for a in args]),
                   num_fn(*values))
    # Nothing tracked. `ops.py` is shared substrate whose other callers pass
    # plain ints in numeric mode and free expressions in symbolic mode, and
    # both have to come back as what `sp.Min` would have returned — a number
    # cannot be `min()`ed against a symbol, and a symbol has no numeric answer.
    if any(getattr(a, "free_symbols", None) for a in args):
        return sym_fn(*args)
    return num_fn(*args)


def ceiling(x: Any) -> Any:
    """`⌈x⌉`, both readings. `int(-(-a // b))` is not available here because the
    argument arrives already divided."""
    return apply(sp.ceiling, math.ceil, x)


def minimum(a: Any, b: Any) -> Any:
    return apply(sp.Min, min, a, b)


def maximum(a: Any, b: Any) -> Any:
    return apply(sp.Max, max, a, b)


def _log2_number(n: Any) -> Any:
    """`log2(n)` of a number: exact for a power of two, as the transforms
    that ask it are sized; a float otherwise."""
    whole = int(n)
    if whole == n and whole > 0 and whole & (whole - 1) == 0:
        return whole.bit_length() - 1
    return math.log2(n)


def log2(x: Any) -> Any:
    """`log₂ x`, both readings (a fast transform's stage count)."""
    return apply(lambda e: sp.log(e, 2), _log2_number, x)


#: Each workload input that is a ratio rather than a whole count, with the
#: least value it takes (`ratio_symbol`). A whole count's least is its
#: declared sign (`least_of`).
_RATIO_FLOORS: Dict[Any, Any] = {}


def ratio_symbol(name: str, least: Any) -> sp.Symbol:
    """A workload input that is a ratio of at least `least` — the hottest
    expert's load over the mean, `rho ≥ 1` — declared as the symbol one walk
    answers every point with. Its domain is a real number from `least` up,
    held where a whole count's is (`Domain`, `declared_workload`).

    A name is declared once. Two declarations of one name are one symbol
    (SymPy's symbols are equal by name and assumptions), so a second floor
    would move the domain every law already written in it was written for,
    silently; it is refused instead. The floor is held to the symbol's own
    assumption: a positive symbol has no floor at or below zero."""
    floor = sp.Rational(least)
    symbol = sp.Symbol(name, real=True, positive=True)
    if floor <= 0:
        raise ValueError(
            f"ratio {name!r} is declared positive, so its least value is "
            f"above zero, not {floor}")
    declared = _RATIO_FLOORS.get(symbol)
    if declared is not None and declared != floor:
        raise ValueError(
            f"ratio {name!r} is already declared from {declared}; a second "
            f"declaration from {floor} would move the domain the laws "
            f"already written in it hold on")
    _RATIO_FLOORS[symbol] = floor
    return symbol


def point_value(value: Any) -> Fraction:
    """One of a point's numbers as the exact `Fraction` a compiled law is
    read at (`analysis.Point.laws`, `row_roofline.LawBook.row_at`: the one
    conversion both readers make).

    From the number as written, so a ratio input stays the ratio it is:
    `int()` read a `rho` of `3/2` as 1 and the law came out at a workload
    nobody asked for."""
    return Fraction(str(value))


def least_of(symbol: Any) -> Optional[Any]:
    """The least value a declared workload symbol takes: a whole count's 0,
    or 1 where it is declared positive; a ratio's declared floor
    (`ratio_symbol`); None for a symbol that declares neither."""
    if symbol.is_integer and symbol.is_nonnegative:
        return 1 if symbol.is_positive else 0
    return _RATIO_FLOORS.get(symbol)


class Domain:
    """The counts a walk holds over, each with the least value it takes —
    the workload's (`transforms.Workload.symbols`: the batch and the verify
    width at least one, the prompt, the output and the cached prefix zero
    or more) and whatever else the deployment binds to a declared count
    (`declared_workload`), as the walk's frame admitted them
    (`transforms.Deployment._walk`). A name not in it is anything. The
    walk's ledger carries it (`assuming`): where a condition is written
    (`where`, `ratio`) it is decided over this domain, and only over it.

    Not a record of the walk's values: a reading of the walk at a point
    leaves it as it is (`analysis._PointReading`)."""

    __slots__ = ("least",)

    def __init__(self, values: Iterable[Any] = ()) -> None:
        """The domain of the declared counts `values` are made of: each
        whole count at least one where it is declared positive, else zero,
        and each ratio from its declared floor (`least_of`;
        `declared_workload` holds every one to a declaration)."""
        exprs = [value_of(value) for value in values]
        self.least: Dict[Any, Any] = {
            symbol: least_of(symbol)
            for expr in exprs if isinstance(expr, sp.Basic)
            for symbol in expr.free_symbols
            if least_of(symbol) is not None}

    def refusal(self, symbol: Any, value: Any) -> Optional[str]:
        """Why `value` is no value of `symbol` — not whole where the symbol
        counts, or below its least — or None where it is one.

        The whole answer, so one reading of a declaration serves both
        bindings: the symbolic one asks it of the point it is read back at
        (`transforms.Workload.symbolic`) and the numeric one of the number
        itself (`declared_workload`). Two readings would be two doors, and
        the numeric one was missing.

        Integrality is SymPy's own reading of the number, and only a
        definite `False` refuses: `Float(2.0)` is not an integer to SymPy
        and is not a non-integer either (`is_integer` is None), and a
        question the algebra does not decide is not one this door decides
        for it — a count stated as a SymPy number is refused where that
        matters, which is the point a symbolic walk is read back at
        (`Workload.symbolic`)."""
        least = self.least[symbol]
        if symbol.is_integer and sp.sympify(value).is_integer is False:
            return "it counts in whole numbers"
        if value >= least:
            return None
        if symbol.is_integer:
            return f"it counts {'at least one' if least else 'zero or more'}"
        return f"it is at least {least}"

    def bounds(self, expr: Any) -> Tuple[Any, Any]:
        """The least and the greatest value `expr` takes over this domain:
        each declared symbol from its least up, every other name
        anything. Interval arithmetic, so a bound is sure but not always
        tight; `(-oo, oo)` where nothing is known."""
        unknown = (-sp.oo, sp.oo)
        if expr.is_Number:
            return expr, expr
        if expr.is_Symbol:
            least = self.least.get(expr)
            return unknown if least is None else (sp.Rational(least), sp.oo)
        parts = [self.bounds(arg) for arg in expr.args] if expr.args else []
        if isinstance(expr, sp.Add):
            return sp.Add(*(lo for lo, _ in parts)), sp.Add(*(hi for _, hi in parts))
        if isinstance(expr, sp.Mul):
            lo, hi = sp.Integer(1), sp.Integer(1)
            for a, b in parts:
                ends = [x * y for x in (lo, hi) for y in (a, b)]
                if any(e is sp.nan for e in ends):
                    return unknown
                lo, hi = sp.Min(*ends), sp.Max(*ends)
            return lo, hi
        if isinstance(expr, (sp.Max, sp.Min)):
            pick = sp.Max if isinstance(expr, sp.Max) else sp.Min
            return pick(*(lo for lo, _ in parts)), pick(*(hi for _, hi in parts))
        if isinstance(expr, (sp.floor, sp.ceiling)):
            (lo, hi), = parts
            return expr.func(lo), expr.func(hi)
        if isinstance(expr, Cases):
            (_c, (a, b), (x, y)) = (None, parts[1], parts[2])
            return sp.Min(a, x), sp.Max(b, y)
        return unknown

    def settled(self, truth: Any) -> Any:
        """`truth` decided where this domain decides it (`bounds`): a
        comparison of a case's operands with a number is the case's own
        condition or its negation where both operands answer it, and a
        comparison whose difference keeps one sign over the domain is that
        sign's answer. What the domain does not decide is returned as it
        is."""
        if truth in (sp.true, sp.false):
            return truth
        if isinstance(truth, (sp.And, sp.Or, sp.Not)):
            return truth.func(*(self.settled(arg) for arg in truth.args))
        if not isinstance(truth, sp.core.relational.Relational):
            return truth
        lhs, rhs = truth.lhs, truth.rhs
        for case, other, flip in ((lhs, rhs, False), (rhs, lhs, True)):
            if isinstance(case, Cases) and other.is_Number:
                condition, then, otherwise = case.args
                a, b = (self.settled(truth.func(other, side) if flip
                                     else truth.func(side, other))
                        for side in (then, otherwise))
                if a is b and a in (sp.true, sp.false):
                    return a
                if (a, b) == (sp.true, sp.false):
                    return condition
                if (a, b) == (sp.false, sp.true):
                    return self.settled(sp.Not(condition))
                return truth
        lo, hi = self.bounds(lhs - rhs)
        decided = {
            sp.Eq: (sp.false if lo > 0 or hi < 0 else
                    sp.true if lo == hi == 0 else None),
            sp.Ne: (sp.true if lo > 0 or hi < 0 else
                    sp.false if lo == hi == 0 else None),
            sp.Lt: sp.true if hi < 0 else sp.false if lo >= 0 else None,
            sp.Le: sp.true if hi <= 0 else sp.false if lo > 0 else None,
            sp.Gt: sp.true if lo > 0 else sp.false if hi <= 0 else None,
            sp.Ge: sp.true if lo >= 0 else sp.false if hi < 0 else None,
        }.get(type(truth))
        return truth if decided is None else decided


def _under(expr: Any, condition: Any, truth: Any) -> Any:
    """`expr` where `condition` is known to be `truth`."""
    return (expr.xreplace({condition: truth})
            if isinstance(expr, sp.Basic) and expr.has(condition) else expr)


def where(holds: Callable[..., Any], over: Tuple[Any, ...], then: Any,
          otherwise: Any) -> Any:
    """`then` at the points where `holds(*over)` is true, `otherwise` at the
    rest — one formula, both halves.

    A value that differs between points is written as the cases it is
    (`Cases`), never chosen by the walk: the walk is every point's, and a
    choice it made would be one point's. Where the operands are numbers the
    condition is a number's truth and the result is the operand it selects,
    itself — so a walk at a point reads exactly what that operand was.
    `holds` builds a SymPy condition (`sp.Lt`, `sp.Eq`, ...) over the
    values. The identity is the same cases over the operands' identities,
    decided by the same condition: read at a point it is the identity of the
    operand chosen there, as a walk at that point names it. Both operands
    are built before the choice, so each must be defined at every point; a
    quotient by what may be zero is `ratio`'s, which takes it only where it
    is defined (`Quotient`)."""
    ledger = _ASSUMED.get()
    truth = (ledger.domain if ledger is not None else Domain()).settled(
        sp.sympify(holds(*(value_of(x) for x in over))))
    if truth is sp.true:
        return then
    if truth is sp.false:
        return otherwise
    if isinstance(truth, (sp.Ge, sp.Gt, sp.Ne)):
        # One spelling per condition — `a < b`, `a <= b`, `a = b` — so the
        # cases of one condition are recognised as one wherever they meet.
        truth, then, otherwise = truth.negated, otherwise, then

    def cases(half: Callable[[Any], Any]) -> Any:
        # Inside each operand the condition is decided: a case of it there
        # is the operand that answer picks.
        return Cases(truth, _under(half(then), truth, sp.true),
                     _under(half(otherwise), truth, sp.false))

    if not any(isinstance(x, Dim) for x in (then, otherwise)):
        return cases(value_of)
    return Dim(cases(expr_of), cases(value_of))


def ratio(part: Any, whole: Any, *, empty: Any) -> Any:
    """`part / whole`, and `empty` at the points where `whole` is zero — a
    share of nothing, a fan-out over an empty input — as one formula
    (`Quotient`, both halves, as `where` writes cases). Where `whole` is a
    number the quotient is taken only when it is not zero, so a point reads
    the quotient a division there makes."""
    if not varies(value_of(whole)):
        if value_of(whole) == 0:
            return empty
        # The quotient EXACTLY, which a bare `/` is only in one of the two
        # domains: `10 / 4` on two Python ints is the float `2.5` where the
        # same ratio of two laws is `5/2`, so this verb answered the
        # proportion it exists to answer differently depending on which
        # binding was walking — and a `Dim`'s value half differently from
        # its own expression half.
        return apply(lambda p, w: p / w, _exact_quotient, part, whole)
    ledger = _ASSUMED.get()
    truth = (ledger.domain if ledger is not None else Domain()).settled(
        sp.Eq(value_of(whole), 0))
    if truth is sp.true:
        return empty
    if truth is sp.false:
        return part / whole

    def quotient(half: Callable[[Any], Any]) -> Any:
        return Quotient(truth, half(empty), half(part), half(whole))

    if not any(isinstance(x, Dim) for x in (empty, part, whole)):
        return quotient(value_of)
    return Dim(quotient(expr_of), quotient(value_of))


def nearest(x: Any) -> Any:
    """The nearest whole number, half up, in both domains.

    One convention, stated, because the two domains have to agree: Python's
    `round` is half-to-EVEN, and an expression's natural reading
    `floor(x + 1/2)` is half-UP, so an average that lands exactly on a half
    — a decode cache of `prompt + output_tokens/2` with an odd
    `output_tokens` — came out 5120 as a number and 5121 as a law. Which
    half-way case goes which way is arbitrary; the two answering
    differently is not."""
    return apply(lambda e: sp.floor(e + sp.Rational(1, 2)),
                 lambda v: math.floor(v + 0.5), x)


#: The operations that need BOTH halves and cannot be written as operators.
#
# `sp.Min(a, b)` on a `Dim` sympifies it through `_sympy_`, gets the expression
# and returns a plain `Expr` — the value half is gone, silently. Measured: 418
# of 5,436 shapes came out with an expression where a number belonged, every one
# of them a MoE dispatch zone, because `M`'s derivation went through
# `sp.ceiling`.
#
# A closed set rather than a rule to remember. `ops.py` calls exactly three
# sympy functions on dims — `Min` (17), `ceiling` (4), and the `//` operator —
# so the set is enumerable, and a gate can assert that no bare `sp.Min` survives
# in a shape or positions lambda. Rewrapping at each call site with `Dim.of` was
# the alternative and is the shape of thing this package keeps removing: it
# works until someone adds the eighteenth call.
#
# `substitute` joined the set later, from outside `ops.py`: putting a dim INTO
# an expression drops the value half by the same mechanism as calling a function
# ON one, and does it more quietly — see its docstring.


def ceil_div(a: Any, b: Any) -> Any:
    """`ceil(a/b)`, on both halves, with the integer arithmetic spelled out.

    `-(-va // vb)` rather than `math.ceil(va/vb)`: the operands are token counts
    and expert counts, and floating division of a 1M-token context by a 4-way
    compression is exact only until it is not.
    """
    return apply(lambda x, y: sp.ceiling(x / y),
                 lambda x, y: -((-x) // y), a, b)


def exact_div(a: Any, b: Any) -> Any:
    """Division the caller DECLARES to be exact — a head axis over its heads.

    Separate from `floor_div` because the two disagree exactly when the
    declaration is wrong, and today `Dim.__floordiv__` picks one behaviour for
    both: the expression divides exactly while the value floors, so a case that
    does not divide evenly comes out with the two halves describing different
    numbers and nothing says so. Here it raises instead.

    Which is why the value half divides exactly too, rather than flooring. A
    declared-exact division has no remainder to drop, so `//` on the value
    added a `floor` the expression half does not have — and in the symbolic
    domain, where the value half is itself a law, that floor SURVIVES:
    `d_model // n_heads` came out `floor(d_model/n_heads)` as a value beside
    `d_model/n_heads` as an expression, two readings of one axis. The raise
    above is the gate that keeps this honest: the two halves can only agree
    exactly because a case that does not divide never gets here.
    """
    ea, va = _parts(a)
    eb, vb = _parts(b)
    if isinstance(va, int) and isinstance(vb, int) and vb and va % vb:
        raise InexactDivision(
            f"{va} / {vb} was declared exact and is not. Use `floor_div` if the "
            f"remainder is meant to be dropped — the two produce different "
            f"expressions, and only one of them is what the spec says."
        )
    return Dim(ea / eb, _exact_quotient(va, vb) if vb else va)


def floor_div(a: Any, b: Any) -> Any:
    """Division that DROPS a remainder, on both halves. The expression keeps the
    floor, so a consumer comparing it against another expression compares the
    same thing the value does."""
    return apply(lambda x, y: sp.floor(x / y), lambda x, y: x // y, a, b)


def substitute(law: Any, symbol: sp.Symbol, extent: Any) -> Any:
    """A law written in one symbol, read AT an extent — both halves.

    `apply` covers calling a sympy function ON a dim. This is the mirror:
    putting a dim INTO an expression. `law.subs(s_kv, cache_len)` sympifies the
    `Dim` through `_sympy_`, gets back the expression half — the bare symbol
    `s_kv` — and substitutes `s_kv` for `s_kv`. Nothing raises, nothing is
    dropped in a way the result admits to: the law comes back UNCHANGED, so a
    KV read of a 4352-position cache reports `1024.0*s_kv` and the first thing
    to notice is the `int()` at the far end of the traffic model.

    That failure mode is worse than `apply`'s. A shape with an expression where
    a number belonged is at least visibly the wrong kind of thing; a law that
    substitutes to itself is indistinguishable from a law nobody asked to
    evaluate.
    """
    if not isinstance(law, sp.Basic):
        # A law with no expression in it — a recurrent state's plain `1` —
        # reads the same at every length. An isinstance rather than a
        # `hasattr(law, "subs")`: the question is what this value IS, and
        # duck-typing it would also swallow a `Dim` handed in by mistake,
        # which sympifies to its expression half and substitutes to itself.
        return law
    if not isinstance(extent, Dim):
        # Nothing tracked — the untracked callers of this package pass plain
        # ints numerically and free symbols symbolically, and both want back
        # exactly what `.subs` would have returned.
        return law.subs(symbol, extent)
    return Dim(law.subs(symbol, expr_of(extent)),
               law.subs(symbol, value_of(extent)))


class InexactDivision(ValueError):
    """An `exact_div` whose operands do not divide."""


# -- the conditions a walk's formulas hold under ----------------------------
#
# A legality condition on an extent — "the tokens divide the
# context-parallel degree", "a slice lies inside its axis" — is checked on
# the spot when the extent is a number. When the extent is a law in the
# workload's symbols it holds at some points and not at others, and the walk
# does not decide it: it records the condition (`require`) as part of the
# domain of every formula it writes, and whoever reads the walk at a point
# checks it there, refusing with the site's own exception
# (`Analysis.admit`).
#
# The ledger is scoped to one walk (`assuming`), not threaded through every
# sizing primitive: the conditions sit in pure layout functions (`Block`'s
# exact split) that have no walk to hand. A reader that projects the walk
# afterwards (per-rank extents) runs under the same ledger
# (`Analysis.assumed`), and the conditions it states join the walk's.
# Outside either, a condition on a law raises.


def _read_at(expr: Any, point: Mapping[str, Any]) -> Any:
    """`expr` with the point's symbols bound — a number where it is one, so
    a refusal's message reads as a walk at the point would."""
    value = sp.sympify(expr)
    value = value.xreplace(
        {s: sp.Integer(point[s.name]) for s in value.free_symbols
         if s.name in point})
    return int(value) if value.is_Integer else value


@dataclass(frozen=True)
class Assumption:
    """A condition the point must meet for a walk's formulas to hold there.

    `condition` is a SymPy boolean over the value halves — expressions in the
    workload's symbols: `Mod(s_ctx, 8) = 0` for a cut that must divide.
    `refusal(at)` builds the exception for a point that breaks it; `at`
    reads an expression at that point, so the message names its numbers."""

    condition: Any
    refusal: Callable[[Callable[[Any], Any]], Exception] = field(
        compare=False, hash=False)

    def at(self, point: Mapping[str, Any]) -> None:
        """Raise this condition's refusal when `point` breaks it."""
        if self.holds(point):
            return
        raise self.refusal(lambda expr: _read_at(expr, point))

    def holds(self, point: Mapping[str, Any]) -> bool:
        """Whether `point` satisfies this condition."""
        truth = _read_at(self.condition, point)
        if truth is sp.true or truth is sp.false:
            return truth is sp.true
        raise ValueError(
            f"{self.condition} is a condition of the walk, and the point "
            f"{dict(point)} does not decide it")


class Assumed:
    """The conditions in force: the ones a walk has recorded so far, or a
    finished walk's, which a reader projecting it adds its own to — and the
    domain they are written over (`Domain`)."""

    def __init__(self, held: Tuple[Assumption, ...] = (),
                 domain: Optional["Domain"] = None) -> None:
        self.held: List[Assumption] = list(held)
        self.domain = domain if domain is not None else Domain()

    def record(self, assumption: Assumption) -> None:
        """Hold `assumption`, unless it is already held. Nothing is proved
        here: a condition joins the domain as it is stated, and the point
        answers it."""
        if assumption in self.held:
            return
        self.held.append(assumption)


_ASSUMED: ContextVar[Optional[Assumed]] = ContextVar("assumed", default=None)

@contextmanager
def assuming(held: Tuple[Assumption, ...] = (),
             domain: Optional["Domain"] = None) -> Iterator[Assumed]:
    """A walk's ledger of the conditions its formulas hold under, in walk
    order, starting from `held` (a finished walk's, for a reader), over the
    walk's `domain` (none: nothing is known of any name)."""
    token = _ASSUMED.set(Assumed(held, domain))
    try:
        yield _ASSUMED.get()
    finally:
        _ASSUMED.reset(token)


def require(condition: Any,
            refusal: Callable[[Callable[[Any], Any]], Exception]) -> None:
    """A condition the point must meet: part of the domain of every formula
    the walk writes from here on.

    Checked now where it is a number's truth, refused with `refusal(at)`
    when false. Over the workload's symbols it is recorded, never decided:
    the walk is every point's, and whether a point meets the condition is
    asked of that point (`Analysis.admit`) with the refusal a walk there
    would raise. `at` reads an expression at the point the check is made,
    so the refusal names its numbers."""
    condition = sp.sympify(condition)
    if condition is sp.true:
        return
    if condition is sp.false:
        raise refusal(lambda expr: expr)
    ledger = _ASSUMED.get()
    if ledger is None:
        raise ValueError(
            f"whether {condition} holds depends on the point, and no walk "
            f"is recording the conditions its formulas hold under")
    ledger.record(Assumption(condition, refusal))


def require_equal(a: Any, b: Any,
                  refusal: Callable[[Callable[[Any], Any]], Exception]
                  ) -> None:
    """These two extents MUST be the same number (`require`): a concat
    whose segments do not tile its axis, two operands whose extents
    disagree, a routed row count that is not the mean the evidence
    recorded. One answer is a modelling error, and the site that can say
    what a disagreement means passes its own refusal, replayed at any point
    that breaks it."""
    require(sp.Eq(value_of(a), value_of(b)), refusal)


def require_multiple(extent: Any, modulus: Any,
                     refusal: Callable[[Any], Exception]) -> None:
    """`extent` must be a multiple of `modulus` (`require`);
    `refusal(extent)` is the exception at an extent that is not."""
    value = value_of(extent)
    remainder = value % modulus
    if not varies(remainder):
        if remainder:
            raise refusal(value)
        return
    require(sp.Eq(remainder, 0), lambda at: refusal(at(value)))


class Undecided(TypeError):
    """A comparison of laws in the workload's symbols, whose answer can
    differ between points.

    A `TypeError`, which is what SymPy raises for the truth of an undecided
    relational: the same question, and the same answer — nobody may act on
    it. The walk is every point's and takes no decision about the workload;
    the asker writes the cases (`where`) or states a condition on the point
    (`require`)."""


def declared_workload(value: Any, what: str, symbol: Any = None) -> Any:
    """A workload input, held to what a walk can reason about — and, where
    the input names the symbol it is swept as, to what that symbol declares.

    A symbolised input is a count of tokens or requests, so it must say so:
    `Symbol(name, integer=True, nonnegative=True)`, or `positive=True` when
    zero is not one of its values. The declaration is the count's domain:
    the walk's laws are written for it (a `Max(1, s_input)` a positive
    batch needs none of), a point is checked against it where it is made
    (`Workload.symbolic`), and a symbol that declares nothing is one about
    which no law is written.

    `symbol` is that declaration, and a NUMBER is held to it too (`Domain`,
    which is also what the point is read back through — one reading, both
    bindings). Without it the two halves disagreed about the same fact: the
    declaration folded `Max(1, s_verify)` away in the symbolic domain while
    the numeric domain let a verify width of 0 through and that same
    `maximum` rewrote it to 1. A value with no symbol — a phase's cache
    length, a dim the mesh binds, where the sampler reads — declares no
    domain, and there is nothing to hold it to but this.

    Refused at the door with the reason, rather than four screens deep in
    the layout algebra.

    Reads the VALUE half, because that is what every later decision reads: a
    bound dim arrives as `Dim(s_ctx, Symbol('S'))` — its own identity beside
    whatever the caller bound it to — and checking the wrapper instead of
    the number inside it let `with_dims(s_ctx=Symbol('S'))` straight through
    a door it was standing in front of."""
    value = value_of(value)
    if not isinstance(value, sp.Basic) or not value.free_symbols:
        # A number, or an expression that is one: the point is here, so the
        # domain is asked here. A law the caller states in the workload's
        # own symbols is NOT asked — what it is worth differs between
        # points, and the walk takes no decision about the workload
        # (`Undecided`); the symbols it is made of are held below.
        if symbol is not None and value is not None:
            refusal = Domain((symbol,)).refusal(symbol, value)
            if refusal is not None:
                raise ValueError(f"{what} is {value}; {refusal}")
        return value
    undeclared = sorted(
        str(free) for free in value.free_symbols
        if least_of(free) is None)
    if undeclared:
        raise ValueError(
            f"{what} is symbolic in {undeclared}, which declare no sign or "
            f"no integrality. A swept workload input is a whole count: "
            f"declare it `Symbol(name, integer=True, nonnegative=True)` "
            f"(or `positive=True` when it is never zero), or a ratio "
            f"`ratio_symbol(name, least)`, so the walk can decide what is "
            f"true of it")
    return value


def _exact_extent(extent: Any) -> Any:
    """An extent's value, held to being one: a FLOAT is refused, not read.

    The three questions below — is it whole, is it zero, is it one — each
    answer by comparing the value against a number, and SymPy reads such a
    comparison STRUCTURALLY: `Float(2.0) == Integer(2)` is False, and
    `Float(2.0).is_integer` is None rather than False, because the
    algebra declines to decide the integrality of an approximation. So all
    three were decided by the value's CARRIER rather than by the value. A
    whole count carried as a `Float` was reported fractional (`is_whole`),
    a zero reported non-zero (`is_zero`), a unit axis reported non-unit
    (`is_unit`) — while the same numbers carried as a Python `float`, which
    `int(v) == v` reads arithmetically, answered the other way. One value,
    two answers, picked by the type holding it.

    REFUSED rather than converted, which is where this differs from
    `_exact_values`. That one converts, and is right to: it stands on an
    arithmetic boundary, where the float has something to compose with and
    `Rational` holding it exactly is strictly better than letting it round
    the product. A question has nothing to compose — it hands back an
    answer a caller turns into an index — and `Rational(Float(2.0))` is
    exact ABOUT THE FLOAT, which is not the same as being the count anyone
    meant: the floats this walk has made came out the true value less
    6e-13, and the two active-parameter rows `520e408c` moved were floored
    one short of their own sum. A door that read such a float as whole
    would hand a layout an off-by-one nothing downstream could question.

    So the float is the defect, at the arithmetic that made it — two int
    counts under Python's `/` is the measured case, where
    `layout.physical.pairs_share` published one pair count as
    "4224.00000000000" on one reader and 4224 on another — and it is NAMED
    here, fixed there. Answering `False`, which the structural reading did,
    is the one outcome that is neither answer nor refusal:
    `layout.geometry` turned a whole split factor into an error message
    about average extents, and `layout.physical_cost` refused a sequence
    axis that divides perfectly.

    Nothing legitimate is refused. What is genuinely not a whole number
    reaches these doors as the exact `Rational` the extent algebra carries
    it in — 138 of the 13,426 `is_whole` calls one corpus sweep makes are
    `1/2`, `1/4`, `1/128` averages, still answered False as averages — and
    over 62 snapshots, nine meshes and both phases (856 walks, 13,426
    calls; 596 walks and 534,965 calls counting `is_zero` and `is_unit`)
    not one float arrives.
    """
    value = value_of(extent)
    if isinstance(value, float) or (isinstance(value, sp.Basic)
                                    and value.has(sp.Float)):
        raise TypeError(
            f"{value!r} is not an extent: an extent is exact — a whole "
            f"count, an exact fraction of one, or a law in the workload's "
            f"symbols — and a float is an approximation, whose integrality "
            f"SymPy itself does not decide. Divide exactly where this value "
            f"was made (`exact_div`, `ratio`, `exact(a) / exact(b)`)")
    return value


def is_whole(extent: Any) -> bool:
    """Whether an extent is a whole number at every point: a number that is
    one, or an expression whole by construction (`floor(s_ctx/4)`). One that
    is whole at some points only (`s_ctx/4`) is an average — the windows a
    decode step completes — and is not: what it indexes cannot depend on
    the point.

    Asked of the value, which has to be one an extent takes
    (`_exact_extent`)."""
    value = _exact_extent(extent)
    if not varies(value):
        return bool(int(value) == value)
    return sp.sympify(value).is_integer is True


def common_divisor(extent: Any, degree: int) -> int:
    """`gcd(extent, degree)` — how many equal parts a cut of `degree` ranks
    can make of `extent` — in both domains.

    For an expression the gcd would vary with the point, and with it the
    layout. So a cut of an extent that grows with the workload is a cut into
    `degree` parts, and that the extent divides is a condition on the point
    (`require_multiple`), refused where it does not."""
    value = value_of(extent)
    if not varies(value):
        return math.gcd(int(value), degree)
    require_multiple(value, degree, lambda at: ValueError(
        f"{at} does not split into {degree} equal parts"))
    return degree


def varies(value: Any) -> bool:
    """Whether a value is an expression in free symbols, not a number."""
    return isinstance(value, sp.Basic) and bool(value.free_symbols)


def exact(extent: Any) -> sp.Basic:
    """The value as an exact SymPy number — or the expression it is.

    The exact-rational carrier for an average extent (a mean of `1/32` row per
    expert), which composes with an expression in the workload where
    `fractions.Fraction` cannot.

    Declared `-> sp.Basic` because the type is the point, not an incidental:
    an operator applied to what this returns is SymPy's in either domain, so
    `exact(a) / exact(b)` is one exact `Rational` where `a / b` on two Python
    ints would have been a float. `tools.scoreboard` reads that annotation."""
    return sp.sympify(value_of(extent), rational=True)


def equal(a: Any, b: Any) -> bool:
    """Whether two extents are the same number. Numbers are compared. Two
    laws in the workload's symbols are not compared: whether they agree
    can differ between points, and the walk takes no decision about the
    workload — the question is an identity (`same_law`), a condition on the
    point (`require_equal`) or the cases of a value (`where`), and asking it
    here raises `Undecided`."""
    difference = value_of(a) - value_of(b)
    if not varies(difference):
        return difference == 0
    raise Undecided(
        f"whether {value_of(a)} = {value_of(b)} is a question about the "
        f"workload, which the walk does not decide")


def is_zero(extent: Any) -> bool:
    """Whether an extent is the number zero (`equal`), asked of a value an
    extent takes (`_exact_extent`)."""
    value = _exact_extent(extent)
    if not varies(value):
        return value == 0
    return equal(value, 0)


def settled_non_negative(extent: Any) -> bool:
    """Whether the declarations already settle that an extent is at least
    zero — so a caller about to `require(Ge(extent, 0), …)` need not.

    `Ge(x, 0)` is `true` exactly where this is (`relational.is_ge`: a number
    by its sign, anything else by that assumption), and `require` returns
    without recording a condition that is already true. So the two answer
    the same question and this one answers it without building a relation
    to carry it. An extent the declarations do NOT settle is False here and
    goes on to `require`, which records it as the walk's assumption or
    refuses it; nothing is decided that was not decided before.

    Sympified STRICTLY, as a relation sympifies its sides: a value that is
    not an extent at all raises here rather than being parsed as text, and
    coercion lives in this module rather than in each caller
    (`tests/meta/test_structure_debt.py` holds that boundary)."""
    return sp.sympify(extent, strict=True).is_extended_nonnegative is True


def same_law(a: Any, b: Any) -> bool:
    """Whether two extents are one law: the same expression of the
    workload, at every point alike. An identity question — whether a KV law
    narrows the step's `s_kv` at all, so the narrowed axis keeps the walk's
    name — and not `equal`'s: a window that binds at some lengths only is a
    different law from the length it caps, even at the points where the two
    come to one number."""
    return value_of(a) == value_of(b)


def is_unit(extent: Any) -> bool:
    """Whether an axis is a unit axis a broadcast stretches: one whose
    extent is the number one.

    An extent that grows with the workload is not one, at any point — not
    even where it comes to one (`B = 1`). Broadcasting a one-long axis
    against another is the identity, so the tensor is the same either way;
    what the answer decides is which operand an elementwise op takes its
    axis from, and that must not change with the batch, or one walk could
    not answer every batch.

    Asked of a value an extent takes (`_exact_extent`)."""
    value = _exact_extent(extent)
    return not varies(value) and value == 1


def whole_count(extent: Any,
                refusal: Callable[[Callable[[Any], Any]], Exception]) -> Any:
    """`extent`'s value as a count: a nonnegative whole number — an `int`
    where it is a number, the law where it grows with the workload, with
    the condition that it is whole and nonnegative recorded for the point
    (`require`)."""
    value = value_of(extent)
    require(sp.And(sp.Eq(value, sp.floor(value)), sp.Ge(value, 0)), refusal)
    return value if varies(value) else int(value)


def as_python(value: Any) -> Any:
    """An exact number as the Python number it stands for: `int` when it is
    whole, `float` when it is not. An expression with a free symbol is
    returned untouched — there is no number in it, and whoever asked for one
    should fail where they asked."""
    if isinstance(value, sp.Basic) and not value.free_symbols:
        return int(value) if value.is_integer else float(value)
    return value


def value_of(extent: Any) -> Any:
    """The number, for a consumer that only wants one. Accepts a bare int so a
    caller need not know whether the walk it read from was tracked."""
    return extent.value if isinstance(extent, Dim) else extent


def names_of(extent: Any) -> FrozenSet[str]:
    """The dimension symbols carried by an extent."""
    if isinstance(extent, Dim):
        return extent.names
    if isinstance(extent, sp.Basic):
        return frozenset(str(symbol) for symbol in extent.free_symbols)
    return frozenset()


def expr_of(extent: Any) -> Any:
    """The identity, or the value itself for an untracked extent — which is the
    honest answer: a bare `512` is an axis nobody named."""
    return extent.expr if isinstance(extent, Dim) else extent


def as_number(extent: Any) -> Any:
    """The value, exactly, as something JSON can carry.

    `Dim.value` is exact, and exact is not always whole. One decode token
    dispatched over `n_route_targets` destinations leaves the expert bank
    `n_experts_active*s_ctx/n_route_targets` of a token — `1/32` on
    DeepSeek-V3.1, and `M` is deliberately kept fractional there: it is the
    AVERAGE per expert, so rounding it multiplies by `n_experts` (see
    `Evaluator._tokens_per_expert`). An unbound axis in the symbolic value
    domain has no number at all.

    So the three readings, and the one function that decides between them: a
    whole value is an `int`, a fractional one is its exact string `"1/32"`, an
    unbound one is its name. Everything past this call is JSON-native, which
    is the property the callers actually need and the one nobody can restate
    correctly for themselves — a `Rational` converts through `int()` without
    raising and merely disagrees, so an is-it-whole test written as a
    try/except lets it through untouched.
    """
    v = value_of(extent)
    if type(v) is int:
        return v
    if type(v) is Fraction:
        return v.numerator if v.denominator == 1 else str(v)
    try:
        return int(v) if v == int(v) else str(v)
    except (TypeError, ValueError):
        return str(v)


def as_extent(extent: Any, *, rounding: str) -> int:
    """The whole number a consumer allocates for, under a convention it names.

    A fractional average is not a geometry: no tensor has `1/32` of a row.
    Turning one into a geometry is a decision — `floor` says the expert bank
    is empty, `ceil` says every expert got a token — and which is right
    depends on who is asking. A display column, a grouped-GEMM launch and a
    coverage total are three different questions, so `rounding` is required
    and has no default: what this replaces is a bare `int()` that made the
    choice without anyone noticing there was one.
    """
    v = value_of(extent)
    if rounding == "floor":
        n = sp.floor(v) if isinstance(v, sp.Basic) else math.floor(v)
    elif rounding == "ceil":
        n = sp.ceiling(v) if isinstance(v, sp.Basic) else math.ceil(v)
    else:
        raise ValueError(f"unknown rounding convention {rounding!r}")
    # `int()` on an unbound axis raises, and should: a geometry cannot come
    # from an extent that has no number.
    return int(n)


def floored(extent: Any) -> Any:
    """`as_extent(…, rounding="floor")` for the layout algebra, which also
    sizes extents in the workload's symbols: an expression comes back as its
    floor — a whole number at every point — where an allocation would need
    a number."""
    value = value_of(extent)
    if varies(value):
        return sp.floor(value)
    return as_extent(value, rounding="floor")


def as_wire(extent: Any) -> Dict[str, Any]:
    """JSON-safe axis for a consumer that must not rebuild identity.

    `expr` is who the axis is; `value` is the number under this binding.
    kernel_bench plan reads these and turns `value` into a tensor extent at
    launch — under its own kernel's padding convention, which is where a
    fractional `M` becomes a shape and nowhere earlier.
    """
    return {"expr": str(expr_of(extent)), "value": as_number(extent)}


# Parsed-expression cache. `parse_expr` is pure of its (string, name-set)
# inputs: which identifiers are bound as Symbols determines whether an
# identifier resolves to a dim or to a same-named parser global, so the cache
# key includes the bound names. Values get substituted per
# call via xreplace, which is O(nodes) instead of reparsing.
#
# cProfile attributed 0.79 s / `/inference/estimate` request to 777
# parse_expr calls; same dim expressions get reparsed every request
# (and 9× inside one scaling sweep). This cache turns the second+
# evaluation into a dict lookup.
_PARSE_CACHE: Dict[Tuple[str, FrozenSet[str]], sp.Expr] = {}


def algebra(value: Any) -> sp.Basic:
    """`value` as a SymPy expression, whatever it arrived as.

    An extent reaches the algebra layer as an `int`, a `Dim`, a SymPy object
    or — for a law the spec authored — a string. The first three coerce; the
    string is read by `arch.law_text`, which is also what `arch.schema` uses,
    so the question "is this name an axis or an operator" has one answer.

    Declared `-> sp.Basic` for the same reason as `exact`: dividing what this
    returns is SymPy's division, exact in either domain.
    """
    if isinstance(value, str):
        return read_law(value)
    return sp.sympify(expr_of(value) if isinstance(value, Dim) else value)


class ReadTracking(Mapping):
    """A dims mapping that remembers which keys an evaluation actually read.

    The substitution record's promise is "the bindings THIS evaluation
    substituted". Fulfilling it needs the set of keys the formula touched,
    and a dict cannot say that — so the evaluation is handed this wrapper
    and the record is built from `read` alone. Membership, iteration and
    length are not reads; only subscripting is. Callers must therefore NOT
    materialise the wrapper with `dict(...)`, and must not iterate
    `.items()` / `.values()` on it: both subscript every key and mark the
    whole environment read. A `get` on a MISSING key still marks it — the
    subscript runs before the KeyError — so consumers of `read` must filter
    against the backing map.
    """

    __slots__ = ("_inner", "read")

    def __init__(self, inner: Mapping) -> None:
        self._inner = inner
        self.read: set = set()

    def __getitem__(self, key: Any) -> Any:
        self.read.add(key)
        return self._inner[key]

    def __iter__(self):
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)

    def __contains__(self, key: object) -> bool:
        return key in self._inner


def eval_dim_expr(expr_str: str, resolved: Dict[str, Any]) -> Any:
    """
    Evaluate a SymPy expression string with resolved dimension values.
    e.g. "d_model // n_heads" with {d_model: 4096, n_heads: 32} → 128

    Expr-tolerant: when `resolved` contains all-int values, returns int
    (numeric mode) when the value is whole, and the exact `Rational` when a
    `/` does not divide. When some values are SymPy Symbols / Expressions
    (the symbolic value domain), returns the resulting
    SymPy Expression unchanged.

    Uses `parse_expr` with a strict `local_dict` so every registered dim name
    resolves to our bound value rather than a same-named SymPy builtin.
    `sympify(..., locals=...)` does NOT reliably override.
    """
    # `//` is floor division, as the author wrote it and as Python and the
    # modeling source mean it: SymPy reads `a // b` as `floor(a/b)`, so a
    # number comes out floored and a symbol keeps the `floor`. This is the
    # reading `law_text.read_law` gives the same text. A `/` stays true
    # division; a caller that rounds one names its convention (`as_extent`,
    # `exact_div`).
    safe_expr = str(expr_str)
    names = frozenset(resolved.keys())

    cache_key = (safe_expr, names)
    parsed = _PARSE_CACHE.get(cache_key)
    if parsed is None:
        from sympy.parsing.sympy_parser import parse_expr
        # Parse with Symbols-only local_dict so the cached tree is pure
        # structure; values get substituted below. This decouples
        # "what does 'd_model // n_heads' MEAN" (parse, expensive, cacheable)
        # from "evaluate it under THIS d_model/n_heads" (substitute, cheap,
        # per-call).
        symbol_dict = {n: sp.Symbol(n) for n in names}
        try:
            parsed = parse_expr(safe_expr, local_dict=symbol_dict, evaluate=True)
        except Exception as e:
            raise ValueError(
                f"Cannot evaluate dimension expression {expr_str!r}: {e}"
            ) from e
        _PARSE_CACHE[cache_key] = parsed

    # Build subs map: int → sp.Integer (avoids float promotion), Expr
    # passes through unchanged. xreplace is the right tool for
    # exact-symbol replacement — O(nodes), no pattern matching, no
    # simplification — and is materially faster than .subs() for this
    # use case.
    # Only the expression's own free symbols are looked up. A dim that sits
    # in the same dict but is not read by THIS expression is not part of
    # this evaluation — which is what lets a ReadTracking `resolved` see
    # exactly which bindings the formula consumed, and keeps an unrelated
    # tracked dim from donating its identity to the result.
    bound: Dict[Any, Any] = {}
    for sym in parsed.free_symbols:
        if sym.name in names:
            bound[sym] = resolved[sym.name]
    tracked = any(isinstance(val, Dim) for val in bound.values())
    subs_map = {}
    for sym, val in bound.items():
        if isinstance(val, Dim):
            subs_map[sym] = sp.Integer(val.value) if isinstance(
                val.value, int) else val.value
        elif isinstance(val, sp.Expr):
            subs_map[sym] = val
        else:
            # A `constant` symbol is a real number (a clamp bound of 7.5):
            # `sp.Integer` would truncate it.
            subs_map[sym] = _as_sympy_number(val)

    result = parsed.xreplace(subs_map)
    # A `/` that does not divide is an exact `Rational` — an average extent,
    # as a per-expert token count is — and `int()` would truncate it.
    if isinstance(result, sp.Rational) and not result.is_Integer:
        grounded: Any = result
    elif isinstance(result, (float, sp.Float)) and result != int(result):
        # A fractional constant stays the number it is.
        grounded = float(result)
    else:
        try:
            grounded = int(result)
        except (TypeError, ValueError):
            grounded = result

    if tracked:
        identities = {
            sym: (val.expr if isinstance(val, Dim)
                  else val if isinstance(val, sp.Expr)
                  else _as_sympy_number(val))
            for sym, val in bound.items()
        }
        return Dim(parsed.xreplace(identities), grounded)
    return grounded


def _as_sympy_number(value: Any) -> Any:
    if isinstance(value, Dim):
        value = value.value
    if isinstance(value, sp.Basic):
        return value
    if isinstance(value, bool):
        return sp.Integer(int(value))
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, float) and value == int(value):
        return sp.Integer(int(value))
    if isinstance(value, float):
        return sp.Float(value)
    return algebra(value)


def bind_keep_products(expr: Any, substitutions: Dict[Any, Any],
                       composites: Optional[Dict[Any, Any]] = None) -> Any:
    """The same expression tree with dim symbols bound, products not multiplied.

    Built at bind time from the formula that summing already kept — not by
    printing the symbolic line and splicing numbers into the text. Keys may be
    Symbols or their names; values may be anything `_as_sympy_number` coerces,
    including a `Dim` (its value half binds).

    `composites` maps a composite expression to its value — a tracked axis
    like `n_heads*d_head → 1536` whose factors have no separate per-rank
    reading (a sharded head count exists only as the product). A factor of a
    flattened product is bound as a unit when it divides the product exactly.
    """
    by_name = {
        (key.name if isinstance(key, sp.Symbol) else str(key)): value
        for key, value in substitutions.items()
    }
    composite_items = sorted(
        ((key, value) for key, value in (composites or {}).items()
         if isinstance(key, sp.Basic) and key.free_symbols),
        key=lambda item: -len(item[0].free_symbols),
    )

    def kept_product(factors: List[Any]) -> Any:
        # A `log` bound to numbers is the value the formula means: the
        # model's `log2(d)` (`log(d)/log(2)`) at `d = 128` is 7, not a ratio
        # of logs. The factors carrying a bound `log` fold to their value
        # when it is a rational number; the rest stay kept.
        logs = [i for i, f in enumerate(factors)
                if isinstance(f, sp.Basic) and f.has(sp.log)
                and not f.free_symbols]
        if logs:
            value = sp.expand_log(sp.Mul(*(factors[i] for i in logs)).doit(),
                                  force=True)
            if value.is_Rational:
                factors = [value if i == logs[0] else f
                           for i, f in enumerate(factors)
                           if i == logs[0] or i not in logs]
        return sp.Mul(*factors, evaluate=False)

    def walk(node: Any) -> Any:
        if not isinstance(node, sp.Basic):
            return node
        if isinstance(node, sp.Symbol):
            if node in substitutions:
                return _as_sympy_number(substitutions[node])
            if node.name in by_name:
                return _as_sympy_number(by_name[node.name])
            return node
        if not node.args:
            return node
        if isinstance(node, (Cases, Quotient)):
            # The cases of a value: at bound numbers the condition is a
            # truth, and the formula is the operand it picks.
            condition = node.args[0]
            truth = condition.xreplace({
                symbol: walk(symbol) for symbol in condition.free_symbols})
            if truth in (sp.true, sp.false):
                return walk(node.chosen(truth))
        if node.is_Mul and composite_items:
            residual = node
            bound_factors = []
            for c_expr, c_val in composite_items:
                if not c_expr.free_symbols <= residual.free_symbols:
                    continue
                quotient = sp.cancel(residual / c_expr)
                if quotient.free_symbols & c_expr.free_symbols:
                    continue
                bound_factors.append(_as_sympy_number(c_val))
                residual = quotient
            if residual == 1:
                args = []
            elif residual.is_Mul:
                args = [walk(arg) for arg in residual.args]
            else:
                args = [walk(residual)]
            return kept_product([*args, *bound_factors])
        # Terms and factors in the order the symbolic line prints them, so
        # each number sits under its name (`_AS_WRITTEN` keeps it).
        if node.is_Add:
            return sp.Add(*map(walk, node.as_ordered_terms()), evaluate=False)
        if node.is_Mul:
            return kept_product([walk(arg) for arg in node.as_ordered_factors()])
        args = [walk(arg) for arg in node.args]
        if node.is_Pow:
            return sp.Pow(*args, evaluate=False)
        try:
            return node.func(*args, evaluate=False)
        except TypeError:
            return node.func(*args)

    return walk(expr)


_AS_WRITTEN = sp.StrPrinter({"order": "none"})


def bound_formula_text(expr: Any, values: Dict[str, Any], expect: Any,
                       composites: Optional[Dict[Any, Any]] = None) -> Any:
    """`expr` with `values` bound (products kept), as display text.

    None when a symbol stays free or the bound form does not reproduce
    `expect` exactly — a substituted line that cannot be shown to equal the
    number beside it is worse than the number alone, so the caller falls
    back to the folded value.
    """
    if not isinstance(expr, sp.Basic) or not expr.free_symbols:
        return None
    bound = bind_keep_products(expr, values, composites)
    if bound.free_symbols:
        return None
    target = expect if isinstance(expect, sp.Basic) else sp.sympify(expect)
    # Compared EVALUATED: the kept products are unevaluated nodes, and a
    # `Max` / `Min` handed an unevaluated difference that is zero (a window
    # law at `s_window = L`) cannot decide its sign and raises. `doit()`
    # evaluates bottom-up — to a number, every symbol being bound — and the
    # text returned keeps the products.
    if bound.doit() - target != 0:
        return None
    # In the symbolic line's term order (`bind_keep_products` built it so):
    # the default printer re-sorts the numbers, and `d_qk_nope + d_qk_rope`
    # at 128 and 64 read `64 + 128`.
    return _AS_WRITTEN.doprint(bound)


def bind_record(expr: Any, record: Optional[Dict[Any, Any]],
                expect: Any) -> Any:
    """`bound_formula_text` straight off a substitution record.

    A record is keyed by identity (`expr_of` of the extent that carried the
    value): bare symbols bind as symbols, everything else as a composite.
    """
    if not record:
        return None
    symbols: Dict[str, Any] = {}
    composites: Dict[Any, Any] = {}
    for key, value in record.items():
        if isinstance(key, sp.Symbol):
            symbols[key.name] = value
        else:
            composites[key] = value
    # Each symbol at its own value first — every factor named — and the
    # axis products only where that does not reproduce `expect` (a
    # sharded head count exists only as the product).
    return (bound_formula_text(expr, symbols, expect)
            or bound_formula_text(expr, symbols, expect, composites))


def bound_or_folded(extent: Any, record: Optional[Dict[Any, Any]] = None,
                    expect: Any = None) -> str:
    """The Inspector values line for an extent.

    `bind_record` of the identity against the evaluation's substitution
    record, kept only when the bound form reproduces `expect`. Without a
    record, or on a mismatch, the folded number stands — a substituted
    line that cannot be shown to equal the figure beside it is worse
    than the number alone. Inspector node popovers, inference per-op
    FLOPs, access-row params, and playground kernel cells all render
    this string.
    """
    if expect is None:
        expect = value_of(extent)
    bound = bind_record(expr_of(extent), record, expect)
    if bound is not None:
        return str(bound)
    folded = value_of(expect)
    if folded is None:
        folded = value_of(extent)
    return "" if folded is None else str(folded)
