"""An extent's questions are answered by the value, not by what holds it.

`is_whole`, `is_zero` and `is_unit` each compare an extent's value against a
number, and SymPy reads such a comparison STRUCTURALLY: `Float(2.0) ==
Integer(2)` is False, and `Float(2.0).is_integer` is None rather than False —
the algebra declines to decide the integrality of an approximation. So every
one of the three used to be decided by the value's CARRIER instead of by the
value, and the same number answered one way as a Python `float` (where
`int(v) == v` reads arithmetically) and the other way as a SymPy `Float`.

What that cost is not hypothetical. `layout.geometry` turns a split's factors
into a layout only when each is whole, so a whole factor carried as a `Float`
was reported an AVERAGE extent and the axis was forced replicated or the walk
raised "split into average extents"; `layout.physical_cost` divides a cache
view's sequence axis by its landed rows and refused a perfectly dividing axis
with "does not divide into local rows". Answering False is the one outcome
that neither answers nor refuses.

A float is refused rather than converted (`dim._exact_extent`, the one door
all three now read their value through): `Rational` of a float is exact about
the float, and the floats this walk makes have been the true value less
6e-13. The door is exactly that wide and no wider — what is genuinely not a
whole number arrives exactly, as the `Rational` average 138 of a sweep's
13,426 `is_whole` calls carry, and still answers.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import (Dim, Undecided, _exact_extent, is_unit,
                                   is_whole, is_zero)

#: Every carrier this walk hands an extent predicate, by the number it holds.
#: A `Dim`'s value half is what is asked (`value_of`), so each appears bare
#: and wrapped.
EXACT = {
    2: (2, sp.Integer(2)),
    0: (0, sp.Integer(0)),
    1: (1, sp.Integer(1)),
    Fraction(5, 2): (Fraction(5, 2), sp.Rational(5, 2)),
    Fraction(1, 2): (Fraction(1, 2), sp.Rational(1, 2)),
}

#: The three predicates, each with what it should answer about a number.
QUESTIONS = (
    ("is_whole", is_whole, lambda n: n.denominator == 1),
    ("is_zero", is_zero, lambda n: n == 0),
    ("is_unit", is_unit, lambda n: n == 1),
)


def _cases():
    for number, carriers in EXACT.items():
        exact = Fraction(number)
        for carrier in carriers:
            for name, question, truth in QUESTIONS:
                yield number, carrier, name, question, truth(exact)


@pytest.mark.parametrize(
    "number,carrier,name,question,expected",
    [pytest.param(*case, id=f"{case[2]}({case[1]!r}:"
                            f"{type(case[1]).__name__})")
     for case in _cases()])
def test_an_exact_carrier_answers_about_its_number(
        number, carrier, name, question, expected):
    """One number, every exact carrier of it, one answer — and the answer is
    the number's. A `Fraction` and a `Rational` of the same ratio are the two
    exact carriers an average extent arrives in (`as_number` publishes both),
    and `int` and `Integer` the two a count arrives in."""
    assert question(carrier) is expected
    assert question(Dim(sp.Symbol("a"), carrier)) is expected


FLOATS = (
    ("a whole count", 2.0, sp.Float(2.0)),
    ("zero", 0.0, sp.Float(0.0)),
    ("a unit axis", 1.0, sp.Float(1.0)),
    ("a fraction", 2.5, sp.Float(2.5)),
)


@pytest.mark.parametrize("what,python,sympy_float",
                         [pytest.param(*f, id=f[0].replace(" ", "-"))
                          for f in FLOATS])
@pytest.mark.parametrize("name,question,_truth", QUESTIONS,
                         ids=[q[0] for q in QUESTIONS])
def test_a_float_is_refused_by_every_extent_question(
        name, question, _truth, what, python, sympy_float):
    """A float is not an extent, so each door says so instead of answering.

    Both carriers, because refusing one and converting the other is the
    carrier dependence again: `int(2.0) == 2.0` answered True for the Python
    float while `Integer(2) == Float(2.0)` answered False for the SymPy one,
    which is one value with two answers.
    """
    for carrier in (python, sympy_float):
        with pytest.raises(TypeError, match="is not an extent"):
            question(carrier)
        with pytest.raises(TypeError, match="is not an extent"):
            question(Dim(sp.Symbol("a"), carrier))


def test_a_law_with_a_float_coefficient_is_refused_too():
    """The same defect in the symbolic binding, where it is quieter.

    `floor(s/4)` is whole at every point and `is_whole` says so. Multiplied
    by a `Float` coefficient of one it is the SAME law and the same object's
    value, and the structural reading said it was not whole — so the numeric
    and the symbolic binding of one walk (CLAUDE.md §2.1) disagreed about a
    cut's legality depending on which was running.
    """
    s = sp.Symbol("s_ctx", integer=True, positive=True)
    assert is_whole(sp.floor(s / 4)) is True
    assert is_whole(s / 4) is False
    with pytest.raises(TypeError, match="is not an extent"):
        is_whole(sp.Float(1.0) * sp.floor(s / 4))


#: The average extents one corpus sweep actually hands `geometry.py:106` —
#: 138 of its 13,426 `is_whole` calls, a window's share of a compressor
#: ratio — each as the exact carrier the extent algebra holds it in.
AVERAGES = (sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 128),
            sp.Rational(5, 2), Fraction(1, 2), Fraction(5, 2))


@pytest.mark.parametrize("average", AVERAGES, ids=[str(a) for a in AVERAGES])
def test_the_door_admits_an_average_extent_and_answers_it_not_whole(average):
    """The door is exactly as wide as "a float is not an extent", and no
    wider.

    An exact fraction of a count IS an extent — a MoE dispatch leaves the
    expert bank `1/32` of a token, and a compressor window leaves `1/128` of
    one — and `layout.geometry` depends on these answering False, because
    False is what routes an average extent to a replicated axis instead of a
    cut. A door that refused every non-integer would turn those 138 calls
    into a crash on exactly the models that reach them, so the carrier is
    handed back untouched and the question answers about the number.
    """
    assert _exact_extent(average) is average
    assert _exact_extent(Dim(sp.Symbol("a"), average)) is average
    assert is_whole(average) is False
    assert is_zero(average) is False
    assert is_unit(average) is False


def test_the_door_admits_every_number_the_census_saw():
    """Counts as `int` and `Integer`, a zero, and averages as `Rational` and
    `Fraction`: each handed back untouched, and each question answered."""
    for carrier in (0, 1, 2, 128, sp.Integer(0), sp.Integer(1), sp.Integer(8),
                    sp.Rational(1, 128), Fraction(1, 2), Fraction(2, 1)):
        assert _exact_extent(carrier) is carrier
        for _name, question, _truth in QUESTIONS:
            assert isinstance(question(carrier), bool)


def test_the_door_admits_every_law_the_census_saw():
    """The laws the symbolic binding hands these doors — `s_input`,
    `Max(1, s_input)`, `2*s_input`, `floor(Max(1, s_input)/128)` — which is
    350,000 of the 535,000 calls a corpus sweep makes.

    Past the door each question answers as it always did: whole by
    construction or not (`is_whole`), a unit axis never (`is_unit`, since an
    extent that grows with the workload is not one at any point), and
    `is_zero` raises `Undecided` — the WALK's refusal, that it decides
    nothing about the workload, which has to keep arriving as itself rather
    than as the door's `TypeError`.
    """
    s = sp.Symbol("s_input", integer=True, nonnegative=True)
    laws = (s, sp.Max(1, s), 2 * s, sp.floor(sp.Max(1, s) / 128),
            sp.Min(1, s), s / 4)
    for law in laws:
        assert _exact_extent(law) is law
        assert isinstance(is_whole(law), bool)
        assert is_unit(law) is False
        with pytest.raises(Undecided):
            is_zero(law)
    assert is_whole(sp.floor(sp.Max(1, s) / 128)) is True
    assert is_whole(2 * s) is True
    assert is_whole(s / 4) is False


def test_the_refusal_says_where_to_divide_exactly():
    """The message names the verbs that keep a quotient exact, because the
    float is made by a division and fixed there (`layout.physical.
    pairs_share`: two int counts under Python's `/`)."""
    with pytest.raises(TypeError) as caught:
        is_whole(sp.Float(4224.0))
    message = str(caught.value)
    assert "exact_div" in message and "ratio" in message
    assert "4224" in message
