"""A dimension symbol must be a name, not something SymPy already owns.

SymPy's default namespace answers `E` with Euler's number, `I` with the
imaginary unit, `S` with its singleton registry, and `beta`, `gamma`, `zeta`
with special functions. Anything that parses an authored expression through
bare `sympify` therefore reads a dim called `beta` as the beta FUNCTION, and
says nothing. Two defences, both cheap:

  * every reader of authored text binds each bare name as a Symbol before
    parsing (`eval.dim.algebra`, and `schema.spec._axis_algebra` for the one
    caller that sits below `eval`);
  * the registry itself declines those names, which is this file.

The second is what keeps the first from being the only thing standing between
a plausible dim — Kimi-K3 already configures `activation_situ_beta` — and a
law that quietly means something else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import dims
from backend.arch.law_text import LAW_OPERATORS, read_law
from backend.arch.eval.dim import algebra


def test_no_dim_symbol_is_a_law_operator():
    """A law's operators are declared (`LAW_OPERATORS`), and a name in that
    table always means the operator. An axis sharing one of those names
    would be unreachable — the reader would hand every mention to `Min`."""
    clash = sorted(set(dims.symbols()) & set(LAW_OPERATORS))
    assert clash == [], (
        f"these dim symbols are law operators and would never be read as "
        f"axes: {clash}. Rename the axis in backend/arch/dims.yaml.")


def test_a_law_may_only_apply_a_declared_operator():
    """An applied name that is not an operator is a mistake, not a function
    SymPy happens to have."""
    import pytest as _pytest
    with _pytest.raises(ValueError, match="does not have"):
        algebra("gamma(s_kv)")


def test_no_dim_symbol_is_a_sympy_builtin():
    stolen = []
    for name in dims.symbols():
        parsed = sp.sympify(name)
        if not (isinstance(parsed, sp.Symbol) and str(parsed) == name):
            stolen.append((name, type(parsed).__name__))
    assert stolen == [], (
        "these dim symbols are names SymPy resolves to something else, so a "
        "law mentioning one reads as the builtin under any bare sympify: "
        f"{stolen}. Rename the axis in backend/arch/dims.yaml.")


def test_the_reader_binds_a_stolen_name_as_an_axis_anyway():
    """The registry gate above is the belt; this is the braces. Even for a
    name SymPy owns, the reader must produce an axis."""
    for name in ("beta", "gamma", "zeta", "E", "I", "S", "N", "Q"):
        got = algebra(f"{name} * 2")
        symbols = got.free_symbols
        assert symbols == {sp.Symbol(name)}, (
            f"{name!r} parsed as {got!r} ({[type(s).__name__ for s in symbols]}), "
            "not as an axis of that name")


def test_a_declared_operator_keeps_its_meaning():
    """Cache read laws are written `Min(s_window, s_kv)` — that is what
    `dim.minimum(...)` prints as. The operator table is what makes the name
    callable; binding every name as an axis made it not."""
    got = algebra("Min(128, s_kv)")
    assert str(got) == "Min(128, s_kv)"
    assert {str(s) for s in got.free_symbols} == {"s_kv"}
    nested = algebra("Max(1, ceiling(s_kv / 128))")
    assert {str(s) for s in nested.free_symbols} == {"s_kv"}


def test_a_law_with_no_names_is_still_algebra():
    assert algebra("128 * 2") == 256
    assert algebra(8) == 8


def test_floor_division_stays_floor_division():
    """`d_model // n_heads` is what the author wrote and what `sympify` gave
    these call sites before there was a reader. SymPy parses `//` itself;
    rewriting it to `/` first would have turned every such law into true
    division without changing a number anyone was looking at."""
    assert str(read_law("a // b")) == "floor(a/b)"
    assert read_law("a // b") == sp.sympify("a // b")


def test_the_schema_and_the_engine_read_alike():
    """`arch.schema` sits below `eval` and compares authored axis text; it
    used to carry its own four-line reader, which bound `Min` as an axis and
    made `Min(a, b)` uncallable. Both now go through the same module."""
    from backend.arch.schema.spec import read_law as schema_reader
    assert schema_reader is read_law
