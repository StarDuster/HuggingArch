"""An encoder's FLOPs per media item, as the Inspector shows them: an
expression in the item's own axes, from the encode phase's symbolic walk
(`projection.formulas.encoder_item_formulas`). Bound to an item's size it is
the encode walk's number for one request carrying that one item."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.projection.formulas import encoder_item_formulas
from backend.arch.eval.transforms import Deployment
from tests.snapshots import SNAPSHOT_DIR, any_snapshot, resolve_snapshot
from tools.corpus_pins.corpus import CorpusModel, shared_loader
from tools.corpus_pins.pins.encode_surface import reference_media

pytestmark = pytest.mark.regression  # walks every snapshot with an encoder


def _encoder_snapshots():
    any_snapshot()
    loader = shared_loader()
    for path in sorted(SNAPSHOT_DIR.glob("*.json")):
        resolved = CorpusModel(path.stem).resolved(loader)
        if any(stack.is_encoder for stack in Program.of(resolved).stacks):
            yield path.stem, resolved


def test_the_formula_at_an_items_size_is_the_encode_walks_number():
    seen = 0
    for stem, resolved in _encoder_snapshots():
        program = Program.of(resolved)
        formulas = encoder_item_formulas(resolved)
        # One item of each media input (the `one` reference: one item).
        media = reference_media(program, "one")
        for stack, formula in formulas.items():
            walk = Deployment(program).at(batch=1, media=media).during(
                "encode", stack=stack).analyze()
            want = int(value_of(walk.by_stack()[stack].flops))
            given = {sp.Symbol(axis): value
                     for entry in media.values() for axis, value in entry.items()}
            got = sp.sympify(formula["per_item"].replace("^", "**"),
                             locals={str(k): k for k in given}).subs(given)
            assert int(got) == want, (stem, stack, formula["per_item"])
            assert set(formula["variables"]) <= {str(k) for k in given}
            seen += 1
    assert seen >= 12


def test_an_encoders_formula_binds_no_item_size():
    # Kimi-K2.5's tower: a polynomial in the image's processed sides.
    resolved = resolve_snapshot("moonshotai/Kimi-K2.5")
    (formula,) = encoder_item_formulas(resolved).values()
    assert formula["variables"] == ["img_h", "img_w"]
    free = {str(s) for s in sp.sympify(
        formula["per_item"].replace("^", "**")).free_symbols}
    assert free == {"img_h", "img_w"}
