"""The formula payload is a display payload — it must stay JSON-encodable.

The producer's internal symbol→value substitution map is deliberately
EXACT: plain ints for most dims, exact `Rational`s where a value is a
fraction. That exactness exists to feed `Expr.subs()` without float
drift, and it means the map holds SymPy numbers, which are NOT
JSON-encodable.

Regression this locks: the payload once dumped that raw map into the
response as `_values`, stringifying only the Symbol KEYS. Once
the phase's fraction became an independent symbol, a SymPy value reached
FastAPI's encoder — `TypeError: 'One' object is not iterable` at seq_len=1
(Rational(2,2) → One) — and /analyze returned 500 for all 32 supported
models. The fix was structural: the internal map is not exposed. The
rendered figures callers want are already in the rendered payloads, with
values substituted.

So the contract is: whatever `display_formulas` returns must survive
`json.dumps`, and no SymPy object may appear anywhere in it.
"""
import json
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import sympy as sp

from backend.arch.loader import get_default_loader
from tests.backend.inference.test_capacity_golden import _snapshot_config

pytestmark = pytest.mark.regression  # walks every snapshot


def _walk(obj, path=""):
    """Every SymPy object reachable in `obj`, with its JSON path."""
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if isinstance(k, sp.Basic):
                out.append((f"{path}[key]", repr(k)))
            out.extend(_walk(v, f"{path}.{k}"))
        return out
    if isinstance(obj, (list, tuple)):
        return [h for i, v in enumerate(obj) for h in _walk(v, f"{path}[{i}]")]
    if isinstance(obj, sp.Basic):
        return [(path, repr(obj))]
    return []


class TestFormulasPayloadContract(unittest.TestCase):
    _SNAP = "meta-llama__Llama-3.2-3B.json"

    def _formulas(self, seq_len):
        cfg = _snapshot_config(self._SNAP)
        resolved = get_default_loader().match(
            cfg, model_id=Path(self._SNAP).stem.replace("__", "/", 1))
        from backend.arch.eval.projection import display_formulas
        return display_formulas(resolved, batch_size=1, seq_len=seq_len)

    def test_payload_is_json_encodable(self):
        # seq_len=1 is the case that collapsed the phase's fraction to SymPy
        # `One`; 128 leaves the pairs a large exact number. Both must
        # serialize.
        for seq_len in (1, 128):
            with self.subTest(seq_len=seq_len):
                payload = self._formulas(seq_len)
                leaks = _walk(payload)
                self.assertEqual(
                    leaks, [],
                    f"SymPy objects leaked into the formulas payload: {leaks}. "
                    f"The internal substitution map must not be exposed; "
                    f"render(...) already carries substituted display values.",
                )
                json.dumps(payload)

    def test_internal_substitution_map_is_not_exposed(self):
        # Named explicitly so a re-add of `_values` (or any alias of the raw
        # map) trips here with the reason, not just on a serialization crash.
        payload = self._formulas(128)
        self.assertNotIn("_values", payload)

    def test_display_payloads_still_carry_substituted_values(self):
        # The flip side: dropping the raw map must not cost callers the numbers.
        payload = self._formulas(128)
        for key in ("attention", "ffn", "lm_head", "total"):
            self.assertIn(key, payload)
            self.assertTrue(
                payload[key].get("with_values"),
                f"{key} lost its substituted display string.")


class TestNoScopeLocalSymbolSurvives(unittest.TestCase):
    """`with_values` means every symbol got a value. It has to be true of the
    whole corpus, not of a model someone checked.

    A `Dim` holds an identity and a value. The identity is scope-local — a
    linear's `out`, the dispatch zone's `M` and `s_expert`, a head-split's
    `dim` — and sympy's `sympify` protocol hands back exactly that half, so
    `sp.Symbol("B") * cost.flops` substituted a map keyed on the model's dim
    vocabulary against an expression written in none of it.

    All 39 snapshots printed at least one at 4096 tokens: `out` in 78 entries
    across all 39 models, `M` and `s_expert` in 48 across the 24 MoE models,
    `dim` in 20 across 9, `in` in 5. Hand-picking a model would not have found
    it either, since it was universal — what hid it was that the NUMBER beside
    the formula is the walk's law half read at the point, and stayed right.
    """

    #: sympy prints these as identifiers; they are operators, not axes.
    CALLABLE = frozenset({"Max", "Min", "ceiling", "floor", "Abs", "sqrt",
                          "Piecewise", "log", "exp", "Rational", "Integer",
                          "Float"})
    #: the memory entry whose `with_values` is prose by design.
    PROSE = frozenset({"memory.kv_cache"})

    def test_every_model_substitutes_every_symbol(self):
        import re
        from backend.arch.eval.projection import display_formulas

        ident = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
        snapshots = sorted(
            (PROJECT_ROOT / "backend" / "arch" / "snapshots").glob("*.json"))
        self.assertGreater(len(snapshots), 30,
                           "this test is a corpus sweep — the defect it locks "
                           "was present on every model at once")
        checked = 0
        loader = get_default_loader()
        for path in snapshots:
            with self.subTest(model=path.stem):
                cfg = json.loads(path.read_text()).get("config")
                resolved = loader.match(
                    cfg, model_id=path.stem.replace("__", "/", 1)) if cfg else None
                payload = display_formulas(resolved, batch_size=1, seq_len=4096)
                entries = {k: payload[k]
                           for k in ("attention", "ffn", "lm_head", "total")}
                entries.update({f"memory.{k}": v for k, v
                                in (payload.get("memory") or {}).items()})
                checked += 1
                for key, entry in entries.items():
                    if key in self.PROSE:
                        continue
                    left = {n for n in ident.findall(entry["with_values"])
                            if n not in self.CALLABLE}
                    self.assertEqual(
                        left, set(),
                        f"{path.stem} {key}: {sorted(left)} still symbols in "
                        f"{entry['with_values'][:120]!r} — the substitution map "
                        f"is keyed on the model's dim vocabulary, so an "
                        f"expression carrying a scope-local identity cannot be "
                        f"bound by it")
        self.assertGreater(checked, 30, "every model skipped; nothing checked")


if __name__ == "__main__":
    unittest.main()
