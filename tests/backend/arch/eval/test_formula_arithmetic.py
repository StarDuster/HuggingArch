"""A formula's substituted text is the arithmetic of the number beside it.

The dashboard prints each formula three ways: the expression, the expression
with every symbol given its value (`with_values`), and the result. All three
come from one walk over the workload's counts: the expression is the walk's
identity halves (the model's names), `with_values` is the same terms each
bound by its node's frame, and the result is the walk read at the request's
point. Evaluating `with_values` must give the result, or the tooltip shows
the reader a sum that does not add up.

That holds for every field the dashboard prints — each compute entry and
`memory.weights` — on every model. Two ways it has failed: a routed bank
whose batch was written once in the formula and again inside the bank's
FLOPs (Qwen3-30B-A3B's `ffn` at batch 2 evaluated to 931,417,128,960 beside
a result of 467,371,917,312), and the weights formula summing the rows the
model stores while its result summed the checkpoint file's, which also holds
the redundant copies of tied weights (HunyuanOCR's evaluated to
1,078,020,096 beside 1,325,455,360). The synthetic one-pool MoE at batch 2 is
the smallest spec where the first can differ; the corpus sweep holds both.
"""
from __future__ import annotations

import shutil
import tempfile
import re
from pathlib import Path

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from backend.arch.snapshots import iter_snapshot_ids
from tests.snapshots import resolve_snapshot

COMPONENTS = Path(__file__).resolve().parents[4] / "backend" / "arch" / "components"
CONFIG = {
    "model_type": "formula_moe_toy",
    "torch_dtype": "bfloat16", "hidden_size": 8,
    "moe_intermediate_size": 4,
    "num_experts": 12,
    "num_experts_per_tok": 4,
    "vocab_size": 32,
}

MODEL = """\
match: { model_type: [formula_moe_toy], model_id: [formula_moe_toy] }
dims:
  d_model: config.hidden_size
  d_ffn_expert: config.moe_intermediate_size
  d_ffn_shared: config.moe_intermediate_size
  n_shared_experts: "1"
  n_vocab: config.vocab_size
  n_layers: "2"
  n_experts: config.num_experts
  n_experts_active: config.num_experts_per_tok
blocks:
  blk:
    inputs: [x, input_ids]
    graph:
      ffn: { op: dsv4_moe_sigmoid, in: [x, input_ids] }
      o: { op: add, in: [x, ffn], dim: d_model }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: layers, repeat: n_layers, template: blk, inputs: [embed_tokens, input_ids],
      streams: { hidden: { carries: true } } }
  - { name: norm, op: rmsnorm, dim: d_model, inputs: [layers],
      weight_axes: [{ replicated: true }] }
  - { name: lm_head, op: linear, boundary_role: output_head, inputs: [norm],
      dims: { in: d_model, out: n_vocab },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""

_FUNCTIONS = {"floor": sp.floor, "ceiling": sp.ceiling, "Min": sp.Min,
              "Max": sp.Max, "sqrt": sp.sqrt}


def _resolve():
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        shutil.copytree(COMPONENTS / "library", base / "components" / "library")
        for sub in ("components/adapters", "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "formula_moe_toy.yaml").write_text(MODEL)
        return SpecLoader(base_dir=str(base)).match(
            CONFIG, model_id="formula_moe_toy")


# A thousands separator sits between digits with exactly three digits after
# it; an argument separator (`Min(a, b)`) is followed by a space.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _number(text: str) -> int:
    return int(_THOUSANDS.sub("", text))


def _fields(formulas):
    """Every printed formula, by its dotted key: each compute entry and each
    memory entry."""
    for key, entry in formulas.items():
        if isinstance(entry, dict) and "with_values" in entry:
            yield key, entry
    for key, entry in (formulas.get("memory") or {}).items():
        if isinstance(entry, dict) and "with_values" in entry:
            yield f"memory.{key}", entry


def _mismatches(formulas):
    """Each field whose `with_values` does not evaluate to its result."""
    bad = []
    for key, entry in _fields(formulas):
        result = _number(entry["result"])
        text = parse_expr(_THOUSANDS.sub("", entry["with_values"])
                          .replace("^", "**"), local_dict=_FUNCTIONS)
        if text.free_symbols or sp.nsimplify(text) != result:
            bad.append(f"{key}: `with_values` evaluates to {text}, the "
                       f"result beside it is {result:,}")
    return bad


@pytest.fixture(scope="module")
def formulas():
    from backend.arch.eval.projection import display_formulas
    return display_formulas(_resolve(), batch_size=2, seq_len=8)


def test_every_formula_adds_up_to_its_result(formulas):
    keys = [key for key, _ in _fields(formulas)]
    assert {"ffn", "total", "memory.weights"} <= set(keys), keys
    assert _mismatches(formulas) == []


def test_tied_weights_copies_are_in_the_weights_formula():
    """HunyuanOCR's checkpoint stores the tied embedding twice; the result
    counts the file, so the formula has to hold the copy too."""
    from backend.arch.eval.projection import display_formulas
    got = display_formulas(resolve_snapshot("tencent/HunyuanOCR"),
                           batch_size=2, seq_len=1024)
    assert _mismatches(got) == []


@pytest.mark.regression  # the whole corpus, a dashboard each
@pytest.mark.parametrize("model_id", sorted(iter_snapshot_ids()))
def test_every_models_formulas_add_up(model_id):
    from backend.arch.eval.projection import display_formulas
    assert _mismatches(display_formulas(resolve_snapshot(model_id),
                                        batch_size=2, seq_len=1024)) == []
