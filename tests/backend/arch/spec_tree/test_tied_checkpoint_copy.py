"""A tie whose checkpoint still ships the tied layer's own tensor.

`tied_weight.checkpoint_copy` says the file carries the reader's tensor as a
redundant copy the serving loader skips (HunyuanOCR's `lm_head.weight`, which
vLLM drops when `tie_word_embeddings`). The tie keeps compute, parameters and
deployed weight bytes; the copy counts where the file is the subject — the
checkpoint byte total and the per-tensor name join. The validate check holds
the declaration to the checkpoint's names in both directions.
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import pydantic

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import capacity
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.loader import SpecLoader
from backend.arch.schema.model import TiedWeightSpec
from backend.arch.validate._tied_copy import run_tied_copy_check
from backend.hf.header import SafetensorsHeader

H, V = 8, 32
_SRC = "vllm/model_executor/models/example.py:10 skip_prefixes=['head.']"


def _analysis(tie: str):
    yaml_text = textwrap.dedent(f"""\
        match: {{ model_type: [test_tied_copy], model_id: [test_tied_copy] }}
        dims:
          d_model: config.hidden_size
          n_vocab: config.vocab_size
        blocks:
          fixed:
            graph:
              n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
            outputs: [n]
        model_inputs:
          input_ids: {{ kind: tokens }}
        model:
          - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
              dims: {{ n_vocab: n_vocab, d_model: d_model }},
              weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
          - {{ name: layers, repeat: "1", template: fixed, inputs: [embed],
              streams: {{ hidden: {{ carries: true }} }} }}
          - {{ name: head, op: linear, boundary_role: output_head, inputs: [layers],
              dims: {{ in: d_model, out: n_vocab }},
              weight_axes: [{{ replicated: true }}, {{ replicated: true }}]{tie} }}
    """)
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "test_tied_copy.yaml").write_text(yaml_text)
        loader = SpecLoader(base_dir=str(base))
        resolved = loader.match(
            {"model_type": "test_tied_copy", "hidden_size": H,
             "vocab_size": V, "tie_word_embeddings": True,
             "torch_dtype": "bfloat16"},
            model_id="test_tied_copy")
        return Deployment(Program.of(resolved)).at(
            batch=1, input_tokens=4).analyze()


def _params(analysis) -> int:
    return int(value_of(sum(
        (c.params for c in analysis.cost.values.values()), 0)))


UNTIED = ""
TIED = ", tied_weight: { to: embed, when: config.tie_word_embeddings }"
TIED_COPY = (", tied_weight: { to: embed, when: config.tie_word_embeddings, "
             f"checkpoint_copy: {{ src: \"{_SRC}\" }} }}")
HEAD_BYTES = H * V * 2  # BF16


class TestTheCopyIsTheFilesNotTheModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.untied = _analysis(UNTIED)
        cls.tied = _analysis(TIED)
        cls.copy = _analysis(TIED_COPY)

    def test_parameters_and_resident_bytes_follow_the_tie(self):
        self.assertEqual(_params(self.copy), _params(self.tied))
        self.assertEqual(_params(self.untied) - _params(self.tied), H * V)
        self.assertEqual(
            int(value_of(capacity.weights(self.copy).parameters)),
            int(value_of(capacity.weights(self.tied).parameters)))
        self.assertEqual(
            {row.name for row in self.copy.tensors()},
            {row.name for row in self.tied.tensors()})

    def test_the_checkpoint_counts_the_stored_copy(self):
        self.assertEqual(self.copy.checkpoint_param_bytes(),
                         self.untied.checkpoint_param_bytes())
        self.assertEqual(self.copy.checkpoint_param_bytes()
                         - self.tied.checkpoint_param_bytes(), HEAD_BYTES)
        self.assertEqual([row.name for row in self.copy.checkpoint_copy_rows()],
                         ["head.weight"])
        self.assertIn("head.weight",
                      {row.name for row in self.copy.iter_checkpoint()})
        self.assertNotIn("head.weight",
                         {row.name for row in self.tied.iter_checkpoint()})

    def test_the_head_still_reads_the_embedding(self):
        program = self.copy.program
        head = next(n for n in program.nodes() if n.tied_to is not None)
        self.assertTrue(head.stores_checkpoint_copy)
        self.assertEqual(program.read_aliases,
                         {head.uid: {head.tied_to_uid: 1}})


class TestTheDeclarationMatchesTheCheckpoint(unittest.TestCase):
    WITH_COPY = SafetensorsHeader.from_tensors({
        "embed.weight": ("BF16", (V, H)), "layers.0.n.weight": ("BF16", (H,)),
        "head.weight": ("BF16", (V, H))})
    WITHOUT_COPY = SafetensorsHeader.from_tensors({
        "embed.weight": ("BF16", (V, H)), "layers.0.n.weight": ("BF16", (H,))})

    @classmethod
    def setUpClass(cls):
        cls.tied = _analysis(TIED)
        cls.copy = _analysis(TIED_COPY)

    def _kinds(self, analysis, header):
        return [f.kind for f in run_tied_copy_check(analysis, header)]

    def test_agreeing_declarations_are_clean(self):
        self.assertEqual(self._kinds(self.copy, self.WITH_COPY), [])
        self.assertEqual(self._kinds(self.tied, self.WITHOUT_COPY), [])

    def test_a_declared_copy_the_file_lacks_is_refused(self):
        findings = run_tied_copy_check(self.copy, self.WITHOUT_COPY)
        self.assertEqual([f.kind for f in findings],
                         ["tied_checkpoint_copy_absent"])
        self.assertIn("delete `checkpoint_copy`", findings[0].detail)

    def test_an_undeclared_copy_the_file_ships_is_refused(self):
        findings = run_tied_copy_check(self.tied, self.WITH_COPY)
        self.assertEqual([f.kind for f in findings],
                         ["tied_checkpoint_copy_undeclared"])
        self.assertIn("checkpoint_copy: { src:", findings[0].detail)
        self.assertIn("'head.weight'", findings[0].detail)

    def test_the_copy_cites_its_loader_line(self):
        with self.assertRaises(pydantic.ValidationError):
            TiedWeightSpec(to="embed", checkpoint_copy={"src": " "})
        with self.assertRaises(pydantic.ValidationError):
            TiedWeightSpec(to="embed", checkpoint_copy={})


if __name__ == "__main__":
    unittest.main()
