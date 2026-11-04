"""Which head shares which embedding is the layer's own declaration.

`tied_weight: { to: <layer>, when: <config flag> }` says it in the spec.
The IR used to keep only the boolean (`owns_weight`) and the read side
re-derived the partner from `config.tie_word_embeddings` plus a hunt for
`boundary_role: input_embedding` — the same fact, twice, with a proxy in
the middle. The name is now resolved to a uid once at build, and every
consumer asks that.
"""
from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader
from backend.arch.eval.analysis import pair_stored
from backend.arch.eval.program import Program
from backend.arch.eval.traffic import derive
from backend.arch.eval.transforms import Deployment

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _program(stem: str) -> Program:
    config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
    return Program.of(SpecLoader().match(
        config, model_id=stem.replace("__", "/", 1)))


class TestTheHeadNamesItsEmbedding(unittest.TestCase):
    TIED = ("meta-llama__Llama-3.2-1B", "google__gemma-2-9b")

    def test_tied_to_resolves_to_the_named_layers_uid(self):
        for stem in self.TIED:
            with self.subTest(model=stem):
                program = _program(stem)
                tied = [n for n in program.nodes() if n.tied_to is not None]
                self.assertEqual(len(tied), 1)
                head = tied[0]
                self.assertEqual(head.boundary_role, "output_head")
                partner = program.by_uid(head.tied_to_uid)
                self.assertEqual(partner.display_path, head.tied_to)
                self.assertEqual(partner.op, "embedding")
                self.assertFalse(head.owns_weight)
                self.assertTrue(partner.owns_weight)
                self.assertEqual(program.read_aliases,
                                 {head.uid: {partner.uid: 1}})

    def test_the_tied_head_streams_the_embedding_table(self):
        """gemma-2 ties without a `tie_word_embeddings` key in its config:
        the declaration alone puts the table's bytes on the head's row."""
        program = _program("google__gemma-2-9b")
        analysis = Deployment(program).at(batch=1, input_tokens=8).analyze()
        head = next(n for n in program.nodes() if n.tied_to is not None)
        table = sum(
            row.rep.resident.weight_bytes_for(row.tensor.elements)
            for row in pair_stored(
                head.tied_to_uid,
                analysis.storage.values.get(head.tied_to_uid),
                analysis.formats.values.get(head.tied_to_uid))
            if row.rep is not None)
        self.assertGreater(table, 0)
        self.assertEqual(
            int(derive(analysis).values[head.uid].weight_reads), int(table))

    def test_an_untied_model_declares_nothing(self):
        program = _program("meta-llama__Llama-2-7b-hf")
        self.assertEqual([n.name for n in program.nodes()
                          if n.tied_to is not None], [])
        self.assertEqual(program.read_aliases, {})


class TestANameTheModelListDoesNotDeclareIsAnError(unittest.TestCase):
    """The schema refuses the spec at load — the one gate on the name;
    `resolve_tied_weights` is the invariant behind it, unreachable from a
    loaded spec and loud if it ever is."""

    def test_unknown_target_is_refused_at_load(self):
        yaml_text = textwrap.dedent("""\
            match: { model_type: [test_tied_nowhere], model_id: [test_tied_nowhere] }
            dims:
              H: config.hidden_size
            blocks:
              fixed:
                graph:
                  n: { op: rmsnorm, in: x, dim: H }
                outputs: [n]
            model_inputs:
              input_ids: { kind: tokens }
            model:
              - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
                  dims: { n_vocab: H, d_model: H } }
              - { name: layers, repeat: "1", template: fixed }
              - { name: head, op: linear, boundary_role: output_head,
                  dims: { in: H, out: H }, tied_weight: { to: nowhere } }
        """)
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for sub in ("components/library", "components/adapters",
                        "components/drafts", "models"):
                (base / sub).mkdir(parents=True)
            (base / "models" / "test_tied_nowhere.yaml").write_text(yaml_text)
            loader = SpecLoader(base_dir=str(base))
            with self.assertRaisesRegex(
                    Exception, "not an earlier top-level model layer"):
                Program.of(loader.match(
                    {"model_type": "test_tied_nowhere", "hidden_size": 8},
                    model_id="test_tied_nowhere"))


if __name__ == "__main__":
    unittest.main()
