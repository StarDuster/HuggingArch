"""A card that states the model's vocab refuses a summary without one.

The output head and the embedding print the published vocab
(`Headline.vocab`, carried as `summary.vocab_size`). A summary that lacks a
positive vocab is malformed; drawing `vocab = 0` would read like a
measurement.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.rendering import build_architecture_view
from tests.snapshots import resolve_snapshot

SNAPSHOT = "THUDM__glm-4-9b"


def _summary(vocab):
    return {"attention_info": {}, "ffn_info": {}, "hidden_size": 4096,
            "vocab_size": vocab, "num_layers": 40, "total_params": 0}


class PublishedVocab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolved = resolve_snapshot(SNAPSHOT)

    def _view(self, vocab):
        return build_architecture_view(_summary(vocab), [], self.resolved,
                                       lambda x: f"{x:,}", {})

    def test_the_head_states_the_published_vocab(self):
        vocab = Program.of(self.resolved).headline.vocab
        heads = [l["detail"] for l in self._view(vocab)["layers"]
                 if l["type"] == "lm_head"]
        self.assertEqual(heads, [f"vocab = {vocab:,}"])

    def test_a_summary_without_a_vocab_is_refused(self):
        for vocab in (None, 0):
            with self.subTest(vocab=vocab), self.assertRaisesRegex(
                    ValueError, "vocab_size"):
                self._view(vocab)


class RepeatedBankVocab(unittest.TestCase):
    """A repeated embedding bank states its own `n_vocab`; the model's
    vocab (or a 0) is no answer for a bank that binds none."""

    def _node(self, dims):
        from types import SimpleNamespace
        return SimpleNamespace(dims=dims, display_path="speech_embeddings")

    def test_the_bank_reads_its_own_vocab(self):
        from backend.rendering.builders import _own_vocab
        self.assertEqual(_own_vocab(self._node({"n_vocab": 1025})), 1025)

    def test_a_bank_without_its_own_vocab_is_refused(self):
        from backend.rendering.builders import _own_vocab
        for dims in ({}, None, {"d_model": 1024}, {"n_vocab": 0}):
            with self.subTest(dims=dims), self.assertRaisesRegex(
                    ValueError, "n_vocab"):
                _own_vocab(self._node(dims))


if __name__ == "__main__":
    unittest.main()
