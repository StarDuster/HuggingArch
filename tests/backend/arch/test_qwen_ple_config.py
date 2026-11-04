"""Qwen3.8-Flash-Next's PLE n-gram layer reads its widths and layers from config.

modeling_qwen4_exp.py builds the PLE on the layers `ple_layer_ids` lists,
counted from 1 (:1268 `layer_idx + 1 in config.ple_layer_ids`); its key and
value projections read the `ple_embed_dim`-wide n-gram embedding (:1202-1203),
and its depthwise conv has kernel `ple_conv_kernel_size` (:1207-1214). A
layer's attention kind is its `layer_types` entry whether or not it has a
PLE (:1261). The snapshot's `ple_embed_dim` equals `hidden_size` and its one
PLE layer is a linear-attention layer, so the facts are checked on a config
where they differ. The table's rows per stored shard are the checkpoint
header's: a sum of primes no config field states.
"""
from __future__ import annotations

import copy
import sys

import pytest
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader, snapshot_config
from backend.hf.header import SafetensorsHeader
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment

MODEL = "Qwen/Qwen3.8-Flash-Next"
SHARD = ("model.language_model.layers.X.ple.ple_embedding.ngram_embedding"
         ".shard_0.weight")


class TestPleReadsConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = copy.deepcopy(snapshot_config(MODEL))
        text = config["text_config"]
        text.update(ple_embed_dim=1280, ple_layer_ids=[2, 4],
                    ple_conv_kernel_size=3)
        cls.hidden = text["hidden_size"]
        cls.copies = text["hc_count"]
        cls.resolved = shared_loader().match(config, model_id=MODEL)
        analysis = Deployment(Program.of(cls.resolved)).at(
            batch=1, input_tokens=8).analyze()
        cls.rows = {row.tensor.module: row for row in analysis.tensors()
                    if ".ple." in row.tensor.module}

    def _shape(self, leaf):
        row = self.rows[f"model.language_model.layers.X.ple.{leaf}"]
        return tuple(int(extent) for extent in row.tensor.shape)

    def test_ple_layers_are_the_listed_ones_with_their_own_attention(self):
        # ple_layer_ids [2, 4] counted from 1: layers 1 and 3; layer_types
        # makes layer 1 linear attention and layer 3 full attention.
        types = snapshot_config(MODEL)["text_config"]["layer_types"]
        self.assertEqual((types[1], types[3]),
                         ("linear_attention", "full_attention"))
        by_template = {}
        for template in Program.of(self.resolved).stack_blocks():
            by_template.setdefault(template.name, set()).update(
                template.layer_indices)
        self.assertEqual(by_template["qwen_hc_ple_block"], {1})
        self.assertEqual(by_template["qwen_hc_full_ple_block"], {3})
        self.assertNotIn(3, by_template["qwen_hc_full_block"])

    def test_projections_read_the_ple_width(self):
        self.assertEqual(self._shape("key_proj"),
                         (self.copies * self.hidden, 1280))
        self.assertEqual(self._shape("value_proj"), (self.hidden, 1280))

    def test_conv_kernel_is_the_configured_one(self):
        self.assertEqual(self._shape("conv1d")[-1], 3)


def test_shard_rows_are_the_headers():
    resolved = shared_loader().match(
        snapshot_config(MODEL), model_id=MODEL,
        header=SafetensorsHeader.from_tensors(
            {SHARD.replace(".X.", ".3."): ("BF16", [7, 160])}))
    assert resolved.dim_values["n_ple_rows_per_shard"] == 7
    assert resolved.dim_values["n_ple_vocab"] == 7 * resolved.dim_values["n_ple_shards"]


def test_no_header_refuses_rather_than_defaults():
    with pytest.raises(ValueError, match="no tensor metadata"):
        shared_loader().match(snapshot_config(MODEL), model_id=MODEL,
                              header=SafetensorsHeader(groups=()))


if __name__ == "__main__":
    unittest.main()
