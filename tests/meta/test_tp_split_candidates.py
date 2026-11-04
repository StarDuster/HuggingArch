"""D1 (`tools/corpus_pins/decisions/tp_split.py`) asks a person how vLLM
splits every weight node a model's admission brings. A node of a block the
model file writes inline is one of them in whichever stack the block runs:
its `template` label carries the stack (`mimo_mtp_block@mtp`), which is
not the name the model file declares the block under."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.decisions.tp_split import node_table


def test_an_inline_block_in_an_aux_stack_brings_its_weight_nodes():
    table = node_table(CorpusModel("XiaomiMiMo__MiMo-7B-RL"))
    assert "mimo_mtp_block@mtp|input_proj" in table["mtp"]
