"""Every weight's checkpoint name and element count, over every snapshot.

The invariant a naming or IR refactor must hold: whatever the spec graph is
called, the tensors it says are on disk do not move. Dumped as a sorted text
file so a diff names exactly which one did.

The rows come from pin P9 (`tools/corpus_pins/pins/gate_surfaces.py`), which
`tests/backend/arch/test_gate_surfaces_golden.py` compares against
`tests/golden/tensor_names.txt`; this script prints them for every snapshot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_pins.corpus import Corpus
from tools.corpus_pins.pins.gate_surfaces import TENSOR_NAMES

if __name__ == "__main__":
    for model in Corpus.models():
        for row in TENSOR_NAMES.rows(model):
            print(row)
