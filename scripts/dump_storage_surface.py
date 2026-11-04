"""Every on-disk tensor `Analysis.tensors()` reports, over every snapshot.

The checkpoint surface: name, shape, elements, dtype, bytes, kind, copies.
Metadata siblings hang off the format and are flattened here so a dump
line is one safetensors entry.

The rows come from pin P10 (`tools/corpus_pins/pins/gate_surfaces.py`), which
`tests/backend/arch/test_gate_surfaces_golden.py` compares against
`tests/golden/storage_surface.txt`; this script prints them for every snapshot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_pins.corpus import Corpus
from tools.corpus_pins.pins.gate_surfaces import STORAGE_SURFACE

if __name__ == "__main__":
    for model in Corpus.models():
        for row in STORAGE_SURFACE.rows(model):
            print(row)
