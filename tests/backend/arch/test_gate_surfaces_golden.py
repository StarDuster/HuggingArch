"""The two gate surfaces, compared by pytest: every snapshot's checkpoint
names (`tests/golden/tensor_names.txt`) and its storage surface
(`tests/golden/storage_surface.txt`).

`scripts/gate.sh` diffs the same two files in the inner loop, and
`update-transformers.yml` diffs them when the dependency moves. Neither runs
on a pull request, so a model whose names or bytes moved, or a new snapshot
with no rows, reached main unseen. Here each file is read as one block of rows
per snapshot: the set of blocks must be the set of snapshots, and each block
must equal what pins P9 / P10 (`tools/corpus_pins/pins/gate_surfaces.py`)
measure for that snapshot today. `scripts/dump_tensor_names.py` and
`dump_storage_surface.py` print the same rows, so the file, the scripts and
this test agree by construction.

Regenerating (an intended change, in its own commit, every moved row
explained):

    .venv/bin/python scripts/dump_tensor_names.py > tests/golden/tensor_names.txt
    .venv/bin/python scripts/dump_storage_surface.py > tests/golden/storage_surface.txt
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.corpus_pins.corpus import Corpus, CorpusModel
from tools.corpus_pins.pins.gate_surfaces import STORAGE_SURFACE, TENSOR_NAMES

pytestmark = pytest.mark.regression  # walks every snapshot

SURFACES = {pin.path.name: pin for pin in (TENSOR_NAMES, STORAGE_SURFACE)}

STEMS = Corpus.stems()


def _blocks(golden: str) -> Dict[str, List[str]]:
    """The golden's rows grouped by snapshot stem, in file order; a
    snapshot's rows must be one contiguous block."""
    return SURFACES[golden].load()


@pytest.mark.parametrize("golden", sorted(SURFACES))
def test_every_snapshot_has_a_block(golden: str) -> None:
    blocks = _blocks(golden)
    missing = sorted(set(STEMS) - set(blocks))
    orphans = sorted(set(blocks) - set(STEMS))
    assert not missing and not orphans, (
        f"{golden}: snapshots with no rows {missing}; rows with no snapshot {orphans}")


@pytest.mark.parametrize("stem", STEMS)
@pytest.mark.parametrize("golden", sorted(SURFACES))
def test_block_equals_measurement(golden: str, stem: str) -> None:
    pin = SURFACES[golden]
    row = pin.measure(CorpusModel(stem))
    failed = pin.failures(row)
    assert not failed, f"{stem}: does not analyze: {failed}"
    measured = row[stem]
    pinned = _blocks(golden).get(stem, [])
    assert measured == pinned, (
        f"{golden} {stem}: + {sorted(set(measured) - set(pinned))[:10]} "
        f"- {sorted(set(pinned) - set(measured))[:10]}"
        + ("" if collections.Counter(measured) != collections.Counter(pinned)
           else " (same rows, different order)"))
