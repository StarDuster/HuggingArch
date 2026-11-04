"""gpu_specs.csv lives in a shared tree other processes rewrite; a read that
races a write can see a row with more columns than the header (DictReader files
the overflow under a None restkey). Guard + retry so that transient race never
surfaces as the cryptic GPUSpec(**{None:...}) 'keywords must be strings' crash
(which the sweep VRAM gate then masked as 'model doesn't fit')."""
from __future__ import annotations

import csv

from backend.inference.gpu_specs import _CSV_PATH, _parse_row, load_gpu_specs


def _real_row() -> dict:
    with open(_CSV_PATH, encoding="utf-8") as f:
        return next(iter(csv.DictReader(f)))


def test_parse_row_ignores_none_restkey():
    row = dict(_real_row())
    row[None] = ["overflow", "cols"]   # what DictReader does on an extra-column row
    spec = _parse_row(row)             # must NOT raise "keywords must be strings"
    assert spec.sku


def test_load_gpu_specs_nonempty_and_has_4090():
    specs = load_gpu_specs()
    assert specs and any(s.sku == "RTX-4090" for s in specs)
