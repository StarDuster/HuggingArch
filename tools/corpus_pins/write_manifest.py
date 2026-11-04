"""Rewrite `tests/meta/corpus_literal_manifest.json` from the scan, every
row `curated`. Review the diff before committing: a new row is a new curated
set, and it needs its `CuratedSet` entry (with the reason it does not grow
with the corpus) in `tools/corpus_pins/curated.py`.

    .venv/bin/python -m tools.corpus_pins.write_manifest
"""
from __future__ import annotations

from tools.corpus_pins.curated import (
    MANIFEST, corpus_identifiers, literal_hits, module_paths, render_manifest)

if __name__ == "__main__":
    hits = literal_hits(module_paths(), corpus_identifiers())
    MANIFEST.write_bytes(render_manifest([hit.row("curated") for hit in hits]))
