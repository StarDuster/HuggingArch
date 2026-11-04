"""Gap detection for the spec-gen → op-writer chain."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.spec_worker._op_chain import find_fresh_gap


def test_find_fresh_gap_picks_this_model(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.spec_worker._op_chain.GAP_DIR", str(tmp_path))
    started = time.time()
    (tmp_path / "other__1.gap.json").write_text(
        json.dumps({"op_name": "aaa", "context_model_id": "org/Other"}),
        encoding="utf-8",
    )
    (tmp_path / "me__1.gap.json").write_text(
        json.dumps({"op_name": "new_scan", "context_model_id": "org/Me"}),
        encoding="utf-8",
    )
    found = find_fresh_gap("org/Me", started)
    assert found is not None
    assert found["op_name"] == "new_scan"


def test_find_fresh_gap_ignores_stale(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.spec_worker._op_chain.GAP_DIR", str(tmp_path))
    stale = tmp_path / "me__old.gap.json"
    stale.write_text(
        json.dumps({"op_name": "old", "context_model_id": "org/Me"}),
        encoding="utf-8",
    )
    stale_mtime = time.time() - 120
    import os
    os.utime(stale, (stale_mtime, stale_mtime))
    assert find_fresh_gap("org/Me", time.time()) is None
