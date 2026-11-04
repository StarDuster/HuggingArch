"""Review and fix prep validate the record against the staged snapshot.

The agent's own validate reads config + metadata from the overlay's
`snapshots/`. Prep validated before that file existed, so its validate fell
back to a live config fetch — which, for a model_type the installed
transformers does not know (Qwen3-VL-MoE), resolved no vision dims and put a
load failure the spec did not have into the reviewer's prompt.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

SNAPSHOT = {"config": {"model_type": "m", "hidden_size": 8},
            "metadata": {"header": {"revision": None, "total_size": 0,
                                    "tensors": []},
                         "quant_scheme": "universal", "num_bits": 4},
            # Prep refuses a model whose source it could not read, so a
            # captured snapshot always carries it.
            "forward_source": {"source": "hf_modeling_files",
                               "paths": ["modeling_m.py"],
                               "files": [{"path": "modeling_m.py",
                                          "text": "class M: ...\n"}]}}


@pytest.mark.parametrize("module, prep, seed, body, who", [
    ("reviewer", "prepare_review", "seed_review_overlay",
     "build_review_task_body", {"reviewer_id": "u"}),
    ("fixer", "prepare_fix", "seed_fix_overlay",
     "build_fix_task_body", {"fixer_id": "u"}),
])
def test_prep_validates_after_the_snapshot_is_staged(
        monkeypatch, tmp_path, module, prep, seed, body, who):
    mod = importlib.import_module(f"backend.spec_worker.{module}")
    monkeypatch.setattr(mod, seed, lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(mod, "_build_prompt_context",
                        lambda model_id: ("sys", "gt", {"_snapshot_data": SNAPSHOT}))
    monkeypatch.setattr(mod, "_stage_source", lambda *a: [])
    monkeypatch.setattr(mod, body, lambda *a, **k: "")
    seen = []
    monkeypatch.setattr(
        mod, "_invoke_validate",
        lambda *a, **k: seen.append(sorted(
            p.name for p in (Path(tmp_path) / "snapshots").glob("*.json"))) or {})

    class Bundle:
        spec_yaml = ""
        def record_files(self):
            return []

    getattr(mod, prep)("org/m", record_id="r", bundle=Bundle(), **who)
    assert seen == [["org__m.json"]]
