"""The checkpoint's own byte total gates `validate`.

`index_total_size` is the bytes the shard headers' `data_offsets` span — the
one figure no dtype table reconstructs. A spec whose stored bytes (weights,
scales, buffers) disagree with it is not valid, even where the two
dtype-table totals agree with each other."""
from __future__ import annotations

from dataclasses import replace

MODEL = "Qwen/Qwen3-8B"


def _validate_with(monkeypatch, total_size):
    from backend.arch.validate import _orchestrator as O
    from backend.arch.validate._matching import _get_safetensors_data

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    data = _get_safetensors_data(MODEL)
    data = {**data, "header": replace(data["header"], total_size=total_size),
            "index_total_size": total_size}
    monkeypatch.setattr(O, "_get_safetensors_data", lambda model_id, overlay_dir=None: data)
    return O.validate(MODEL), data


def test_the_snapshot_reproduces_its_file(monkeypatch):
    from backend.arch.validate._matching import _get_safetensors_data
    total = _get_safetensors_data(MODEL)["index_total_size"]
    report, _ = _validate_with(monkeypatch, total)
    assert report["valid"] is True
    assert report["checkpoint_storage_bytes"]["measured_diff"] == 0


def test_a_file_total_the_spec_misses_is_an_error(monkeypatch):
    from backend.arch.validate._matching import _get_safetensors_data
    total = _get_safetensors_data(MODEL)["index_total_size"] + 4096
    report, _ = _validate_with(monkeypatch, total)
    assert report["valid"] is False
    assert any(e.startswith("Checkpoint bytes:") and "diff=+4,096" in e
               for e in report["errors"])
