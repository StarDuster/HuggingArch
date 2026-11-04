"""A stored record is analysed against the snapshot it was judged against.

A record carries its spec and the snapshot its generation captured. Validate
read that snapshot and passed; the Inspector staged only the spec, read the
checkpoint from the Hub under no quant scheme, and showed a new FP8 model
(GLM-5.3) off by exactly its `weight_scale_inv` bytes. Every reader now takes
the record's overlay: the record's snapshot is staged beside its spec, and the
config and checkpoint record are read from it.

The record here is DeepSeek-V3's FP8 checkpoint under a model id with no
shared spec or snapshot, so nothing but the record can answer.
"""
from __future__ import annotations

import json

import pytest

from backend.arch import snapshots
from backend.arch.loader import SpecLoader, overlay_kwargs
from backend.hf import metadata as hf_metadata
from backend.spec_worker import overlay as _ov
from backend.spec_worker.bundle import FormalSource, normalize_bundle

SOURCE = "deepseek-ai/DeepSeek-V3"
RECORD = "acme/DeepSeek-V3-record"
SPEC = f"base: deepseek_v3\nmatch:\n  model_id: [{RECORD}]\n"


@pytest.fixture
def record_root(tmp_path, monkeypatch):
    monkeypatch.delenv("HA_SPEC_OVERLAY_DIR", raising=False)
    monkeypatch.setattr(_ov, "user_specs_root", lambda: tmp_path / "user_specs")

    def hub(*a, **k):
        raise AssertionError("a stored record is read from its snapshot, "
                             "never the Hub")
    monkeypatch.setattr(hf_metadata, "read_header_hf", hub)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hub)

    snapshot = {**snapshots.load_snapshot(SOURCE), "model_id": RECORD}
    bundle = normalize_bundle(RECORD, SPEC, [], FormalSource.from_deployment())
    return _ov.materialize_user_spec("u1", RECORD, bundle, snapshot)


def test_the_record_snapshot_is_staged_where_readers_look(record_root):
    assert snapshots.load_snapshot(RECORD) is None
    staged = json.loads(snapshots.snapshot_path(RECORD, record_root).read_text())
    assert staged["metadata"]["quant_scheme"] == "fp8"
    assert hf_metadata.fetch_config(RECORD, record_root) == \
        hf_metadata.fetch_config(SOURCE)
    assert hf_metadata.fetch_metadata(RECORD, record_root)["quant_scheme"] == "fp8"


def test_the_loader_names_the_overlay_it_serves_a_record_from(record_root):
    loader = SpecLoader(**overlay_kwargs(record_root))
    assert loader.overlay_for(RECORD) == record_root
    assert loader.overlay_for(SOURCE) is None


def test_the_page_reconciles_a_record_as_validate_does(record_root, monkeypatch):
    import backend.analyzer as analyzer
    monkeypatch.setattr(analyzer, "_fetch_model_str", lambda *a, **k: "")

    summary = analyzer.analyze_model(
        model_id=RECORD, overlay_dir=record_root)["summary"]

    assert summary["hf_quantization_metadata_bytes"] > 0
    assert summary["storage_diff"] == 0
    assert summary["metadata_storage_diff"] == 0
