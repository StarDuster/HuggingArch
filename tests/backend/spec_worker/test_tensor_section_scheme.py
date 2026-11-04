"""Spec generation reads a new model's checkpoint under its config's scheme.

The tensor section of the prompt is also what the post-success snapshot
records. It used to come from the serving `fetch_metadata`, which reads a
model with no snapshot under `SCHEME_NONE`; the snapshot then said
`quant_scheme: none`, and a new FP8 checkpoint (GLM-5.3) counted its
`weight_scale_inv` sidecars as weights. The validator compared that record's
model storage against the spec's and failed by exactly the scale bytes, no
matter what the spec said.
"""
from __future__ import annotations

from backend.hf import metadata as hf_metadata
from backend.hf.header import SafetensorsHeader
from backend.spec_worker import _metadata

FP8_CONFIG = {"quantization_config": {
    "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}}


def test_a_new_fp8_checkpoint_counts_its_scales_as_metadata(monkeypatch):
    header = SafetensorsHeader.from_tensors({
        "model.layers.0.mlp.down_proj.weight": ("F8_E4M3", [128, 128]),
        "model.layers.0.mlp.down_proj.weight_scale_inv": ("F32", [1, 1]),
    }, total_size=128 * 128 + 4)
    monkeypatch.setattr(hf_metadata, "cache_load_raw", lambda m: None)
    monkeypatch.setattr(hf_metadata, "cache_store", lambda m, d: None)
    monkeypatch.setattr(hf_metadata, "read_header_hf", lambda m: header)

    captured: dict = {}
    _metadata.get_tensor_section("org/new-fp8", FP8_CONFIG, meta_out=captured)
    meta = captured["metadata"]

    assert meta["quant_scheme"] == "fp8"
    assert meta["quantization_metadata_bytes"] == 4
    assert meta["model_storage_bytes"] == 128 * 128


def _section(monkeypatch, tensors, config):
    header = SafetensorsHeader.from_tensors(tensors, total_size=0)
    monkeypatch.setattr(hf_metadata, "cache_load_raw", lambda m: None)
    monkeypatch.setattr(hf_metadata, "cache_store", lambda m, d: None)
    monkeypatch.setattr(hf_metadata, "read_header_hf", lambda m: header)
    return _metadata.get_tensor_section("org/model", config)


def test_the_section_lists_the_resolved_schemes_sidecars_apart(monkeypatch):
    """Which tensors to leave undeclared comes from the scheme the config
    resolves to, not a fixed FP8 suffix: a DeepSeek-V4-style config
    (`scale_fmt: ue8m0`) also stores `fp8_ue8m0`'s `.scale` sidecars."""
    config = {"quantization_config": {**FP8_CONFIG["quantization_config"],
                                      "scale_fmt": "ue8m0"}}
    text = _section(monkeypatch, {
        "model.layers.0.mlp.down_proj.weight": ("F8_E4M3", [128, 128]),
        "model.layers.0.mlp.down_proj.weight_scale_inv": ("F32", [1, 1]),
        "model.layers.0.mlp.experts.0.w1.scale": ("F8_E8M0", [1, 1]),
    }, config)
    weights, sidecars = text.split("Quantization metadata of the checkpoint's scheme")
    assert "down_proj.weight:" in weights
    assert "weight_scale_inv" not in weights and ".scale:" not in weights
    assert "`fp8+fp8_ue8m0`" in sidecars
    assert "`*.weight_scale_inv`" in sidecars and "`*.scale`" in sidecars
    assert "down_proj.weight_scale_inv:" in sidecars
    assert "w1.scale:" in sidecars


def test_an_unquantized_checkpoint_has_no_sidecar_list(monkeypatch):
    text = _section(monkeypatch, {
        "model.layers.0.mlp.down_proj.weight": ("BF16", [128, 128]),
    }, {})
    assert "Quantization metadata" not in text
    assert "scale_inv" not in text


def test_an_unregistered_quant_method_is_named(monkeypatch):
    text = _section(monkeypatch, {
        "model.layers.0.mlp.down_proj.weight": ("I32", [16, 128]),
        "model.layers.0.mlp.down_proj.weight_scale": ("F16", [1, 128]),
    }, {"quantization_config": {"quant_method": "brand_new"}})
    assert "`quant_method: brand_new` names no registered quant scheme" in text
    assert "Quantization metadata of" not in text
