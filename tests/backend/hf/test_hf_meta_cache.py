"""HF metadata cache honors HUGGINGARCH_HF_META_CACHE_DIR."""
from __future__ import annotations

import json
from pathlib import Path


def test_store_and_load_use_the_env_cache_dir(monkeypatch, tmp_path):
    from backend.hf import _cache as hf_cache

    monkeypatch.setenv("HUGGINGARCH_HF_META_CACHE_DIR", str(tmp_path))
    payload = {
        "header": {"revision": None, "total_size": 2,
                   "tensors": [{"name": "x", "dtype": "BF16", "shape": [1]}]},
    }
    hf_cache.cache_store("org/model", payload)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    loaded = hf_cache.cache_load_raw("org/model")
    assert loaded is not None
    assert loaded["header"] == payload["header"]
    on_disk = json.loads(Path(files[0]).read_text(encoding="utf-8"))
    assert on_disk["header"] == payload["header"]
