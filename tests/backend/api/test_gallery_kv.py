"""The gallery's KV leaderboard: each model's per-request cache under every
preset that sets only the cache, from the same facts walk."""
from __future__ import annotations

import backend.api.gallery_overview as g


def test_the_leaderboard_offers_the_cache_only_presets():
    assert list(g.KV_PRECISIONS) == ["as_shipped", "kv_fp8", "kv_nvfp4"]
    assert [p["id"] for p in g._kv_precisions()] == list(g.KV_PRECISIONS)


def test_each_precision_prices_the_cache_in_its_own_format():
    config, resolved, _ = g._resolve_one("meta-llama/Llama-3.2-1B")
    kv = g._model_facts_sync("meta-llama/Llama-3.2-1B", config, resolved)[
        "kv_per_request"]
    assert set(kv) == set(g.KV_PRECISIONS)
    for seq in map(str, g.KV_SEQ_LENS):
        shipped = kv["as_shipped"][seq]["result_bytes"]
        # BF16 → FP8 halves it; NVFP4 is half a byte plus an E4M3 scale
        # per 16 elements: 9/32 of BF16.
        assert kv["kv_fp8"][seq]["result_bytes"] == shipped / 2
        assert kv["kv_nvfp4"][seq]["result_bytes"] == shipped * 9 / 32


def test_each_precision_names_its_cache():
    config, resolved, _ = g._resolve_one("meta-llama/Llama-3.2-1B")
    facts = g._model_facts_sync("meta-llama/Llama-3.2-1B", config, resolved)
    ids = facts["kv_cache_identity"]
    assert set(ids) == set(g.KV_PRECISIONS)
    # Another cache format is another cache.
    assert len(set(ids.values())) == len(ids)


def test_each_precision_names_its_cache_dtypes():
    config, resolved, _ = g._resolve_one("meta-llama/Llama-3.2-1B")
    dtypes = g._model_facts_sync(
        "meta-llama/Llama-3.2-1B", config, resolved)["kv_dtypes"]
    assert dtypes["as_shipped"] == ["BF16"]
    assert set(dtypes) == set(g.KV_PRECISIONS)
