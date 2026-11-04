"""The Inspector "Try:" quick-pick is derived from the corpus, not hardcoded.

`featured_examples` returns the latest N models above a params floor, newest
first by release date, computed from the shipped snapshots — so the list stays
current as bigger/newer models land instead of drifting from a hand-edited
constant.
"""

from __future__ import annotations


from backend.api.gallery_overview import (
    _attention_families,
    _model_size_and_date,
    featured_examples,
)


def test_featured_examples_are_latest_large_models():
    ids = featured_examples(min_params=100_000_000_000, limit=4)
    assert 1 <= len(ids) <= 4
    # every pick is >= the floor and carries a release date
    dated = []
    for mid in ids:
        params, release = _model_size_and_date(mid)
        assert params is not None and params >= 100_000_000_000, mid
        assert release, mid
        dated.append(release)
    # newest first
    assert dated == sorted(dated, reverse=True)


def test_the_floor_reads_the_main_models_params():
    """The params floor is the main model's — the headline `/analyze` and
    the gallery state: an encoder's weights and an MTP head's are their own
    accounting. For a model with a vision tower it is below the checkpoint's
    tensor total, which carries the tower."""
    from backend.analyzer import analyze_model
    from backend.hf.metadata import fetch_metadata
    from tests.snapshots import resolve_snapshot
    model = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    resolve_snapshot(model)
    params, _release = _model_size_and_date(model)
    summary = analyze_model(model_id=model)["summary"]
    assert params == summary["main_total_params"]
    assert params < summary["total_params"]
    assert params < fetch_metadata(model)["total_params"]


def test_attention_families_are_the_backbones_mixers():
    """Gallery badges read the backbone's mixers (`Program.attention_mix`):
    both of a hybrid's, by their display labels, and not an aux tower's."""
    from tests.snapshots import resolve_snapshot
    assert _attention_families(resolve_snapshot("moonshotai__Kimi-K3")) == ["KDA", "MLA"]
    assert _attention_families(resolve_snapshot("stepfun-ai__Step-3.7-Flash")) == ["GQA", "SWA"]
    assert _attention_families(None) == []


def test_featured_examples_respects_limit_and_floor():
    two = featured_examples(min_params=100_000_000_000, limit=2)
    assert len(two) <= 2
    # a floor no snapshot meets yields an empty list, never a hardcoded default
    assert featured_examples(min_params=10**15, limit=4) == []


def test_featured_examples_memoized():
    a = featured_examples(limit=4)
    b = featured_examples(limit=4)
    assert a == b


def test_inspector_examples_endpoint():
    from fastapi.testclient import TestClient
    import backend.api  # noqa: F401  # registers routes
    from backend.api._app import app

    resp = TestClient(app).get("/inspector/examples")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("examples"), list)
    assert body["examples"] == featured_examples(min_params=100_000_000_000, limit=4)
