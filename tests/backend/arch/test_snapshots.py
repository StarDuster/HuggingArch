"""Snapshot lookup: shared dir first, then the run overlay's snapshots/.

The overlay leg ($HA_SPEC_OVERLAY_DIR) is what lets a sandboxed spec-gen
agent's validate loop run offline — the worker stages the prefetched
snapshot there before launching the agent.
"""
import json
import sys

import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.arch import snapshots


def _write(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version": 3, "model_id": "x", "marker": "%s"}' % marker)


def test_overlay_leg_is_consulted(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_OVERLAY_DIR", str(tmp_path))
    _write(tmp_path / "snapshots" / "acme__NoRepoSnapshot.json", "overlay")

    snap = snapshots.load_snapshot("acme/NoRepoSnapshot")

    assert snap is not None and snap["marker"] == "overlay"


def test_shared_dir_wins_over_overlay(tmp_path, monkeypatch):
    # Qwen/Qwen3-8B ships a repo snapshot; the overlay must not shadow it.
    monkeypatch.setenv("HA_SPEC_OVERLAY_DIR", str(tmp_path))
    _write(tmp_path / "snapshots" / "Qwen__Qwen3-8B.json", "overlay")

    snap = snapshots.load_snapshot("Qwen/Qwen3-8B")

    assert snap is not None and snap.get("marker") != "overlay"


def test_no_overlay_env_means_no_overlay_leg(monkeypatch):
    monkeypatch.delenv("HA_SPEC_OVERLAY_DIR", raising=False)
    assert snapshots.load_snapshot("acme/NoRepoSnapshot") is None


def test_release_dates_for_uses_snapshot_then_skips_hub_when_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGARCH_HF_RELEASE_DATE_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    out = snapshots.release_dates_for(["Qwen/Qwen3-30B-A3B", "no-such/model", "Qwen/Qwen3-30B-A3B"])
    assert out["Qwen/Qwen3-30B-A3B"] == "2025-04-27"
    assert out["no-such/model"] is None
    assert list(out) == ["Qwen/Qwen3-30B-A3B", "no-such/model"]


def test_a_hub_miss_is_asked_once_per_ttl(monkeypatch, tmp_path):
    """The account page lists every generation; a repo with no date must not
    cost a Hub round-trip on every load."""
    monkeypatch.setenv("HUGGINGARCH_HF_RELEASE_DATE_DIR", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    asked = []
    monkeypatch.setattr(snapshots, "_fetch_hf_release_date",
                        lambda mid: asked.append(mid) or None)
    for _ in range(2):
        assert snapshots.release_dates_for(["no-such/model"]) == {"no-such/model": None}
    assert asked == ["no-such/model"]


def test_a_v1_created_at_cache_is_a_miss(monkeypatch, tmp_path):
    """ModelInfo.createdAt can be months before a private repo becomes public;
    those old cache entries must not masquerade as initial-commit dates."""
    monkeypatch.setenv("HUGGINGARCH_HF_RELEASE_DATE_DIR", str(tmp_path))
    snapshots._release_date_cache_path("moonshotai/Kimi-K3").write_text(json.dumps({
        "release_date": "2026-06-13",
        "checked_at": 0,
    }))
    assert snapshots._read_release_date_cache("moonshotai/Kimi-K3") == (False, None)


def test_release_date_uses_the_last_commit_on_the_last_page(monkeypatch):
    """The Hub lists newest-first and may paginate; the release date is the
    oldest visible (initial public) commit, not the newest one."""
    monkeypatch.setattr(
        snapshots,
        "_iter_hf_commit_pages",
        lambda mid: [
            [{"date": "2026-09-02T02:22:41.000Z"}],
            [
                {"date": "2026-07-27T15:14:52.000Z"},
                {"date": "2026-07-27T13:31:26.000Z"},
            ],
        ],
    )
    assert snapshots._fetch_hf_release_date("moonshotai/Kimi-K3") == "2026-07-27"


@pytest.mark.parametrize("payload", [
    {"source": snapshots._RELEASE_DATE_CACHE_SOURCE,
     "release_date": None, "checked_at": "yesterday"},
    {"source": snapshots._RELEASE_DATE_CACHE_SOURCE,
     "release_date": 20250427, "checked_at": 0},
])
def test_a_malformed_release_date_cache_is_a_miss(payload, monkeypatch, tmp_path, caplog):
    """A wrongly shaped cache file warns and is asked again; it never
    reaches the account API as an exception."""
    monkeypatch.setenv("HUGGINGARCH_HF_RELEASE_DATE_DIR", str(tmp_path))
    snapshots._release_date_cache_path("no-such/model").write_text(json.dumps(payload))
    with caplog.at_level("WARNING"):
        assert snapshots._read_release_date_cache("no-such/model") == (False, None)
    assert "malformed release-date cache" in caplog.text


def test_every_shipped_snapshot_freezes_its_forward_source():
    """A snapshot is the offline record a spec is judged against, and the
    forward source is the only witness for what the checkpoint bytes cannot
    say (cache formats, recurrent-state dtypes, block topology). Three
    snapshots once shipped without one — written while the installed
    transformers lacked their model_type — and every source-reading check
    passed them by having nothing to read. Refresh with
    `python -m backend.hf.snapshot <model_id>`."""
    sourceless = [model_id for model_id in snapshots.iter_snapshot_ids()
                  if not snapshots.has_forward_source(
                      snapshots.load_snapshot(model_id).get("forward_source"))]
    assert sourceless == []


def test_a_snapshot_without_forward_source_is_refused(tmp_path):
    """`write_snapshot` is where every snapshot is assembled — the CLI
    refresh, a spec-gen run's capture, an overlay's staging — so it is
    where a sourceless one stops."""
    for fs in (None, {}, {"source": "x", "files": [{"path": "m.py", "text": ""}]}):
        with pytest.raises(ValueError, match="no forward source"):
            snapshots.write_snapshot("acme/NoSource", {}, {}, forward_source=fs,
                                     release_date="2026-01-01",
                                     overlay_dir=str(tmp_path))
    assert not (tmp_path / "snapshots" / "acme__NoSource.json").exists()
