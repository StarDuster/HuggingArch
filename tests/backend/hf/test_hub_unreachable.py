"""A Hub that cannot be read degrades the way each Hub reader documents.

The hub's HTTP client is httpx2; a DNS / connect / proxy failure from it is
an `httpx2.HTTPError`, neither `httpx.HTTPError` nor `OSError`. Every reader
catches `backend.hf.metadata.HUB_UNREACHABLE` for it: the shard-header read
retries, the hub metadata fetch returns {}, and a snapshot refresh writes a
FAIL line instead of stopping. No network: the Hub client is monkeypatched.
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx2  # the hub's HTTP client: what its calls raise
import pytest

from backend.arch.quant_schemes import SCHEME_NONE
from backend.hf import metadata, snapshot


@pytest.fixture(autouse=True)
def _no_meta_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGARCH_HF_META_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("time.sleep", lambda _s: None)


class _Hub:
    """A one-shard repo whose header reads fail `header_failures` times."""

    def __init__(self, *, model_info=None, header_failures=0, listing=None):
        self._model_info = model_info
        self._header_failures = header_failures
        self._listing = listing
        self.header_calls = 0

    def __call__(self):  # stands in for the `HfApi` class
        return self

    def model_info(self, model_id, **_kw):
        if isinstance(self._model_info, BaseException):
            raise self._model_info
        shard = SimpleNamespace(rfilename="model.safetensors")
        return SimpleNamespace(siblings=[shard], sha="abc123")

    def parse_safetensors_file_metadata(self, repo, shard, **_kw):
        self.header_calls += 1
        if self.header_calls <= self._header_failures:
            raise httpx2.ConnectError("connection refused")
        t = SimpleNamespace(dtype="BF16", shape=[4, 4], data_offsets=[0, 32])
        return SimpleNamespace(tensors={"lm_head.weight": t})

    def list_repo_files(self, model_id):
        if isinstance(self._listing, BaseException):
            raise self._listing
        return self._listing or []


def test_a_shard_header_read_retries_through_a_connect_error(monkeypatch):
    hub = _Hub(header_failures=2)
    monkeypatch.setattr("huggingface_hub.HfApi", hub)
    header = metadata.read_header_hf("org/m")
    assert hub.header_calls == 3
    assert header.total_size == 32


def test_a_shard_header_that_never_reads_is_an_os_error(monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi", _Hub(header_failures=3))
    with pytest.raises(OSError, match="header unreadable"):
        metadata.read_header_hf("org/m")


def test_the_hub_metadata_fetch_is_empty_when_the_hub_is_unreachable(monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi",
                        _Hub(model_info=httpx2.ConnectError("dns failure")))
    assert metadata._fetch_metadata_hf(
        "org/m", scheme=SCHEME_NONE, num_bits=0) == {}


def test_a_processor_listing_the_hub_cannot_answer_is_a_fail_line(monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi",
                        _Hub(listing=httpx2.ConnectError("offline")))
    monkeypatch.setattr(snapshot, "fetch_config_raw", lambda m: {"model_type": "x"})
    monkeypatch.setattr(snapshot, "fetch_metadata_raw", lambda m, c: {"t": 1})
    monkeypatch.setattr(snapshot, "_fetch_forward_source_raw",
                        lambda m, c: {"files": [{"path": "m.py", "text": "x"}]})
    written = []
    monkeypatch.setattr("backend.arch.snapshots.write_snapshot",
                        lambda *a, **k: written.append(a))
    line = snapshot.snapshot_one("org/never-written")
    assert line.startswith("FAIL processor: org/never-written")
    assert "ConnectError" in line
    assert written == []


def test_a_config_read_the_hub_cannot_answer_is_a_fail_line(monkeypatch):
    def proxy_down(**_kw):
        raise httpx2.ProxyError("proxy down")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", proxy_down)
    line = snapshot.snapshot_one("org/never-written")
    assert line.startswith("FAIL config: org/never-written")
