"""backend.service.sweep_cache.SupabaseBackend — durable cold tier.

All Supabase calls intercepted (pytest-httpx); the backend silent-fails on
every error so a miss / outage / unconfigured env degrades to the tiers in
front of it and never raises into the endpoint.
"""
from __future__ import annotations

import re

import pytest


URL = "https://test-sweep-cache.supabase.co"
GET_RE    = re.compile(re.escape(URL) + r"/rest/v1/rpc/sweep_cache_get")
UPSERT_RE = re.compile(re.escape(URL) + r"/rest/v1/rpc/sweep_cache_upsert")
CLEAR_RE  = re.compile(re.escape(URL) + r"/rest/v1/rpc/sweep_cache_clear")


@pytest.fixture
def supabase_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", URL)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")


@pytest.fixture
def no_supabase(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)


class TestSilentFailUnconfigured:
    def test_get_none(self, no_supabase):
        from backend.service.sweep_cache import SupabaseBackend
        # No httpx_mock — a real call would explode.
        assert SupabaseBackend().get("k") is None

    def test_put_no_op(self, no_supabase):
        from backend.service.sweep_cache import SupabaseBackend
        SupabaseBackend().put("k", {"v": 1})  # must not raise

    def test_clear_no_op(self, no_supabase):
        from backend.service.sweep_cache import SupabaseBackend
        SupabaseBackend().clear()  # must not raise


class TestRoundTrip:
    def test_get_hit_returns_payload(self, supabase_env, httpx_mock):
        from backend.service.sweep_cache import SupabaseBackend
        httpx_mock.add_response(
            method="POST", url=GET_RE,
            json=[{"payload": {"results": [1, 2, 3], "code_version": "abc"}}],
        )
        assert SupabaseBackend().get("k") == {"results": [1, 2, 3], "code_version": "abc"}

    def test_get_miss_returns_none(self, supabase_env, httpx_mock):
        from backend.service.sweep_cache import SupabaseBackend
        httpx_mock.add_response(method="POST", url=GET_RE, json=[])
        assert SupabaseBackend().get("k") is None

    def test_put_posts_upsert(self, supabase_env, httpx_mock):
        from backend.service.sweep_cache import SupabaseBackend
        httpx_mock.add_response(method="POST", url=UPSERT_RE, status_code=204, text="")
        SupabaseBackend().put("k", {"v": 1})
        req = httpx_mock.get_requests()[-1]
        assert req is not None  # a request was actually made

    def test_put_sends_the_encoded_arguments(self, supabase_env, httpx_mock):
        import json
        from backend.service.sweep_cache import SupabaseBackend
        httpx_mock.add_response(method="POST", url=UPSERT_RE, status_code=204, text="")
        SupabaseBackend().put("k", {"v": [1, "é"]})
        req = httpx_mock.get_requests()[-1]
        assert req.headers["content-type"] == "application/json"
        assert json.loads(req.content) == {"p_key": "k", "p_payload": {"v": [1, "é"]}}

    def test_an_oversize_put_stays_local(self, supabase_env, httpx_mock):
        """The store takes one jsonb value per statement: a gallery frontier
        walk (over 1 GB) only ever timed out. Over the cap, nothing is sent."""
        from backend.service.sweep_cache import SupabaseBackend
        SupabaseBackend(max_payload_bytes=1024).put("k", {"v": "x" * 2048})
        assert httpx_mock.get_requests() == []

    def test_get_500_swallowed(self, supabase_env, httpx_mock):
        from backend.service.sweep_cache import SupabaseBackend
        httpx_mock.add_response(method="POST", url=GET_RE, status_code=500, text="down")
        assert SupabaseBackend().get("k") is None  # no raise

    def test_put_500_swallowed(self, supabase_env, httpx_mock):
        from backend.service.sweep_cache import SupabaseBackend
        httpx_mock.add_response(method="POST", url=UPSERT_RE, status_code=500, text="down")
        SupabaseBackend().put("k", {"v": 1})  # no raise
