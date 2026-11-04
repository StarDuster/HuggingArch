"""POST /analytics/event — anonymous-safe usage event ingestion.

Service-role key writes the row; the JWT (if present) sets user_id.
Outages on Supabase MUST NOT bubble up; the endpoint silent-fails."""
from __future__ import annotations

from .conftest import TEST_SUPABASE_URL


class TestAnalyticsIngest:
    def test_anonymous_writes_with_null_user_id(self, client, httpx_mock):
        """No JWT → backend POSTs to analyze_logs with user_id=None."""
        captured = {}

        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(201, json={"ok": True})

        httpx_mock.add_callback(
            _capture,
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/analyze_logs",
        )

        r = client.post("/analytics/event", json={
            "endpoint": "analyze",
            "model_id": "meta-llama/Llama-3",
            "duration_ms": 250,
            "metadata": {"gpu_sku": "H100"},
        })
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        assert captured["body"]["user_id"] is None
        assert captured["body"]["endpoint"] == "analyze"
        assert captured["body"]["metadata"] == {"gpu_sku": "H100"}

    def test_signed_in_writes_with_user_id(self, client, user_token, httpx_mock):
        captured = {}

        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(201, json={"ok": True})

        httpx_mock.add_callback(
            _capture,
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/analyze_logs",
        )

        r = client.post(
            "/analytics/event",
            json={"endpoint": "inference_estimate", "model_id": "x"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 200
        assert captured["body"]["user_id"] == "user-uuid-1"

    def test_silent_fail_on_supabase_500(self, client, httpx_mock):
        """A 500 from Supabase must not surface to the caller."""
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/rest/v1/analyze_logs",
            status_code=500,
            text="boom",
        )
        r = client.post("/analytics/event", json={
            "endpoint": "analyze", "model_id": "x",
        })
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_silent_fail_on_network_error(self, client, monkeypatch):
        """Network blow-ups (httpx.RequestError) also must not bubble up."""
        from backend.service import auth as _auth

        def _explode(*a, **kw):
            raise RuntimeError("network down")

        monkeypatch.setattr(_auth, "_service_role_client", _explode)
        r = client.post("/analytics/event", json={
            "endpoint": "analyze", "model_id": "x",
        })
        assert r.status_code == 200

    def test_no_supabase_env_skips_silently(self, client, monkeypatch):
        """Without SERVICE_ROLE_KEY, ingestion is a no-op (no exception)."""
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        r = client.post("/analytics/event", json={
            "endpoint": "analyze", "model_id": "x",
        })
        assert r.status_code == 200
