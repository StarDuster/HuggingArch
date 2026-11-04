"""backend.service.rate_limit — sliding-window key-based limiter.

Probe via the actual FastAPI middleware: TestClient hits a non-rate-
limited GET to confirm pass-through, then a rate-limited POST to confirm
429 + Retry-After once over budget. Time is monkeypatched to stay
deterministic (no `time.sleep` in tests).
"""
from __future__ import annotations

from .conftest import make_user_jwt


# ---------------------------------------------------------------------------
# Time + bucket isolation helpers
# ---------------------------------------------------------------------------

def _patch_clock(monkeypatch):
    """Return a callable `tick(seconds)` that advances the limiter's clock."""
    from backend.service import rate_limit
    state = {"now": 0.0}

    def fake_monotonic():
        return state["now"]

    monkeypatch.setattr(rate_limit.time, "monotonic", fake_monotonic)

    def tick(seconds: float):
        state["now"] += seconds
    return tick


def _reset_buckets(monkeypatch):
    """Wipe the global window state between tests."""
    from backend.service import rate_limit
    monkeypatch.setattr(rate_limit, "_buckets", {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_first_n_calls_pass_then_429(self, client, monkeypatch, httpx_mock):
        """At ANALYZE_RPM=3, the 4th POST in the same window returns 429."""
        from backend.service import rate_limit
        monkeypatch.setitem(rate_limit._LIMITS, "/analyze", 3)
        monkeypatch.setitem(rate_limit._LIMITS, "/inference/estimate", 0)
        _reset_buckets(monkeypatch)
        _patch_clock(monkeypatch)

        codes = [
            client.post("/analyze", json={"model_id": "x"}).status_code
            for _ in range(5)
        ]
        # /analyze POSTs miss spec lookup → 4xx, but rate limiter
        # runs BEFORE the route handler, so we still see the limiter's
        # 429s for #4 and #5. Crucially, #1-#3 pass through to the
        # handler (returning whatever the handler decides).
        assert codes[3] == 429
        assert codes[4] == 429
        assert codes[0] != 429
        assert codes[1] != 429
        assert codes[2] != 429

    def test_window_slides(self, client, monkeypatch):
        """Calls older than 60s should drop out of the window."""
        from backend.service import rate_limit
        monkeypatch.setitem(rate_limit._LIMITS, "/analyze", 2)
        _reset_buckets(monkeypatch)
        tick = _patch_clock(monkeypatch)

        # Burn the bucket
        c1 = client.post("/analyze", json={"model_id": "x"}).status_code
        c2 = client.post("/analyze", json={"model_id": "x"}).status_code
        assert c1 != 429 and c2 != 429
        c3 = client.post("/analyze", json={"model_id": "x"}).status_code
        assert c3 == 429

        # Slide past the 60s window — old entries should expire
        tick(61)
        c4 = client.post("/analyze", json={"model_id": "x"}).status_code
        assert c4 != 429, f"expected pass after window slid, got {c4}"

    def test_429_response_carries_retry_after(self, client, monkeypatch):
        from backend.service import rate_limit
        monkeypatch.setitem(rate_limit._LIMITS, "/analyze", 1)
        _reset_buckets(monkeypatch)
        _patch_clock(monkeypatch)

        client.post("/analyze", json={"model_id": "x"})  # consume the budget
        r = client.post("/analyze", json={"model_id": "x"})
        assert r.status_code == 429
        assert int(r.headers["Retry-After"]) > 0
        assert r.headers["X-RateLimit-Limit"] == "1"


class TestKeyResolution:
    def test_user_keyed_when_signed_in(self, client, monkeypatch, httpx_mock):
        """Two anonymous IPs share a bucket only when both are anon. Once
        a JWT shows up, the limiter keys on `user:<sub>` and they get
        their own budget."""
        from backend.service import rate_limit
        monkeypatch.setitem(rate_limit._LIMITS, "/analyze", 1)
        _reset_buckets(monkeypatch)
        _patch_clock(monkeypatch)

        # Anonymous request — uses default IP key
        client.post("/analyze", json={"model_id": "x"})
        r = client.post("/analyze", json={"model_id": "x"})
        assert r.status_code == 429

        # A signed-in user is keyed on their `sub`, not the IP
        token = make_user_jwt(sub="alice-uuid")
        r = client.post(
            "/analyze",
            json={"model_id": "x"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code != 429, f"signed-in user inherited anon's exhausted bucket (got {r.status_code})"

    def test_edge_ip_key_separates_clients(self, client, monkeypatch):
        from backend.service import rate_limit
        monkeypatch.setitem(rate_limit._LIMITS, "/analyze", 1)
        _reset_buckets(monkeypatch)
        _patch_clock(monkeypatch)

        client.post("/analyze", json={"model_id": "x"},
                    headers={"Fly-Client-IP": "1.1.1.1"})
        r = client.post("/analyze", json={"model_id": "x"},
                        headers={"Fly-Client-IP": "1.1.1.1"})
        assert r.status_code == 429
        # Different edge IP — different bucket → still allowed
        r = client.post("/analyze", json={"model_id": "x"},
                        headers={"Fly-Client-IP": "2.2.2.2"})
        assert r.status_code != 429


class TestNonRateLimitedPaths:
    def test_get_passes(self, client, monkeypatch):
        """GET /inference/gpus is read-only, never rate limited."""
        from backend.service import rate_limit
        monkeypatch.setitem(rate_limit._LIMITS, "/analyze", 1)
        _reset_buckets(monkeypatch)
        _patch_clock(monkeypatch)
        for _ in range(10):
            assert client.get("/inference/gpus").status_code == 200

    def test_unrelated_post_passes(self, client, monkeypatch):
        """POST /analytics/event isn't on the rate-limit list."""
        from backend.service import rate_limit
        monkeypatch.setitem(rate_limit._LIMITS, "/analyze", 1)
        _reset_buckets(monkeypatch)
        _patch_clock(monkeypatch)
        for _ in range(5):
            r = client.post("/analytics/event",
                            json={"endpoint": "analyze", "model_id": "x"})
            assert r.status_code == 200, r.text


class TestCoverage:
    def test_every_threaded_route_is_limited(self):
        """`_LIMITS` must equal the set of routes, under `api/routes` AND
        `service/routes`, whose handler hands work off the event loop
        (`offload` / `offload_response`, or `asyncio.to_thread`).
        A new heavy endpoint that forgets the table fails here instead of
        shipping unlimited — `/api/spec/analyze` did exactly that."""
        import ast
        from pathlib import Path
        from backend.service import rate_limit

        backend = Path(rate_limit.__file__).resolve().parents[1]
        threaded = set()
        for routes_dir in (backend / "api" / "routes", backend / "service" / "routes"):
            for path in sorted(routes_dir.glob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for fn in ast.walk(tree):
                    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    route = None
                    for dec in fn.decorator_list:
                        if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                                and dec.args and isinstance(dec.args[0], ast.Constant)):
                            route = dec.args[0].value
                    if route is None:
                        continue
                    uses_thread = any(
                        (isinstance(n, ast.Attribute) and n.attr == "to_thread")
                        or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id in ("offload", "offload_response"))
                        for n in ast.walk(fn))
                    if uses_thread:
                        threaded.add(route)
        assert threaded, "route scan found nothing — scanner broken"
        # /compare, the quick-pick list and the gallery frontier hand their
        # jobs off under their own deadline (`within_deadline`), and the two
        # sweeps one call down (`sweep_runtime`, shared with the warm job),
        # out of this scan's sight. The two auth session endpoints are
        # limited as credential-stuffing relays, not for their CPU cost.
        expected = threaded | {
            "/compare",
            "/inspector/examples",
            "/gallery/frontier",
            "/gallery/models",
            "/inference/estimate-sweep",
            "/inference/estimate-parallel-sweep",
            "/inference/parallel-sweep/frontier",
            "/inference/estimate-sweep/frontier",
            "/api/auth/login",
            "/api/auth/exchange",
        }
        assert expected == set(rate_limit._LIMITS), (
            f"unlimited threaded routes: {sorted(expected - set(rate_limit._LIMITS))}; "
            f"limited but not threaded: {sorted(set(rate_limit._LIMITS) - expected)}")

    def test_fly_client_ip_wins_over_forwarded_for(self, client, monkeypatch):
        """`X-Forwarded-For` is client-supplied and must not pick the bucket."""
        from backend.service import rate_limit
        monkeypatch.setitem(rate_limit._LIMITS, "/analyze", 1)
        _reset_buckets(monkeypatch)
        _patch_clock(monkeypatch)
        client.post("/analyze", json={"model_id": "x"},
                    headers={"Fly-Client-IP": "9.9.9.9", "X-Forwarded-For": "1.1.1.1"})
        r = client.post("/analyze", json={"model_id": "x"},
                        headers={"Fly-Client-IP": "9.9.9.9", "X-Forwarded-For": "2.2.2.2"})
        assert r.status_code == 429, "spoofed X-Forwarded-For escaped the bucket"

    def test_fly_client_ip_ignored_when_proxy_headers_untrusted(self, client, monkeypatch):
        """Off Fly (no opt-in), Fly-Client-IP is client-supplied and must
        NOT key the bucket — otherwise a caller rotates it to dodge 429s."""
        from backend.service import rate_limit
        monkeypatch.delenv("FLY_APP_NAME")
        monkeypatch.setitem(rate_limit._LIMITS, "/analyze", 1)
        _reset_buckets(monkeypatch)
        _patch_clock(monkeypatch)
        client.post("/analyze", json={"model_id": "x"},
                    headers={"Fly-Client-IP": "1.1.1.1"})
        r = client.post("/analyze", json={"model_id": "x"},
                        headers={"Fly-Client-IP": "2.2.2.2"})
        assert r.status_code == 429, "untrusted Fly-Client-IP rotated the bucket"
