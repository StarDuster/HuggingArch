"""An admin answer is cached for a bounded time and dropped when the user's
rows go: a revoked or deleted admin does not keep admin access until the
process restarts."""
from __future__ import annotations

import contextlib

from backend.service import auth as _auth


class _Resp:
    status_code = 200

    def __init__(self, rows):
        self._rows = rows

    def json(self):
        return self._rows


def _admins(monkeypatch, rows):
    calls = []

    @contextlib.contextmanager
    def client():
        class C:
            def get(self, path, params=None):
                calls.append(params)
                return _Resp(list(rows))
        yield C()

    monkeypatch.setattr(_auth, "_service_role_client", client)
    return calls


def test_the_answer_expires(monkeypatch):
    rows = [{"user_id": "u"}]
    calls = _admins(monkeypatch, rows)
    now = [1000.0]
    monkeypatch.setattr(_auth.time, "monotonic", lambda: now[0])
    assert _auth.is_admin("u") is True
    rows.clear()                                    # revoked in the database
    assert _auth.is_admin("u") is True              # still inside the TTL
    now[0] += _auth._ADMIN_TTL_SECONDS + 1
    assert _auth.is_admin("u") is False
    assert len(calls) == 2


def test_forget_drops_the_cached_answer(monkeypatch):
    rows = [{"user_id": "u"}]
    _admins(monkeypatch, rows)
    assert _auth.is_admin("u") is True
    rows.clear()
    _auth.forget_admin("u")
    assert _auth.is_admin("u") is False


def test_a_lookup_in_flight_across_forget_does_not_write_back(monkeypatch):
    import contextlib
    calls = []

    @contextlib.contextmanager
    def client():
        class C:
            def get(self, path, params=None):
                _auth.forget_admin("u")              # deleted mid-lookup
                calls.append(1)
                return _Resp([{"user_id": "u"}])
        yield C()

    monkeypatch.setattr(_auth, "_service_role_client", client)
    assert _auth.is_admin("u") is True               # the answer it read
    assert "u" not in _auth._ADMIN_CACHE             # but not cached
