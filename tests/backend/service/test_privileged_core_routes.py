"""Conditional admin gating for privileged routes owned by the core API."""
from __future__ import annotations

import pytest

from .conftest import mock_admins_no, mock_admins_yes


_PROTECTED_REQUESTS = [
    ("GET", "/list-op-gaps", None),
    ("GET", "/list-op-proposals", None),
    ("POST", "/propose-op", {"op_name": "probe", "gap": {}}),
    ("POST", "/stop-propose-op", {"op_name": "probe"}),
    ("POST", "/review-op", {"op_name": "probe"}),
    ("POST", "/stop-review-op", {"op_name": "probe"}),
    ("GET", "/debug", None),
    ("GET", "/debug/spec-locks", None),
    ("GET", "/spec-status?run_key=someone-else%7Cmodel", None),
    ("POST", "/stop-spec", {"run_key": "someone-else|model"}),
]


def _request(client, method: str, path: str, body, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.request(method, path, json=body, headers=headers)


@pytest.mark.parametrize(("method", "path", "body"), _PROTECTED_REQUESTS)
def test_service_mode_rejects_anonymous(client, method, path, body):
    response = _request(client, method, path, body)
    assert response.status_code == 401, response.text


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/list-op-gaps", None),
        ("GET", "/debug/spec-locks", None),
        ("GET", "/spec-status?model_id=probe", None),
    ],
)
def test_service_mode_rejects_non_admin(
    client, user_token, httpx_mock, method, path, body,
):
    mock_admins_no(httpx_mock, "user-uuid-1")
    response = _request(client, method, path, body, user_token)
    assert response.status_code == 403, response.text


def test_service_mode_allows_admin(client, admin_token, httpx_mock):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    response = _request(client, "GET", "/list-op-gaps", None, admin_token)
    assert response.status_code == 200, response.text


def test_local_mode_keeps_privileged_routes_open(client, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    response = _request(client, "GET", "/list-op-gaps", None)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    ("body", "key"),
    [
        ({"model_id": "org/m"}, "org/m"),
        ({"model_id": "org/m", "kind": "generate"}, "org/m"),
        ({"model_id": "org/m", "kind": "refine"}, "refine|org/m"),
    ],
)
def test_local_stop_spec_derives_the_key_from_the_kind(client, monkeypatch, body, key):
    from backend import spec_worker
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    stopped = []
    monkeypatch.setattr(spec_worker, "stop", lambda k: stopped.append(k) or False)
    response = _request(client, "POST", "/stop-spec", body)
    assert response.status_code == 200, response.text
    assert stopped == [key]


def test_local_stop_spec_refuses_an_unknown_kind(client, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    response = _request(client, "POST", "/stop-spec", {"model_id": "org/m", "kind": "other"})
    assert response.status_code == 422, response.text
