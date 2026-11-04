"""/api/account/modules — my module variants CRUD (ownership + validation)."""
from __future__ import annotations
from backend.arch.eval.precision import AS_SHIPPED

import json
import re
import uuid

from .conftest import TEST_SUPABASE_URL

_MV_RE = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/user_module_variants") + r"(\?.*)?")
_OWNER = "user-uuid-1"  # the sub minted into user_token (service conftest)
_V1 = str(uuid.uuid4())


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


_ROW = {"id": _V1, "component": "mla_dsa_attention", "name": "big-kv",
        "dims": {"d_kv_lora": 640}, "base_model_id": "deepseek-ai/DeepSeek-V3.2",
        "updated_at": "2026-09-23T00:00:00Z"}


def test_requires_login_on_all_verbs(client):
    assert client.get("/api/account/modules").status_code == 401
    assert client.post("/api/account/modules", json={}).status_code == 401
    assert client.patch(f"/api/account/modules/{_V1}", json={}).status_code == 401
    assert client.delete(f"/api/account/modules/{_V1}").status_code == 401


def test_list_returns_own_rows_and_pins_owner(client, user_token, httpx_mock):
    httpx_mock.add_response(url=_MV_RE, method="GET", json=[_ROW])
    r = client.get("/api/account/modules", headers=_auth(user_token))
    assert r.status_code == 200
    assert [v["name"] for v in r.json()["variants"]] == ["big-kv"]
    req = httpx_mock.get_request(method="GET", url=_MV_RE)
    assert f"user_id=eq.{_OWNER}" in str(req.url)


def test_list_failure_is_502_not_a_fake_empty(client, user_token, httpx_mock):
    httpx_mock.add_response(url=_MV_RE, method="GET", status_code=500, json={})
    r = client.get("/api/account/modules", headers=_auth(user_token))
    assert r.status_code == 502


def test_create_rejects_unknown_component(client, user_token):
    r = client.post("/api/account/modules", headers=_auth(user_token),
                    json={"component": "not_a_component", "name": "x",
                          "dims": {"d_head": 128}})
    assert r.status_code == 422


def test_create_rejects_bad_dims(client, user_token):
    for dims in ({}, {"d_head": -1}, {"d_head": "big"},
                 {"d_model": 8192},  # shared-contract dim — never a variant dim
                 ):
        r = client.post("/api/account/modules", headers=_auth(user_token),
                        json={"component": "mla_dsa_attention", "name": "x", "dims": dims})
        assert r.status_code == 422, dims


def test_create_upserts_and_stamps_owner_from_token(client, user_token, httpx_mock):
    httpx_mock.add_response(url=_MV_RE, method="POST", json=[_ROW])
    r = client.post("/api/account/modules", headers=_auth(user_token),
                    json={"component": "mla_dsa_attention", "name": "big-kv",
                          "dims": {"d_kv_lora": 640},
                          "base_model_id": "deepseek-ai/DeepSeek-V3.2",
                          # a client-supplied owner must be ignored (pydantic
                          # drops unknown fields; the owner comes from the JWT)
                          "user_id": "someone-else"})
    assert r.status_code == 200, r.text
    assert r.json()["variant"]["dims"] == {"d_kv_lora": 640}
    req = httpx_mock.get_request(method="POST", url=_MV_RE)
    assert "on_conflict=user_id%2Ccomponent%2Cname" in str(req.url)
    assert json.loads(req.content)[0]["user_id"] == _OWNER


def test_update_and_delete_pin_owner(client, user_token, httpx_mock):
    httpx_mock.add_response(url=_MV_RE, method="PATCH", json=[_ROW])
    httpx_mock.add_response(url=_MV_RE, method="DELETE", json=[_ROW])
    assert client.patch(f"/api/account/modules/{_V1}", headers=_auth(user_token),
                        json={"name": "renamed"}).status_code == 200
    assert client.delete(f"/api/account/modules/{_V1}", headers=_auth(user_token)).status_code == 200
    for req in httpx_mock.get_requests():
        assert f"user_id=eq.{_OWNER}" in str(req.url)


def test_update_delete_404_on_missing_or_foreign(client, user_token, httpx_mock):
    httpx_mock.add_response(url=_MV_RE, method="PATCH", json=[])
    httpx_mock.add_response(url=_MV_RE, method="DELETE", json=[])
    assert client.patch(f"/api/account/modules/{_V1}", headers=_auth(user_token),
                        json={"name": "renamed"}).status_code == 404
    assert client.delete(f"/api/account/modules/{_V1}", headers=_auth(user_token)).status_code == 404


def test_update_rename_conflict_is_409(client, user_token, httpx_mock):
    httpx_mock.add_response(url=_MV_RE, method="PATCH", status_code=409, json={})
    r = client.patch(f"/api/account/modules/{_V1}", headers=_auth(user_token),
                     json={"name": "big-kv"})
    assert r.status_code == 409


def test_bad_uuid_is_422_not_502(client, user_token):
    assert client.patch("/api/account/modules/not-a-uuid", headers=_auth(user_token),
                        json={"name": "x"}).status_code == 422
    assert client.delete("/api/account/modules/not-a-uuid", headers=_auth(user_token)).status_code == 422


def test_entry_label_passthrough_in_module_bench(monkeypatch):
    """A non-model entry (a saved variant) keeps its `label` as the result and
    echo identity — two same-component entries no longer collide on the bare
    component name, and the variant's echo is EXACTLY its own dims (no extra
    dims from anywhere else)."""
    from tests.snapshots import snapshot_config
    from backend.arch.loader import SpecLoader
    from backend.playground import module_bench as mb
    loader = SpecLoader()
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    out = mb.module_bench(base_model_id="deepseek-ai/DeepSeek-V3.2", batch=1,
                          input_tokens=2048, precision=AS_SHIPPED, entries=[
        {"name": "mla_dsa_attention"},
        {"name": "mla_dsa_attention", "label": "mla_dsa_attention~my:big-kv",
         "dims": {"d_indexer_head": 128}, "seeded": True},
    ])
    assert out["errors"] == []
    keys = [r["entry"] for r in out["results"]]
    assert keys == ["mla_dsa_attention", "mla_dsa_attention~my:big-kv"]
    assert out["contract"]["per_component_dims"]["mla_dsa_attention~my:big-kv"] \
        == {"d_indexer_head": 128}


def test_entry_label_validation(monkeypatch):
    """Labels dedupe on the EFFECTIVE key (label or name@model_id), and a
    label containing '@' is rejected."""
    import pytest
    from pydantic import ValidationError
    from backend.api._models import ModuleBenchRequest
    with pytest.raises(ValidationError):
        ModuleBenchRequest(base_model_id="b", entries=[
            {"name": "a", "label": "a@m"},        # effective key a@m
            {"name": "a", "model_id": "m"},       # same effective key
        ])
    with pytest.raises(ValidationError):
        ModuleBenchRequest(base_model_id="b", entries=[
            {"name": "a", "label": "bad@label"},
        ])
