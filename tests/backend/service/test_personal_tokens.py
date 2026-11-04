"""Personal access tokens (PAT) — helpers, auth dependency, account routes."""
from __future__ import annotations


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_pat_shape_and_hash():
    from backend.service import personal_tokens as P
    assert P.looks_like_pat("hba_abc") is True
    assert P.looks_like_pat("eyJhbGc...") is False   # a JWT is not a PAT
    assert P.looks_like_pat("") is False
    # hash is deterministic + one-way (never the plaintext)
    h = P._hash("hba_secret")
    assert h == P._hash("hba_secret") and h != "hba_secret" and len(h) == 64


# ---------------------------------------------------------------------------
# require_user_or_pat dependency
# ---------------------------------------------------------------------------
def test_require_user_or_pat_accepts_valid_pat(monkeypatch):
    from backend.service import auth as A
    monkeypatch.setattr("backend.service.personal_tokens.verify", lambda t: "user-9")
    u = A.require_user_or_pat(authorization="Bearer hba_good")
    assert u == {"sub": "user-9", "auth": "pat"}


def test_require_user_or_pat_rejects_bad_pat(monkeypatch):
    from fastapi import HTTPException
    from backend.service import auth as A
    monkeypatch.setattr("backend.service.personal_tokens.verify", lambda t: None)
    try:
        A.require_user_or_pat(authorization="Bearer hba_revoked")
        raise AssertionError("should have raised")
    except HTTPException as e:
        assert e.status_code == 401


def test_require_user_or_pat_falls_back_to_jwt(client, user_token):
    """A non-PAT bearer still goes through the JWT path (verified via a real
    route that depends on require_user_or_pat)."""
    # /api/bench/upload uses require_user_or_pat; a valid JWT must pass auth
    # (it then fails later on an empty/invalid CSV, NOT on 401).
    r = client.post("/api/bench/upload", json={"csv_text": "", "label": "x"},
                    headers=_auth(user_token))
    assert r.status_code != 401


def test_bench_upload_anonymous_401(client, bench_proxied):
    r = client.post("/api/bench/upload", json={"csv_text": "", "label": "x"})
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Account token routes
# ---------------------------------------------------------------------------
def test_issue_list_revoke_routes(client, user_token, monkeypatch):
    from backend.service import personal_tokens as P
    issued = {"id": "tok-1", "token": "hba_PLAINTEXT_ONCE", "token_prefix": "hba_PLAINT", "name": "box"}
    monkeypatch.setattr(P, "issue", lambda *, user_id, name="": issued)
    monkeypatch.setattr(P, "list_for_user",
                        lambda uid: [{"id": "tok-1", "name": "box", "token_prefix": "hba_PLAINT",
                                      "created_at": "2026-08-01", "last_used_at": None, "revoked_at": None}])
    revoked = []
    monkeypatch.setattr(P, "revoke", lambda *, token_id, user_id: revoked.append(token_id) or True)

    r = client.post("/api/account/tokens", json={"name": "box"}, headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["token"] == "hba_PLAINTEXT_ONCE"

    r = client.get("/api/account/tokens", headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["tokens"][0]["token_prefix"] == "hba_PLAINT"
    assert "token_hash" not in r.json()["tokens"][0]   # hash never exposed

    r = client.delete("/api/account/tokens/tok-1", headers=_auth(user_token))
    assert r.status_code == 200 and revoked == ["tok-1"]


def test_token_routes_anonymous_401(client):
    assert client.get("/api/account/tokens").status_code == 401
    assert client.post("/api/account/tokens", json={"name": "x"}).status_code == 401
