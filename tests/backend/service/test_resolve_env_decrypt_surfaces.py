"""resolve_env must distinguish "stored but undecryptable" from "none stored".

Returning None on a decrypt failure conflated a rotated/missing
HA_CREDS_ENC_KEY with "no credential" — a non-admin was told to "add a key"
(it is added, just unreadable) and the systemic cause was silent. resolve_env
now raises CredentialDecryptError; a genuinely absent credential still
returns None.
"""
import re

import pytest

from backend.service import agent_credentials as C

_URL = re.compile(r".*user_agent_credentials.*")


def _row():
    return {
        "agent_cli": "claude",
        "api_key_enc": "unrecoverable",
        "base_url": None,
        "model": None,
        "is_default": True,
    }


def test_decrypt_failure_raises(monkeypatch, httpx_mock):
    monkeypatch.setattr(C, "_supabase", lambda: ("https://test-project.supabase.co", "svc"))
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])

    def _boom(_ciphertext):
        raise ValueError("InvalidToken")

    monkeypatch.setattr(C.crypto, "decrypt", _boom)

    with pytest.raises(C.CredentialDecryptError):
        C.resolve_env("user-1")


def test_absent_credential_returns_none(monkeypatch, httpx_mock):
    monkeypatch.setattr(C, "_supabase", lambda: ("https://test-project.supabase.co", "svc"))
    httpx_mock.add_response(method="GET", url=_URL, json=[])

    assert C.resolve_env("user-1") is None


def test_successful_decrypt_returns_env(monkeypatch, httpx_mock):
    monkeypatch.setattr(C, "_supabase", lambda: ("https://test-project.supabase.co", "svc"))
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    monkeypatch.setattr(C.crypto, "decrypt", lambda _ct: "sk-real")

    out = C.resolve_env("user-1")
    assert out["agent_cli"] == "claude"
    assert "sk-real" in out["env"].values()
