"""A stored credential that can't be decrypted must surface, not look absent.

`saved_key_and_base` used to `return None` on any decrypt failure — identical
to "no credential stored". A systemic cause (rotated / missing
`HA_CREDS_ENC_KEY`) would then make every user's saved key silently look
absent, and the models route would tell the user to "save this credential
first" when it is already saved. Now it raises `CredentialDecryptError`, which
the route turns into a truthful 409.
"""
import re

from backend.service import agent_credentials as _creds

_CREDS_RE = re.compile(
    r"https://test-project\.supabase\.co/rest/v1/user_agent_credentials"
)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_decrypt_failure_raises_credential_decrypt_error(monkeypatch, client, user_token, httpx_mock):
    # A credential row exists...
    httpx_mock.add_response(
        method="GET",
        url=_CREDS_RE,
        json=[{"api_key_enc": "unrecoverable", "base_url": None}],
    )
    # ...but decryption fails (as a rotated master key would cause).
    def _boom(_ciphertext):
        raise ValueError("InvalidToken")

    monkeypatch.setattr(_creds.crypto, "decrypt", _boom)

    # Blank api_key ⇒ the route falls back to the stored credential.
    r = client.post(
        "/api/account/agent-credentials/models",
        headers=_auth(user_token),
        json={"agent_cli": "claude", "api_key": "", "base_url": ""},
    )
    assert r.status_code == 409
    assert "decrypt" in r.json()["detail"].lower()


def test_absent_credential_still_says_save_first(client, user_token, httpx_mock):
    # No row ⇒ genuinely absent ⇒ unchanged 400 "save this credential first".
    httpx_mock.add_response(method="GET", url=_CREDS_RE, json=[])
    r = client.post(
        "/api/account/agent-credentials/models",
        headers=_auth(user_token),
        json={"agent_cli": "claude", "api_key": "", "base_url": ""},
    )
    assert r.status_code == 400
    assert "save this credential first" in r.json()["detail"].lower()
    # The page asks for a key on this reason, never on the wording.
    assert r.json()["reason"] == "key_required"
