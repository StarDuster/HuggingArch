"""Edit + set-default for BYOK agent credentials.

Two capabilities added on top of the create/replace `upsert`:
  • blank-key PUT edits an EXISTING row (endpoint / model / default) while
    keeping the stored key — no re-paste, no encryption key needed;
  • POST …/{agent}/default flips the default with no key at all.

Both are PATCH-only against the service-role REST endpoint, so they're mocked
with httpx_mock. FIFO response ordering matches the code's call order
(clear-other-defaults, then the targeted row).
"""
import re

_CREDS_RE = re.compile(
    r"https://test-project\.supabase\.co/rest/v1/user_agent_credentials"
)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


class TestSetDefault:
    def test_anonymous_401(self, client):
        r = client.post("/api/account/agent-credentials/claude/default")
        assert r.status_code == 401

    def test_unknown_agent_400_no_db_call(self, client, user_token):
        # Rejected before any REST call (no httpx_mock ⇒ a call would error).
        r = client.post(
            "/api/account/agent-credentials/bogus/default", headers=_auth(user_token)
        )
        assert r.status_code == 400

    def test_flips_default_and_clears_others(self, client, user_token, httpx_mock):
        httpx_mock.add_response(method="PATCH", url=_CREDS_RE, status_code=204)  # clear others
        httpx_mock.add_response(method="PATCH", url=_CREDS_RE, json=[{"agent_cli": "claude"}])
        r = client.post(
            "/api/account/agent-credentials/claude/default", headers=_auth(user_token)
        )
        assert r.status_code == 200
        assert r.json()["is_default"] is True

    def test_404_when_no_such_credential(self, client, user_token, httpx_mock):
        httpx_mock.add_response(method="PATCH", url=_CREDS_RE, status_code=204)  # clear others
        httpx_mock.add_response(method="PATCH", url=_CREDS_RE, json=[])          # nothing updated
        r = client.post(
            "/api/account/agent-credentials/kimi/default", headers=_auth(user_token)
        )
        assert r.status_code == 404


class TestBlankKeyEdit:
    def test_blank_key_edits_existing_row(self, client, user_token, httpx_mock):
        # is_default omitted (False) ⇒ a single targeted PATCH, no clear step.
        httpx_mock.add_response(
            method="PATCH", url=_CREDS_RE, json=[{"agent_cli": "claude"}]
        )
        r = client.put(
            "/api/account/agent-credentials",
            json={"agent_cli": "claude", "api_key": "", "model": "claude-opus-4-8"},
            headers=_auth(user_token),
        )
        assert r.status_code == 200
        assert r.json()["saved"] is True

    def test_blank_key_no_existing_row_400(self, client, user_token, httpx_mock):
        httpx_mock.add_response(method="PATCH", url=_CREDS_RE, json=[])  # nothing to edit
        r = client.put(
            "/api/account/agent-credentials",
            json={"agent_cli": "codex", "api_key": "  "},
            headers=_auth(user_token),
        )
        assert r.status_code == 400
