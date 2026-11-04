"""The backends a signed-in run may fail over to are resolved as its own
credentials were: the user's stored key per CLI, the host's only for an
admin, and never a key that cannot be chosen or read."""
from __future__ import annotations

from backend.service import agent_credentials as C


def _resolve(table):
    def fake(user_id, agent_cli=None, *, name=None, task=None):
        value = table.get(agent_cli)
        if isinstance(value, Exception):
            raise value
        return value
    return fake


def test_the_users_own_keys_then_the_hosts_for_an_admin(monkeypatch):
    monkeypatch.setattr(C, "resolve_env", _resolve({
        "codex": {"agent_cli": "codex", "env": {"OPENAI_API_KEY": "sk-user"}},
        "claude": None,
    }))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host")
    assert C.fallback_envs("u", "kimi", task="generate", host_allowed=True) == {
        "codex": {"OPENAI_API_KEY": "sk-user"}, "claude": None}
    assert C.fallback_envs("u", "kimi", task="generate", host_allowed=False) == {
        "codex": {"OPENAI_API_KEY": "sk-user"}}


def test_a_host_gateway_token_is_a_claude_credential(monkeypatch):
    """A host with ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL and no API key
    has claude credentials; with the URL alone it has none."""
    monkeypatch.setattr(C, "resolve_env", _resolve({}))
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY",
                 "KIMI_MODEL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert C.fallback_envs("u", "kimi", task="generate", host_allowed=True) == {}
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "gateway-token")
    assert C.fallback_envs("u", "kimi", task="generate", host_allowed=True) == {
        "claude": None}


def test_a_stored_claude_secret_is_still_the_api_key(monkeypatch):
    """BYOK keeps one secret per row and sends it as the API key — a host
    auth token does not change what a stored credential becomes."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "gateway-token")
    assert C._env_for("claude", "sk-user", "https://u.example", "m") == {
        "ANTHROPIC_API_KEY": "sk-user",
        "ANTHROPIC_BASE_URL": "https://u.example",
        "ANTHROPIC_MODEL": "m"}


def test_an_unusable_stored_key_is_left_out_not_replaced(monkeypatch):
    monkeypatch.setattr(C, "resolve_env", _resolve({
        "codex": C.CredentialDecryptError("rotated"),
        "claude": C.CredentialChoiceRequired("two keys, none chosen"),
    }))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-host")
    assert C.fallback_envs("u", "kimi", task="generate", host_allowed=True) == {}
