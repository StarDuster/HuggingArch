"""Named credentials (migration 0000): several stored configs per
(user, agent_cli) — e.g. two Claude Code endpoints side by side.

`name` ('' = the unnamed config) widens the identity from (user_id,
agent_cli) to (user_id, agent_cli, name): CRUD targets one config, task
routing may point at 'agent_cli:name', and resolve_env falls back to the
default / unnamed config when a run names only the agent. The REST
endpoint is mocked with httpx_mock; no database, no agent.
"""
from __future__ import annotations

import json
import re

import pytest

import backend.service.routes.account as _account  # noqa: F401  (route import)
from backend.service import agent_credentials as C

_URL = re.compile(r".*user_agent_credentials.*")
_DURL = re.compile(r".*agent_task_defaults.*")
_PURL = re.compile(r".*profiles.*")
_MISSING_NAME_WRITE = {"code": "PGRST204", "message":
                       "Could not find the 'name' column of "
                       "'user_agent_credentials' in the schema cache"}


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def _row(**over):
    row = {"agent_cli": "claude", "api_key_enc": "ct", "base_url": None,
           "model": "claude-default", "is_default": True, "name": ""}
    row.update(over)
    return row


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(C, "_supabase", lambda: ("https://test-project.supabase.co", "svc"))
    monkeypatch.setattr(C.crypto, "encrypt", lambda k: f"enc:{k}")
    monkeypatch.setattr(C.crypto, "decrypt", lambda ct: {"ct": "sk-real"}.get(ct, ct))
    monkeypatch.setattr(C.crypto, "enabled", lambda: True)


# ---------------------------------------------------------------------------
# Writes target one named config
# ---------------------------------------------------------------------------

def test_upsert_conflicts_on_the_named_key(store, httpx_mock):
    httpx_mock.add_response(method="POST", url=_URL)
    C.upsert(user_id="u", agent_cli="claude", name="work", api_key="sk-w",
             base_url=None, is_default=False)
    req = httpx_mock.get_requests()[0]
    assert req.url.params["on_conflict"] == "user_id,agent_cli,name"
    assert json.loads(req.content)["name"] == "work"


def test_update_fields_filters_by_name(store, httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_URL, json=[{"id": 1}])
    assert C.update_fields(user_id="u", agent_cli="claude", name="work",
                           base_url=None, model="m", is_default=False) is True
    assert httpx_mock.get_requests()[0].url.params["name"] == "eq.work"


def test_update_fields_of_a_missing_config_is_false(store, httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_URL, json=[])
    assert C.update_fields(user_id="u", agent_cli="claude", name="ghost",
                           base_url=None, model="m", is_default=False) is False


def test_set_default_filters_by_name_and_clears_every_other_row(store, httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_URL)  # clear others
    httpx_mock.add_response(method="PATCH", url=_URL, json=[{"id": 1}])
    assert C.set_default(user_id="u", agent_cli="claude", name="work") is True
    clear, setone = httpx_mock.get_requests()
    assert clear.url.params["or"] == "(agent_cli.neq.claude,name.neq.work)"
    assert setone.url.params["name"] == "eq.work"


def test_delete_filters_by_name(store, httpx_mock):
    httpx_mock.add_response(method="DELETE", url=_URL)
    assert C.delete("u", "claude", name="work") is True
    assert httpx_mock.get_requests()[0].url.params["name"] == "eq.work"


def test_saved_key_and_base_filters_by_name(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL,
                            json=[{"api_key_enc": "sk-work", "base_url": None}])
    assert C.saved_key_and_base("u", "claude", name="work") == ("sk-work", None)
    assert httpx_mock.get_requests()[0].url.params["name"] == "eq.work"


def test_name_column_undeployed_blank_name_retries_without_it(store, httpx_mock):
    httpx_mock.add_response(method="POST", url=_URL, status_code=400,
                            json=_MISSING_NAME_WRITE)
    httpx_mock.add_response(method="POST", url=_URL)
    C.upsert(user_id="u", agent_cli="claude", api_key="sk-x",
             base_url=None, is_default=False)
    first, second = httpx_mock.get_requests()
    assert "name" in json.loads(first.content)
    assert "name" not in json.loads(second.content)


def test_name_column_undeployed_real_name_raises(store, httpx_mock):
    httpx_mock.add_response(method="POST", url=_URL, status_code=400,
                            json=_MISSING_NAME_WRITE)
    with pytest.raises(C.CredentialNamesUnavailable):
        C.upsert(user_id="u", agent_cli="claude", name="work", api_key="sk-x",
                 base_url=None, is_default=False)


def test_name_rejects_colon(store):
    with pytest.raises(ValueError):
        C.normalize_name("a:b")


# ---------------------------------------------------------------------------
# Listing exposes the name
# ---------------------------------------------------------------------------

def test_listing_carries_name_and_the_flag(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(), _row(name="work", is_default=False)])
    view = C.listing("u")
    assert view["names_enabled"] is True
    assert [c["name"] for c in view["items"]] == ["", "work"]


def test_listing_tolerates_the_missing_name_column(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, status_code=400, json={
        "code": "42703",
        "message": "column user_agent_credentials.name does not exist"})
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    view = C.listing("u")
    assert view["names_enabled"] is False
    assert view["items"][0]["name"] == ""
    first, second = httpx_mock.get_requests()
    assert "name" in first.url.params["select"]
    assert "name" in first.url.params.get("order", "")
    assert "name" not in second.url.params["select"]
    assert "name" not in second.url.params.get("order", "")


# ---------------------------------------------------------------------------
# resolve_env picks among configs
# ---------------------------------------------------------------------------

def test_explicit_name_picks_that_config(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(api_key_enc="sk-main"), _row(name="work", api_key_enc="sk-work",
                                          is_default=False, model="m-work")])
    out = C.resolve_env("u", "claude", name="work")
    assert out["env"]["ANTHROPIC_API_KEY"] == "sk-work"
    assert out["env"]["ANTHROPIC_MODEL"] == "m-work"


def test_agent_without_name_prefers_the_default_row(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(api_key_enc="sk-main"), _row(name="work", api_key_enc="sk-work",
                                          is_default=False)])
    assert C.resolve_env("u", "claude")["env"]["ANTHROPIC_API_KEY"] == "sk-main"


def test_agent_without_name_and_no_default_asks_the_user_to_pick(store, httpx_mock):
    """Several named configs, none default, no unnamed one: the user has keys
    but none is chosen — an error asking them to pick, never None (which an
    admin's run would read as "use the host's credentials")."""
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(name="work", api_key_enc="sk-work", is_default=False),
        _row(name="personal", api_key_enc="sk-personal", is_default=False),
    ])
    with pytest.raises(C.CredentialChoiceRequired):
        C.resolve_env("u", "claude")


def test_several_credentials_none_default_nothing_routed_asks_to_pick(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(name="work", api_key_enc="sk-work", is_default=False),
        _row(agent_cli="codex", name="team", api_key_enc="sk-team", is_default=False),
    ])
    with pytest.raises(C.CredentialChoiceRequired) as err:
        C.resolve_env("u")
    assert err.value.refusal("generate specs")["status"] == "credential_choice_required"


def test_an_admin_with_several_undefaulted_credentials_is_refused_not_given_host_keys(
        client, admin_token, store, httpx_mock, monkeypatch):
    """The admin route used to read "no credential" and spend the host's
    keys; now it asks the admin to pick one of their own."""
    from backend.service import spec_generations as _gens
    from .conftest import mock_admins_yes
    import backend.spec_worker as _sw
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    monkeypatch.setattr(_gens, "get", lambda rid: {"id": rid, "model_id": "m", "user_id": "u"})
    monkeypatch.setattr("backend.service.routes.admin.superseded_by", lambda rec: None)
    monkeypatch.setattr(_sw, "acquire", lambda k: pytest.fail("the review started"))
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(name="work", is_default=False), _row(name="home", is_default=False)])
    httpx_mock.add_response(method="GET", url=_PURL, json=[{"agent_task_routing": {}}])
    r = client.post("/api/admin/spec-generations/rec-1/review", headers=_auth(admin_token))
    assert r.json()["status"] == "credential_choice_required"


def test_unknown_named_config_resolves_to_nothing(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    assert C.resolve_env("u", "claude", name="ghost") is None


def test_task_routing_to_a_named_config(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(api_key_enc="sk-main"), _row(name="work", api_key_enc="sk-work",
                                          is_default=False)])
    httpx_mock.add_response(method="GET", url=_PURL, json=[
        {"agent_task_routing": {"review": "claude:work"}}])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    assert C.resolve_env("u", task="review")["env"]["ANTHROPIC_API_KEY"] == "sk-work"


def test_task_routing_plain_agent_picks_among_its_configs(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(api_key_enc="sk-main"), _row(name="work", api_key_enc="sk-work",
                                          is_default=False)])
    httpx_mock.add_response(method="GET", url=_PURL, json=[
        {"agent_task_routing": {"review": "claude"}}])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    assert C.resolve_env("u", task="review")["env"]["ANTHROPIC_API_KEY"] == "sk-main"


# ---------------------------------------------------------------------------
# put_task_routing understands 'agent:name' targets
# ---------------------------------------------------------------------------

def test_fetch_task_routing_keeps_named_targets(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_PURL, json=[
        {"agent_task_routing": {"review": "claude:work", "fix": "kimi",
                                "generate": "nobody", "refine": "claude:"}}])
    assert C.fetch_task_routing("u") == {"review": "claude:work", "fix": "kimi"}


def test_put_task_routing_stores_the_named_target(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_PURL, json=[{"agent_task_routing": None}])
    httpx_mock.add_response(method="GET", url=_URL, json=[_row(name="work")])
    httpx_mock.add_response(method="PATCH", url=_PURL)
    C.put_task_routing("u", "review", "claude:work")
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body["agent_task_routing"] == {"review": "claude:work"}


def test_put_task_routing_rejects_a_config_the_user_lacks(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_PURL, json=[{"agent_task_routing": None}])
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])  # unnamed only
    with pytest.raises(ValueError, match="claude:work"):
        C.put_task_routing("u", "review", "claude:work")


def test_put_task_routing_rejects_a_malformed_target(store):
    with pytest.raises(ValueError):
        C.put_task_routing("u", "review", "claude:a:b")


# ---------------------------------------------------------------------------
# Routes pass the name through
# ---------------------------------------------------------------------------

def test_put_credential_route_roundtrip_with_name(client, user_token, store, httpx_mock):
    httpx_mock.add_response(method="POST", url=_URL)
    r = client.put("/api/account/agent-credentials",
                   json={"agent_cli": "claude", "api_key": "sk-w",
                         "name": "work", "is_default": False},
                   headers=_auth(user_token))
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "work"
    req = httpx_mock.get_requests()[-1]
    assert json.loads(req.content)["name"] == "work"


def test_delete_route_passes_the_name(client, user_token, httpx_mock):
    httpx_mock.add_response(method="DELETE", url=_URL)
    r = client.delete("/api/account/agent-credentials/claude?name=work",
                      headers=_auth(user_token))
    assert r.status_code == 200
    assert httpx_mock.get_requests()[-1].url.params["name"] == "eq.work"


def test_test_connection_route_picks_the_named_config(client, user_token, store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        {"api_key_enc": "sk-work", "base_url": None}])
    httpx_mock.add_response(method="GET", url=re.compile(r".*anthropic.*"),
                            json={"data": [{"id": "claude-x"}]})
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(name="work", model="claude-x", is_default=False)])
    httpx_mock.add_response(method="GET", url=re.compile(r".*agent_task_defaults.*"),
                            json=[])
    httpx_mock.add_response(method="POST", url=re.compile(r".*anthropic.*"),
                            json={"id": "msg"})
    r = client.post("/api/account/agent-credentials/claude/test?name=work",
                    headers=_auth(user_token))
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    first = httpx_mock.get_requests()[0]
    assert first.url.params["name"] == "eq.work"


@pytest.mark.parametrize("answer", [{"status_code": 503, "text": "down"},
                                    {"status_code": 401, "text": "JWT expired"}])
def test_credentials_that_cannot_be_read_are_unavailable_not_absent(store, httpx_mock, answer):
    """An outage is not "none stored", which would let an admin's run take
    the host's keys."""
    httpx_mock.add_response(method="GET", url=_URL, **answer)
    with pytest.raises(C.CredentialsUnavailable) as err:
        C.resolve_env("u")
    assert err.value.refusal("run a review")["status"] == "credentials_unavailable"


def test_an_admin_whose_credentials_cannot_be_read_is_refused_not_given_host_keys(
        client, admin_token, store, httpx_mock, monkeypatch):
    from backend.service import spec_generations as _gens
    from .conftest import mock_admins_yes
    import backend.spec_worker as _sw
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    monkeypatch.setattr(_gens, "get", lambda rid: {"id": rid, "model_id": "m", "user_id": "u"})
    monkeypatch.setattr("backend.service.routes.admin.superseded_by", lambda rec: None)
    monkeypatch.setattr(_sw, "acquire", lambda k: pytest.fail("the review started"))
    httpx_mock.add_response(method="GET", url=_URL, status_code=503, text="down")
    r = client.post("/api/admin/spec-generations/rec-1/review", headers=_auth(admin_token))
    assert r.json()["status"] == "credentials_unavailable"
