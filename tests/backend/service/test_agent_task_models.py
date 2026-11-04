"""Per-task model choice on a BYOK agent credential.

One key + endpoint per (user, agent_cli); `task_models` (migration 0036) may
pin a different model for each agent run (generate / refine / review / fix).
A task without an entry runs on the credential's own `model`. Before the
column is deployed every read treats it as "no overrides". The REST endpoint
is mocked with httpx_mock; no database, no agent.
"""
from __future__ import annotations

import json
import re

import pytest

import backend.service.routes.account as _account
import backend.service.routes.admin as _admin
from backend.service import agent_credentials as C
from backend.service import spec_generation_reviews as _reviews
from backend.service import spec_generations as _gens
from .conftest import mock_admins_yes

_URL = re.compile(r".*user_agent_credentials.*")
_DURL = re.compile(r".*agent_task_defaults.*")
_PURL = re.compile(r".*profiles.*")
_MISSING_SELECT = {"code": "42703", "message":
                   "column user_agent_credentials.task_models does not exist"}
_MISSING_WRITE = {"code": "PGRST204", "message":
                  "Could not find the 'task_models' column of "
                  "'user_agent_credentials' in the schema cache"}


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def _row(**over):
    row = {"agent_cli": "claude", "api_key_enc": "ct", "base_url": None,
           "model": "claude-default", "is_default": True}
    row.update(over)
    return row


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(C, "_supabase", lambda: ("https://test-project.supabase.co", "svc"))
    monkeypatch.setattr(C.crypto, "decrypt", lambda _ct: "sk-real")


# ---------------------------------------------------------------------------
# resolve_env
# ---------------------------------------------------------------------------

def test_task_model_overrides_default(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(task_models={"review": "claude-reviewer"})])
    out = C.resolve_env("u", task="review")
    assert out["env"]["ANTHROPIC_MODEL"] == "claude-reviewer"
    assert "task_models" in httpx_mock.get_requests()[0].url.params["select"]


def test_task_without_entry_falls_back_to_credential_model(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(task_models={"review": "claude-reviewer"})])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    assert C.resolve_env("u", task="fix")["env"]["ANTHROPIC_MODEL"] == "claude-default"


def test_null_task_models_is_the_old_behaviour(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[_row(task_models=None, model=None)])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    env = C.resolve_env("u", task="generate")["env"]
    assert env == {"ANTHROPIC_API_KEY": "sk-real"}


def test_unknown_task_is_a_programming_error(store):
    with pytest.raises(ValueError):
        C.resolve_env("u", task="summarize")


def test_column_absent_reads_without_it(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, status_code=400, json=_MISSING_SELECT)
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    out = C.resolve_env("u", task="review")
    assert out["env"]["ANTHROPIC_MODEL"] == "claude-default"
    first, second = httpx_mock.get_requests()[:2]
    assert "task_models" in first.url.params["select"]
    assert "task_models" not in second.url.params["select"]


# ---------------------------------------------------------------------------
# Site-wide task defaults (migration 0039): pin → site default → credential
# model → endpoint default
# ---------------------------------------------------------------------------

def test_site_default_beats_credential_model(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_DURL, json=[
        {"agent_cli": "claude", "task": "review", "model": "claude-cheap"}])
    assert C.resolve_env("u", task="review")["env"]["ANTHROPIC_MODEL"] == "claude-cheap"


def test_pin_beats_site_default(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(task_models={"review": "claude-reviewer"})])
    out = C.resolve_env("u", task="review")
    assert out["env"]["ANTHROPIC_MODEL"] == "claude-reviewer"
    # The pin wins without even reading the defaults table.
    assert len(httpx_mock.get_requests()) == 1


def test_site_default_of_another_agent_does_not_apply(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_DURL, json=[
        {"agent_cli": "codex", "task": "review", "model": "gpt-x"}])
    assert C.resolve_env("u", task="review")["env"]["ANTHROPIC_MODEL"] == "claude-default"


def test_missing_defaults_table_falls_back(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_DURL, status_code=404)
    assert C.resolve_env("u", task="review")["env"]["ANTHROPIC_MODEL"] == "claude-default"


def test_fetch_task_defaults_shape_and_filter(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_DURL, json=[
        {"agent_cli": "claude", "task": "fix", "model": " m-f "},
        {"agent_cli": "claude", "task": "bogus", "model": "x"},
        {"agent_cli": "nobody", "task": "fix", "model": "x"},
        {"agent_cli": "kimi", "task": "fix", "model": ""}])
    assert C.fetch_task_defaults() == {"claude": {"fix": "m-f"}}


def test_put_task_default_validates(store):
    with pytest.raises(ValueError):
        C.put_task_default(agent_cli="nobody", task="fix", model="m")
    with pytest.raises(ValueError):
        C.put_task_default(agent_cli="claude", task="summarize", model="m")


def test_put_task_default_upserts_and_clears(store, httpx_mock):
    httpx_mock.add_response(method="POST", url=_DURL)
    C.put_task_default(agent_cli="claude", task="fix", model=" m-f ")
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body == {"agent_cli": "claude", "task": "fix", "model": "m-f",
                    "updated_at": body["updated_at"]}
    httpx_mock.add_response(method="DELETE", url=_DURL)
    C.put_task_default(agent_cli="claude", task="fix", model="")
    assert httpx_mock.get_requests()[1].method == "DELETE"


# ---------------------------------------------------------------------------
# listing / writes before and after the column exists
# ---------------------------------------------------------------------------

def test_listing_reports_column_and_items(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(task_models={"fix": "m-fix", "gone": "x"})])
    view = C.listing("u")
    assert view["task_models_enabled"] is True
    assert view["items"][0]["task_models"] == {"fix": "m-fix"}
    assert "api_key_enc" not in view["items"][0]


def test_listing_without_column(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, status_code=400, json=_MISSING_SELECT)
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    view = C.listing("u")
    assert view["task_models_enabled"] is False
    assert view["items"][0]["task_models"] == {}


def test_update_omits_task_models_when_not_given(store, httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_URL, json=[{"agent_cli": "claude"}])
    C.update_fields(user_id="u", agent_cli="claude", base_url=None,
                    model="m", is_default=False)
    assert "task_models" not in json.loads(httpx_mock.get_requests()[0].content)


def test_update_writes_normalized_task_models(store, httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_URL, json=[{"agent_cli": "claude"}])
    C.update_fields(user_id="u", agent_cli="claude", base_url=None, model="m",
                    is_default=False, task_models={"review": " m-r ", "fix": ""})
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["task_models"] == {"review": "m-r"}


def test_update_setting_overrides_without_column_raises(store, httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_URL, status_code=400, json=_MISSING_WRITE)
    with pytest.raises(C.TaskModelsUnavailable):
        C.update_fields(user_id="u", agent_cli="claude", base_url=None, model="m",
                        is_default=False, task_models={"review": "m-r"})


def test_update_clearing_without_column_goes_through(store, httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_URL, status_code=400, json=_MISSING_WRITE)
    httpx_mock.add_response(method="PATCH", url=_URL, json=[{"agent_cli": "claude"}])
    assert C.update_fields(user_id="u", agent_cli="claude", base_url=None, model="m",
                           is_default=False, task_models={}) is True
    assert "task_models" not in json.loads(httpx_mock.get_requests()[1].content)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_put_rejects_unknown_task_key(client, user_token):
    # Rejected before any REST call (no httpx_mock response ⇒ a call would error).
    r = client.put("/api/account/agent-credentials",
                   json={"agent_cli": "claude", "api_key": "",
                         "task_models": {"summarize": "m"}},
                   headers=_auth(user_token))
    assert r.status_code == 400
    assert "summarize" in r.json()["detail"]


def test_put_stores_task_models(client, user_token, httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_URL, json=[{"agent_cli": "claude"}])
    r = client.put("/api/account/agent-credentials",
                   json={"agent_cli": "claude", "api_key": "",
                         "task_models": {"generate": "m-g"}},
                   headers=_auth(user_token))
    assert r.status_code == 200
    assert json.loads(httpx_mock.get_requests()[0].content)["task_models"] == {"generate": "m-g"}


def test_put_without_column_is_503(client, user_token, httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_URL, status_code=400, json=_MISSING_WRITE)
    r = client.put("/api/account/agent-credentials",
                   json={"agent_cli": "claude", "api_key": "",
                         "task_models": {"generate": "m-g"}},
                   headers=_auth(user_token))
    assert r.status_code == 503


def test_list_exposes_tasks_and_column_state(client, user_token, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        {"agent_cli": "claude", "model": "m", "task_models": {"review": "m-r"},
         "is_default": True}])
    httpx_mock.add_response(method="GET", url=_DURL, json=[
        {"agent_cli": "claude", "task": "fix", "model": "m-site"}])
    httpx_mock.add_response(method="GET", url=_PURL, json=[
        {"agent_task_routing": {"review": "claude"}}])
    body = client.get("/api/account/agent-credentials", headers=_auth(user_token)).json()
    assert [t["key"] for t in body["agent_tasks"]] == C.AgentTask.known()
    assert body["task_models_enabled"] is True
    assert body["items"][0]["task_models"] == {"review": "m-r"}
    assert body["task_defaults"] == {"claude": {"fix": "m-site"}}
    assert body["task_agents"] == {"review": "claude"}


# ---------------------------------------------------------------------------
# Each caller names its task
# ---------------------------------------------------------------------------

@pytest.fixture
def tasks_seen(monkeypatch):
    seen = []

    def fake_resolve(uid, agent_cli=None, *, task=None):
        seen.append(task)
        return {"agent_cli": "claude", "env": {"ANTHROPIC_MODEL": f"m-{task}"}}
    monkeypatch.setattr(C, "resolve_env", fake_resolve)
    return seen


def _record(rid, owner="user-uuid-1"):
    return {"id": rid, "model_id": "m", "user_id": owner,
            "spec_yaml": "x", "new_component_yamls": []}


def test_user_review_resolves_review_model(client, user_token, monkeypatch, tasks_seen):
    monkeypatch.setattr(_gens, "get", _record)
    launched = {}
    monkeypatch.setattr(_account, "launch_review",
                        lambda rec, **kw: launched.update(kw) or {"status": "started"})
    r = client.post("/api/account/generations/rec-1/review", headers=_auth(user_token))
    assert r.json()["status"] == "started"
    assert tasks_seen == ["review"]
    assert launched["agent_env"] == {"ANTHROPIC_MODEL": "m-review"}


def test_user_fix_resolves_fix_model(client, user_token, monkeypatch, tasks_seen):
    monkeypatch.setattr(_gens, "get", _record)
    monkeypatch.setattr(_reviews, "list_for_record_owner",
                        lambda rid, uid: [{"verdict": "no_pass", "report": "r"}])
    launched = {}
    monkeypatch.setattr(_account, "launch_fix",
                        lambda rec, **kw: launched.update(kw) or {"status": "started"})
    r = client.post("/api/account/generations/rec-1/fix", headers=_auth(user_token))
    assert r.json()["status"] == "started"
    assert tasks_seen == ["fix"]
    assert launched["agent_env"] == {"ANTHROPIC_MODEL": "m-fix"}


def test_admin_review_resolves_review_model(client, admin_token, monkeypatch,
                                            httpx_mock, tasks_seen):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    monkeypatch.setattr(_gens, "get", lambda rid: _record(rid, owner="someone"))
    launched = {}
    monkeypatch.setattr(_admin, "launch_review",
                        lambda rec, **kw: launched.update(kw) or {"status": "started"})
    r = client.post("/api/admin/spec-generations/rec-1/review", headers=_auth(admin_token))
    assert r.json()["status"] == "started"
    assert tasks_seen == ["review"]
    assert launched["agent_env"] == {"ANTHROPIC_MODEL": "m-review"}


# ---------------------------------------------------------------------------
# Admin site-defaults API
# ---------------------------------------------------------------------------

def test_admin_task_defaults_roundtrip(client, admin_token, httpx_mock):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    httpx_mock.add_response(method="GET", url=_DURL, json=[
        {"agent_cli": "claude", "task": "fix", "model": "m-f"}])
    body = client.get("/api/admin/agent-task-defaults",
                      headers=_auth(admin_token)).json()
    assert body["defaults"] == {"claude": {"fix": "m-f"}}
    assert [t["key"] for t in body["tasks"]] == C.AgentTask.known()

    httpx_mock.add_response(method="POST", url=_DURL)
    r = client.put("/api/admin/agent-task-defaults",
                   json={"agent_cli": "claude", "task": "fix", "model": "m-f"},
                   headers=_auth(admin_token))
    assert r.status_code == 200


def test_admin_task_default_rejects_unknown_task(client, admin_token, httpx_mock):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    r = client.put("/api/admin/agent-task-defaults",
                   json={"agent_cli": "claude", "task": "summarize", "model": "m"},
                   headers=_auth(admin_token))
    assert r.status_code == 400
    assert "summarize" in r.json()["detail"]


def test_admin_task_default_requires_admin(client, user_token, httpx_mock):
    from .conftest import mock_admins_no
    mock_admins_no(httpx_mock, "user-uuid-1")
    r = client.get("/api/admin/agent-task-defaults", headers=_auth(user_token))
    assert r.status_code in (403, 404)


# ---------------------------------------------------------------------------
# Per-user task → agent routing (migration 0040)
# ---------------------------------------------------------------------------


def _row2(**over):
    row = {"agent_cli": "kimi", "api_key_enc": "ct", "base_url": None,
           "model": "kimi-k3", "is_default": False}
    row.update(over)
    return row


def test_task_routing_picks_the_routed_agent(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[_row(), _row2()])
    httpx_mock.add_response(method="GET", url=_PURL, json=[
        {"agent_task_routing": {"review": "kimi"}}])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    out = C.resolve_env("u", task="review")
    assert out["agent_cli"] == "kimi"
    assert out["env"]["KIMI_MODEL_NAME"] == "kimi-k3"


def test_unrouted_task_uses_the_default_agent(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[_row(), _row2()])
    httpx_mock.add_response(method="GET", url=_PURL, json=[
        {"agent_task_routing": {"review": "kimi"}}])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    out = C.resolve_env("u", task="fix")
    assert out["agent_cli"] == "claude"


def test_routing_to_a_deleted_credential_falls_back(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(), _row2(agent_cli="codex")])
    httpx_mock.add_response(method="GET", url=_PURL, json=[
        {"agent_task_routing": {"review": "kimi"}}])  # kimi's key was removed
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    assert C.resolve_env("u", task="review")["agent_cli"] == "claude"


def test_fetch_task_routing_tolerates_the_missing_column(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_PURL, status_code=400,
                            json={"code": "42703", "message": "column profiles.agent_task_routing does not exist"})
    assert C.fetch_task_routing("u") == {}


def test_put_task_routing_validates(store):
    with pytest.raises(ValueError):
        C.put_task_routing("u", "summarize", "kimi")
    with pytest.raises(ValueError):
        C.put_task_routing("u", "fix", "nobody")


def test_put_task_routing_requires_a_credential(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_PURL, json=[{"agent_task_routing": None}])
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])  # only claude
    with pytest.raises(ValueError, match="kimi"):
        C.put_task_routing("u", "fix", "kimi")


def test_put_task_routing_sets_and_clears(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_PURL, json=[{"agent_task_routing": {"fix": "claude"}}])
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="PATCH", url=_PURL)
    C.put_task_routing("u", "review", "claude")
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body["agent_task_routing"] == {"fix": "claude", "review": "claude"}

    httpx_mock.add_response(method="GET", url=_PURL, json=[{"agent_task_routing": {"fix": "claude"}}])
    httpx_mock.add_response(method="PATCH", url=_PURL)
    C.put_task_routing("u", "fix", None)
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body["agent_task_routing"] is None


def test_task_routing_route_roundtrip(client, user_token, httpx_mock):
    httpx_mock.add_response(method="GET", url=_PURL, json=[{"agent_task_routing": None}])
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="PATCH", url=_PURL)
    r = client.put("/api/account/agent-task-routing",
                   json={"task": "review", "agent_cli": "claude"},
                   headers=_auth(user_token))
    assert r.status_code == 200
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body["agent_task_routing"] == {"review": "claude"}


def test_task_routing_route_rejects_unknown_task(client, user_token):
    r = client.put("/api/account/agent-task-routing",
                   json={"task": "summarize", "agent_cli": "claude"},
                   headers=_auth(user_token))
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Test-connection probe
# ---------------------------------------------------------------------------

_MODELS_URL = re.compile(r".*anthropic\.com/v1/models.*")
_MESSAGES_URL = re.compile(r".*anthropic\.com/v1/messages.*")


def test_connection_ok(client, user_token, httpx_mock, store):
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_MODELS_URL,
                            json={"data": [{"id": "m1"}, {"id": "m2"}]})
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    httpx_mock.add_response(method="POST", url=_MESSAGES_URL, json={"id": "msg"})
    r = client.post("/api/account/agent-credentials/claude/test",
                    headers=_auth(user_token))
    body = r.json()
    assert r.status_code == 200 and body["ok"] is True
    assert body["models"] == 2 and body["latency_ms"] >= 0
    assert body["generation"]["ok"] is True
    assert body["generation"]["model"] == "claude-default"


def test_connection_auth_failure_is_ok_false(client, user_token, httpx_mock, store):
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_MODELS_URL, status_code=401)
    body = client.post("/api/account/agent-credentials/claude/test",
                       headers=_auth(user_token)).json()
    assert body["ok"] is False and "rejected" in body["detail"]


def test_connection_without_credential_is_404(client, user_token, httpx_mock, store):
    httpx_mock.add_response(method="GET", url=_URL, json=[])
    r = client.post("/api/account/agent-credentials/claude/test",
                    headers=_auth(user_token))
    assert r.status_code == 404


def test_connection_generation_failure_is_named(client, user_token, httpx_mock, store):
    """Models list fine but generation rejected (the model alias doesn't
    exist on this endpoint): the probe names the generation failure."""
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_MODELS_URL,
                            json={"data": [{"id": "m1"}]})
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    httpx_mock.add_response(method="POST", url=_MESSAGES_URL, status_code=404)
    body = client.post("/api/account/agent-credentials/claude/test",
                       headers=_auth(user_token)).json()
    assert body["ok"] is True and body["generation"]["ok"] is False
    assert "no model" in body["generation"]["detail"]


def test_connection_probe_uses_the_models_runs_use(client, user_token, httpx_mock, store):
    """The probe runs on each model a run of this credential uses, by the
    rule `resolve_env` applies: the task pin, else the site-wide task
    default, else the credential's own model."""
    httpx_mock.add_response(method="GET", url=_URL, json=[_row()])
    httpx_mock.add_response(method="GET", url=_MODELS_URL, json={"data": [{"id": "k3"}]})
    httpx_mock.add_response(method="GET", url=_URL, json=[
        _row(task_models={"fix": "k3", "review": "k3"})])
    httpx_mock.add_response(method="GET", url=_DURL, json=[
        {"agent_cli": "claude", "task": "generate", "model": "site-gen"}])
    httpx_mock.add_response(method="POST", url=_MESSAGES_URL, json={"id": "msg"},
                            is_reusable=True)
    body = client.post("/api/account/agent-credentials/claude/test",
                       headers=_auth(user_token)).json()
    assert body["generation"]["ok"] is True
    assert body["generation"]["model"] == "site-gen, claude-default, k3"
    probed = [json.loads(r.content)["model"] for r in httpx_mock.get_requests()
              if r.method == "POST"]
    assert probed == ["site-gen", "claude-default", "k3"]


def test_fetch_models_200_success_false_is_an_error(httpx_mock):
    """Zhipu `/api/v1/models` answers 200 `{success: false, msg}` — not an empty list."""
    httpx_mock.add_response(
        method="GET",
        url="https://open.bigmodel.cn/api/v1/models",
        json={"code": 1001, "msg": "Header中未收到Authorization参数，无法进行身份验证。",
              "success": False},
    )
    with pytest.raises(ValueError, match="Authorization"):
        C.fetch_agent_models("codex", "sk-x", "https://open.bigmodel.cn/api/v1")


def test_fetch_models_accepts_string_ids(httpx_mock):
    httpx_mock.add_response(
        method="GET",
        url="https://api.openai.com/v1/models",
        json={"data": ["gpt-4.1", {"id": "gpt-5"}]},
    )
    assert C.fetch_agent_models("codex", "sk-x") == ["gpt-4.1", "gpt-5"]
