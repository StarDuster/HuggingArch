"""A generate/refine run gives its lock back on every path, and a click that
does not start a run touches nothing a running one uses."""
from __future__ import annotations

import pytest


def test_a_failed_prep_releases_the_run_lock(monkeypatch):
    """Prep raises before `stream_agent` (whose finally releases the lock on
    every other path) ever starts: the job itself gives the lock back."""
    from backend.spec_worker import worker
    released = []
    monkeypatch.setattr(worker, "release", released.append)

    def prep():
        raise OSError("hf down")

    events = list(worker.generation_job("org/m", prep, run_key="u|org/m")())
    assert events[-1]["type"] == "done" and not events[-1]["success"]
    assert released == ["u|org/m"]


def test_the_stamp_reaches_the_agent(monkeypatch):
    from backend.spec_worker import worker
    seen = {}

    def fake_stream(model_id, prep_result, *, run_key=None, failover=None):
        seen.update(prep_result, run_key=run_key)
        yield 'data: {"type": "done", "success": true}\n\n'

    monkeypatch.setattr(worker, "stream_agent", fake_stream)
    events = list(worker.generation_job(
        "org/m", lambda: {"prompt": "p"}, run_key="u|org/m",
        stamp={"user_id": "u", "agent_cli": "claude"})())
    assert events[-1] == {"type": "done", "success": True}
    assert seen == {"prompt": "p", "user_id": "u", "agent_cli": "claude", "run_key": "u|org/m"}


@pytest.mark.parametrize("status", ["duplicate", "overloaded"])
def test_a_refused_click_spends_no_quota(monkeypatch, status):
    from backend.service.routes import spec as routes
    monkeypatch.setattr(routes.spec_worker, "acquire", lambda key: status)
    monkeypatch.setattr(routes, "_quota_hits", {})
    refused = routes._claim_run("u|org/m", "org/m", "u", is_admin=False)
    assert refused["status"] == status
    assert not routes._quota_hits.get("u")


def test_over_quota_gives_the_lock_back(monkeypatch):
    from backend.service.routes import spec as routes
    released = []
    monkeypatch.setattr(routes.spec_worker, "acquire", lambda key: "ok")
    monkeypatch.setattr(routes.spec_worker, "release", released.append)
    monkeypatch.setattr(routes, "_quota_ok", lambda user_id: False)
    refused = routes._claim_run("u|org/m", "org/m", "u", is_admin=False)
    assert refused["status"] == "quota_exceeded"
    assert released == ["u|org/m"]


def test_a_refused_generate_leaves_the_running_overlay_alone(monkeypatch):
    """The overlay is prepared after the lock: a duplicate click must not
    clear the agent home a running agent is using."""
    from backend.service.routes import spec as routes
    touched = []
    monkeypatch.setattr(routes, "_peruser_active", lambda: True)
    monkeypatch.setattr(routes._overlay, "allocate_run_overlay",
                        lambda *a, **k: touched.append(a) or "/tmp/x")
    monkeypatch.setattr(routes.spec_worker, "acquire", lambda key: "duplicate")
    monkeypatch.setattr(routes, "_quota_room", lambda user_id: True)
    monkeypatch.setattr(routes._creds, "resolve_env",
                        lambda uid, **kw: {"agent_cli": "claude", "env": {}})
    monkeypatch.setattr(routes.spec_worker, "get_agent_binary", lambda: "sh")
    monkeypatch.setattr(routes._auth, "is_admin", lambda uid: False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **k: (_ for _ in ()).throw(OSError("offline")))
    out = routes.service_generate_spec(routes.GenerateSpecRequest(model_id="org/m"),
                                       user={"sub": "u"})
    assert out["status"] == "duplicate"
    assert touched == []
