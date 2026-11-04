"""Job durability: dead-owner locks are reclaimed; interrupted jobs have a terminal state."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

import pytest

from backend.spec_worker import runtime


SUPABASE_URL = "https://test-spec.supabase.co"
ACQUIRE_V2 = re.compile(re.escape(SUPABASE_URL) + r"/rest/v1/rpc/try_acquire_spec_lock_v2")
RELEASE_V2 = re.compile(re.escape(SUPABASE_URL) + r"/rest/v1/rpc/release_spec_lock_v2")
ACQUIRE_V1 = re.compile(re.escape(SUPABASE_URL) + r"/rest/v1/rpc/try_acquire_spec_lock$")
RELEASE_V1 = re.compile(re.escape(SUPABASE_URL) + r"/rest/v1/rpc/release_spec_lock$")
LOGS_URL = re.compile(re.escape(SUPABASE_URL) + r"/rest/v1/spec_generation_logs")
PGRST202 = {
    "code": "PGRST202",
    "message": "Could not find the function public.try_acquire_spec_lock_v2 "
               "in the schema cache",
}
PGRST204 = {
    "code": "PGRST204",
    "message": "Could not find the 'state' column of 'spec_generation_logs' "
               "in the schema cache",
}


@pytest.fixture(autouse=True)
def _reset_runtime(monkeypatch):
    import threading as _th
    # The run lock and the log store are local unless a test opts into
    # Supabase (`supabase_env`, mocked by httpx_mock): a SUPABASE_* the
    # process inherited must not send acquire()/start_job() to a real
    # project, where every worker contends for the same "u|org/m" row.
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    from backend.service import spec_generation_logs as logs
    runtime._reset_schema_compat_for_tests()
    logs._reset_schema_compat_for_tests()
    runtime._model_locks.clear()
    runtime._model_procs.clear()
    runtime._slot_held.clear()
    runtime._lock_epoch.clear()
    runtime._jobs.clear()
    runtime._job_store.clear()
    runtime.semaphore = _th.Semaphore(runtime._max_concurrent)
    yield
    runtime._jobs.clear()
    runtime._job_store.clear()
    runtime._model_locks.clear()


def test_owner_process_alive_matches_proc_ctime(tmp_path):
    pid_dir = tmp_path / "4242"
    pid_dir.mkdir()
    start = pid_dir.stat().st_ctime_ns
    assert runtime.owner_process_alive(f"4242:{start}", proc_root=str(tmp_path))
    assert not runtime.owner_process_alive("4242:0", proc_root=str(tmp_path))
    assert not runtime.owner_process_alive("9999:1", proc_root=str(tmp_path))
    assert not runtime.owner_process_alive("not-a-boot", proc_root=str(tmp_path))


def test_lock_holder_is_local_and_dead_same_host_only(tmp_path):
    pid_dir = tmp_path / "7"
    pid_dir.mkdir()
    live = f"7:{pid_dir.stat().st_ctime_ns}"
    host = "box-a"
    dead = {"host_id": host, "boot_id": "1:0"}
    assert runtime.lock_holder_is_local_and_dead(
        dead, host=host, live_boot=live, proc_root=str(tmp_path))
    live_row = {"host_id": host, "boot_id": live}
    assert not runtime.lock_holder_is_local_and_dead(
        live_row, host=host, live_boot=live, proc_root=str(tmp_path))
    other = {"host_id": "other-vm", "boot_id": "1:0"}
    assert not runtime.lock_holder_is_local_and_dead(
        other, host=host, live_boot=live, proc_root=str(tmp_path))


def test_old_hostname_pid_lock_is_dead_when_pid_gone(tmp_path):
    host = "box-a"
    row = {"host_id": f"{host}#999001", "boot_id": ""}
    assert runtime.lock_holder_is_local_and_dead(
        row, host=host, live_boot="1:1", proc_root=str(tmp_path))


def test_dead_owner_lock_is_reclaimed(monkeypatch):
    from backend.spec_worker import runtime as rt
    released = []
    monkeypatch.setattr(rt, "_list_lock_rows", lambda: [
        {"model_id": "org/m", "host_id": "box-a", "boot_id": "1:0"},
        {"model_id": "keep/me", "host_id": "other-vm", "boot_id": "1:0"},
    ])
    monkeypatch.setattr(rt, "_host_id", lambda: "box-a")
    monkeypatch.setattr(rt, "boot_id", lambda: "9:9")
    monkeypatch.setattr(rt, "owner_process_alive", lambda boot, proc_root="/proc": False)
    monkeypatch.setattr(
        rt, "_delete_lock_row",
        lambda model_id, host_id, owner_boot: released.append(model_id) or True,
    )
    n = rt.reclaim_dead_owner_locks()
    assert n == 1
    assert released == ["org/m"]


def test_live_owner_lock_is_not_reclaimed(monkeypatch, tmp_path):
    from backend.spec_worker import runtime as rt
    pid_dir = tmp_path / str(os.getpid())
    pid_dir.mkdir()
    live = f"{os.getpid()}:{pid_dir.stat().st_ctime_ns}"
    released = []
    monkeypatch.setattr(rt, "_list_lock_rows", lambda: [
        {"model_id": "org/m", "host_id": "box-a", "boot_id": live},
    ])
    monkeypatch.setattr(rt, "_host_id", lambda: "box-a")
    monkeypatch.setattr(rt, "boot_id", lambda: live)
    monkeypatch.setattr(
        rt, "_delete_lock_row",
        lambda *a: released.append(a) or True,
    )
    assert rt.reclaim_dead_owner_locks(proc_root=str(tmp_path)) == 0
    assert released == []


def test_job_status_names_the_agent(monkeypatch):
    from backend.spec_worker import runtime as rt
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-test")
    assert rt.acquire("u|org/m") == "ok"

    def job():
        yield {"type": "done", "success": True, "message": "ok"}

    rt.start_job("u|org/m", job, agent_cli="claude")
    status = rt.job_status("u|org/m")
    assert status["agent_cli"] == "claude"
    assert status["agent_model"] == "claude-opus-test"


def test_interrupted_job_has_terminal_state():
    from backend.spec_worker import runtime as rt
    assert rt.acquire("u|org/m") == "ok"
    woke, done = threading.Event(), threading.Event()

    def job():
        woke.set()
        done.wait(5)
        yield {"type": "done", "success": True, "message": "ok"}

    rt.start_job("u|org/m", job)
    assert woke.wait(5)
    assert rt.job_status("u|org/m")["state"] == "running"
    rt.interrupt_on_shutdown()
    status = rt.job_status("u|org/m")
    assert status["state"] == "interrupted"
    assert status["success"] is False
    assert "retry" in (status["message"] or "").lower()
    # Simulate a new process: in-memory registry gone, store still has the terminal row.
    rt._jobs.clear()
    again = rt.job_status("u|org/m")
    assert again["state"] == "interrupted"
    assert again["success"] is False
    done.set()


def test_orphaned_running_log_is_marked_interrupted(monkeypatch):
    from backend.spec_worker import runtime as rt

    class Logs:
        @staticmethod
        def list_running():
            return [
                {"id": "dead", "run_key": "u|org/m", "host_id": "box-a", "boot_id": "1:0"},
                {"id": "other", "run_key": "u|keep", "host_id": "other-vm", "boot_id": "1:0"},
            ]

        @staticmethod
        def interrupt_ids(ids, message):
            Logs.ids = list(ids)
            Logs.message = message
            return len(ids)

    monkeypatch.setattr(rt, "_host_id", lambda: "box-a")
    monkeypatch.setattr(rt, "boot_id", lambda: "9:9")
    monkeypatch.setattr(rt, "owner_process_alive", lambda boot, proc_root="/proc": False)
    import backend.service.spec_generation_logs as real
    monkeypatch.setattr(real, "list_running", Logs.list_running)
    monkeypatch.setattr(real, "interrupt_ids", Logs.interrupt_ids)
    n = rt.interrupt_orphaned_runs()
    assert n == 1
    assert Logs.ids == ["dead"]
    assert rt.job_status("u|org/m")["state"] == "interrupted"


def test_refine_run_key_does_not_collide_with_generate():
    from backend.api._spec_preflight import job_run_key
    gen = job_run_key("org/m", kind="generate", user_id="u")
    ref = job_run_key("org/m", kind="refine", user_id="u")
    assert gen == "u|org/m"
    assert ref == "u|refine|org/m"
    assert gen != ref


def test_persisted_interrupted_reports_success_false_even_if_row_says_true(monkeypatch):
    from backend.spec_worker import runtime as rt
    monkeypatch.setattr(rt, "_persist_lookup", lambda key: {
        "id": "row-1",
        "state": "interrupted",
        "success": True,
        "message": rt.INTERRUPTED_MESSAGE,
        "run_key": "u|org/m",
    })
    status = rt.job_status("u|org/m")
    assert status["state"] == "interrupted"
    assert status["success"] is False


def test_job_status_idle_when_nothing_was_persisted():
    from backend.spec_worker import runtime as rt
    assert rt.job_status("never-ran") == {"state": "idle"}


def test_unknown_lock_identity_is_not_judged_dead(tmp_path):
    """Same machine, empty boot_id, no hostname#pid — wait for TTL."""
    row = {"host_id": "box-a", "boot_id": ""}
    assert not runtime.lock_holder_is_local_and_dead(
        row, host="box-a", live_boot="1:1", proc_root=str(tmp_path))


def test_machine_id_prefers_explicit_then_fly(monkeypatch):
    monkeypatch.setenv("FLY_MACHINE_ID", "01fly")
    monkeypatch.delenv("HA_MACHINE_ID", raising=False)
    assert runtime.machine_id() == "01fly"
    monkeypatch.setenv("HA_MACHINE_ID", "explicit")
    assert runtime.machine_id() == "explicit"


def test_job_status_on_other_host_keeps_running(monkeypatch):
    """Owner A is alive; a status request that lands on B stays running and
    is not cached as interrupted."""
    from backend.spec_worker import runtime as rt

    persist = {
        "id": "row-a",
        "state": "running",
        "host_id": "machine-a",
        "boot_id": "11:22",
        "success": None,
        "message": "",
        "model_id": "org/m",
        "run_key": "u|org/m",
    }
    monkeypatch.setattr(rt, "_persist_lookup", lambda key: dict(persist))
    monkeypatch.setattr(rt, "_host_id", lambda: "machine-b")
    monkeypatch.setattr(rt, "boot_id", lambda: "99:00")
    interrupted = []
    monkeypatch.setattr(rt, "_persist_interrupt_row", interrupted.append)

    status = rt.job_status("u|org/m")
    assert status["state"] == "running"
    assert interrupted == []
    assert rt._job_store.get("u|org/m") is None
    # A second poll still consults persist (not a cached interrupted).
    again = rt.job_status("u|org/m")
    assert again["state"] == "running"
    assert rt._job_store.get("u|org/m") is None


def test_record_after_interrupt_does_not_hide_terminal_state(monkeypatch):
    """interrupt then record() then a fresh process still sees interrupted."""
    from backend.spec_worker import runtime as rt
    from backend.service import spec_generation_logs as logs

    rows = []

    def begin_run(**kw):
        row = {
            "id": "row-1",
            "state": "running",
            "success": None,
            "message": "",
            "host_id": kw.get("host_id"),
            "boot_id": kw.get("boot_id"),
            "model_id": kw.get("model_id"),
            "user_id": kw.get("user_id"),
            "kind": kw.get("kind"),
            "run_key": kw["run_key"],
            "created_at": "1",
        }
        rows.append(row)
        return "row-1"

    def finish_run(**kw):
        rid = kw.get("row_id")
        for r in rows:
            if rid and r["id"] != rid:
                continue
            if not rid and not (r.get("run_key") == kw["run_key"] and r.get("state") == "running"):
                continue
            r["state"] = kw.get("state", "done")
            r["success"] = kw.get("success")
            r["message"] = kw.get("message")
            return True
        return True

    def latest_by_run_key(run_key):
        cand = [r for r in rows if r.get("run_key") == run_key
                and r.get("state") in ("running", "done", "interrupted")]
        return cand[-1] if cand else None

    def record(**kw):
        existing = latest_by_run_key(kw.get("run_key"))
        if existing:
            existing["log"] = kw.get("log")
            existing["user_id"] = kw.get("user_id")
            existing["model_id"] = kw.get("model_id")
            return True
        rows.append({**kw, "state": None})
        return True

    monkeypatch.setattr(logs, "begin_run", begin_run)
    monkeypatch.setattr(logs, "finish_run", finish_run)
    monkeypatch.setattr(logs, "latest_by_run_key", latest_by_run_key)
    monkeypatch.setattr(logs, "record", record)
    monkeypatch.setattr(rt, "_persist_lookup", latest_by_run_key)

    assert rt.acquire("u|org/m") == "ok"
    woke, done = threading.Event(), threading.Event()

    def job():
        woke.set()
        done.wait(5)
        yield {"type": "done", "success": True, "message": "ok"}

    rt.start_job("u|org/m", job, model_id="org/m", user_id="u", kind="generate")
    assert woke.wait(5)
    rt.interrupt_on_shutdown()
    assert rows[0]["state"] == "interrupted"
    assert rows[0]["model_id"] == "org/m"
    record(user_id="u", model_id="org/m", run_key="u|org/m",
           success=True, log="agent transcript")
    assert rows[0]["state"] == "interrupted"
    assert rows[0]["log"] == "agent transcript"
    assert rows[0]["success"] is False
    assert len(rows) == 1
    rt._jobs.clear()
    rt._job_store.clear()
    status = rt.job_status("u|org/m")
    assert status["state"] == "interrupted"
    assert status["success"] is False
    assert "retry" in (status["message"] or "").lower()
    done.set()


def test_late_done_event_after_interrupt_keeps_success_false():
    """The job thread may still yield a successful `done` after shutdown
    marked interrupted; that must not become the terminal verdict."""
    from backend.spec_worker import runtime as rt
    assert rt.acquire("u|org/m") == "ok"
    woke, gate, yielded = threading.Event(), threading.Event(), threading.Event()

    def job():
        woke.set()
        gate.wait(5)
        yield {"type": "done", "success": True, "message": "ok"}
        yielded.set()

    rt.start_job("u|org/m", job)
    assert woke.wait(5)
    rt.interrupt_on_shutdown()
    gate.set()
    assert yielded.wait(5)
    status = rt.job_status("u|org/m")
    assert status["state"] == "interrupted"
    assert status["success"] is False


def test_job_run_key_rejects_unknown_kind():
    from backend.api._spec_preflight import job_run_key
    with pytest.raises(ValueError, match="unknown job kind"):
        job_run_key("org/m", kind="other")


@pytest.fixture
def supabase_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", SUPABASE_URL)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")


def _wait_job_done(run_key: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    status = runtime.job_status(run_key)
    while time.time() < deadline:
        status = runtime.job_status(run_key)
        if status.get("state") == "done":
            return status
        time.sleep(0.02)
    return status


def test_v2_rpc_404_falls_back_to_v1_and_job_completes(supabase_env, httpx_mock, caplog):
    """Production without 0034: v2 is 404/PGRST202. Probe once, then the
    0009 RPCs must admit the run and the job must finish."""
    v1_host = runtime._legacy_lock_host_id()
    httpx_mock.add_response(
        method="POST", url=ACQUIRE_V2, status_code=404, json=PGRST202,
    )
    httpx_mock.add_response(
        method="POST", url=ACQUIRE_V1,
        json=[{"acquired": True, "held_by": v1_host,
               "expires_at": "2099-01-01T00:00:00Z"}],
        is_reusable=True,
    )
    httpx_mock.add_response(method="POST", url=RELEASE_V1, json=True, is_reusable=True)
    httpx_mock.add_response(
        method="POST", url=LOGS_URL, status_code=201, json=[{"id": "row"}],
        is_reusable=True,
    )
    httpx_mock.add_response(method="PATCH", url=LOGS_URL, status_code=204, is_reusable=True)

    with caplog.at_level(logging.WARNING):
        assert runtime.acquire("u|org/m") == "ok"
        assert runtime.acquire("u|org/n") == "ok"
    v2_acq = [r for r in httpx_mock.get_requests()
              if r.url.path.endswith("/rpc/try_acquire_spec_lock_v2")]
    assert len(v2_acq) == 1
    v1_acq = [r for r in httpx_mock.get_requests()
              if r.url.path.endswith("/rpc/try_acquire_spec_lock")]
    assert len(v1_acq) == 2
    body = json.loads(v1_acq[0].content)
    assert body["p_host_id"] == v1_host
    assert "p_boot_id" not in body
    assert any("try_acquire_spec_lock" in r.message and "hostname#pid" in r.message
               for r in caplog.records)

    done = threading.Event()

    def job():
        yield {"type": "done", "success": True, "message": "ok"}
        done.set()

    runtime.start_job("u|org/m", job, model_id="org/m", user_id="u", kind="generate")
    assert done.wait(5)
    status = _wait_job_done("u|org/m")
    assert status["state"] == "done"
    assert status["success"] is True
    runtime.release("u|org/m")
    runtime.release("u|org/n")
    rel = [r for r in httpx_mock.get_requests()
           if r.url.path.endswith("/rpc/release_spec_lock")]
    assert rel
    rel_body = json.loads(rel[0].content)
    assert rel_body["p_host_id"] == v1_host
    assert "p_boot_id" not in rel_body


def test_unknown_log_columns_do_not_fail_generation(supabase_env, httpx_mock, caplog):
    """0034 log columns missing: skip running-state persistence, keep the
    original transcript INSERT, and the job still finishes."""
    from httpx import Response
    from backend.service import spec_generation_logs as logs

    httpx_mock.add_response(
        method="POST", url=ACQUIRE_V2,
        json=[{"acquired": True, "held_by": runtime._host_id(),
               "boot_id": runtime.boot_id(),
               "expires_at": "2099-01-01T00:00:00Z"}],
        is_reusable=True,
    )
    httpx_mock.add_response(method="POST", url=RELEASE_V2, json=True, is_reusable=True)

    def on_logs(request):
        if request.method == "GET":
            return Response(400, json=PGRST204)
        body = json.loads(request.content) if request.content else {}
        if any(k in body for k in ("state", "host_id", "boot_id", "finished_at", "kind")):
            return Response(400, json=PGRST204)
        if request.method == "PATCH" and "message" in body:
            return Response(400, json=PGRST204)
        return Response(201, json=[])

    httpx_mock.add_callback(on_logs, url=LOGS_URL, is_reusable=True)

    with caplog.at_level(logging.WARNING):
        assert runtime.acquire("u|org/m") == "ok"
        done = threading.Event()

        def job():
            yield {"type": "done", "success": True, "message": "ok"}
            done.set()

        runtime.start_job("u|org/m", job, model_id="org/m", user_id="u", kind="generate")
        assert done.wait(5)
        status = _wait_job_done("u|org/m")
        assert status["state"] == "done"
        assert status["success"] is True
        assert logs.record(user_id="u", model_id="org/m", log="transcript",
                           run_key="u|org/m", success=True) is True
    warns = [r for r in caplog.records if "0034 durability columns" in r.message]
    assert len(warns) == 1
    posts = [r for r in httpx_mock.get_requests()
             if r.method == "POST" and "spec_generation_logs" in str(r.url)]
    legacy = [json.loads(r.content) for r in posts
              if r.content and "state" not in json.loads(r.content)]
    assert legacy
    assert legacy[-1]["log"] == "transcript"
    assert "state" not in legacy[-1]
    runtime.release("u|org/m")


def test_bind_job_log_sidecar_flushes_on_orphan(tmp_path, monkeypatch):
    logp = tmp_path / "run.log"
    logp.write_text("partial transcript\n")
    monkeypatch.setattr(runtime, "_LOG_DIR", str(tmp_path))
    runtime.bind_job_log("u|org/m", str(logp), user_id="u", model_id="org/m")
    sidecar = tmp_path / (logp.name + runtime._LOG_SIDECAR_SUFFIX)
    assert sidecar.is_file()
    calls = []
    monkeypatch.setattr(
        "backend.service.spec_generation_logs.record",
        lambda **kw: calls.append(kw) or True)
    n = runtime.flush_orphaned_log_sidecars()
    assert n == 1
    assert calls and calls[0]["success"] is False
    assert "partial transcript" in calls[0]["log"]
    assert not sidecar.exists()


def test_orphan_sidecar_stays_when_record_fails(tmp_path, monkeypatch):
    logp = tmp_path / "run.log"
    logp.write_text("partial transcript\n")
    monkeypatch.setattr(runtime, "_LOG_DIR", str(tmp_path))
    runtime.bind_job_log("u|org/m", str(logp), user_id="u", model_id="org/m")
    sidecar = tmp_path / (logp.name + runtime._LOG_SIDECAR_SUFFIX)
    assert sidecar.is_file()
    monkeypatch.setattr(
        "backend.service.spec_generation_logs.record",
        lambda **kw: False)
    n = runtime.flush_orphaned_log_sidecars()
    assert n == 0
    assert sidecar.is_file()


def test_a_blocked_runs_error_kind_and_gaps_reach_its_log_row(monkeypatch):
    """The terminal `done`'s machine fields are written with the verdict,
    and a process that lost the job reads them back from the row."""
    from backend.service import spec_generation_logs as logs
    rows = {}
    monkeypatch.setattr(logs, "begin_run", lambda **kw: "row-1")

    def finish_run(**kw):
        rows[kw["row_id"]] = {"state": kw["state"], "success": kw["success"],
                              "message": kw["message"],
                              "error_kind": kw["error_kind"],
                              "engine_gaps": kw["engine_gaps"]}
        return True
    monkeypatch.setattr(logs, "finish_run", finish_run)
    monkeypatch.setattr(runtime, "_persist_lookup", lambda key: rows.get("row-1"))
    gaps = [{"capability": "gather extent", "deployments": ["tp8"], "stages": ["decode"]}]

    assert runtime.acquire("u|org/m") == "ok"

    def job():
        yield {"type": "done", "success": False, "message": "blocked",
               "error_kind": "blocked_by_engine", "engine_gaps": gaps,
               "record": {"outcome": "recorded", "id": "p1"}}

    runtime.start_job("u|org/m", job, model_id="org/m", user_id="u", kind="generate")
    status = _wait_job_done("u|org/m")
    assert status["state"] == "done"
    deadline = time.time() + 5
    while "row-1" not in rows and time.time() < deadline:
        time.sleep(0.01)
    assert rows["row-1"]["error_kind"] == "blocked_by_engine"
    assert rows["row-1"]["engine_gaps"] == gaps
    runtime._jobs.clear()
    runtime._job_store.clear()
    after_restart = runtime.job_status("u|org/m")
    assert after_restart["error_kind"] == "blocked_by_engine"
    assert after_restart["engine_gaps"] == gaps


def test_closing_a_row_before_migration_0002_retries_without_its_columns(
        supabase_env, httpx_mock):
    from httpx import Response
    from backend.service import spec_generation_logs as logs
    bodies = []

    def on_logs(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "error_kind" in body:
            return Response(400, json={"code": "PGRST204", "message":
                                       "Could not find the 'error_kind' column"})
        return Response(204)

    httpx_mock.add_callback(on_logs, url=LOGS_URL, is_reusable=True)
    assert logs.finish_run(run_key="k", success=False, message="m", row_id="r",
                           error_kind="blocked_by_engine", engine_gaps=[]) is True
    assert [("error_kind" in b) for b in bodies] == [True, False]
    assert logs._durability_on()


def test_a_late_upload_lands_on_the_row_its_own_run_began(monkeypatch):
    """Run A releases its key before it uploads its transcript, so run B of
    the same key can begin first. A's transcript (and the record it wrote)
    still goes onto A's row, never onto B's newer row of that key."""
    from backend.service import spec_generation_logs as logs
    began = iter(["row-A", "row-B"])
    monkeypatch.setattr(logs, "begin_run", lambda **kw: next(began))
    monkeypatch.setattr(logs, "finish_run", lambda **kw: True)
    class _Env:  # the log store is configured; the run lock stays local
        configured = staticmethod(lambda: True)
        service_role = staticmethod(lambda: ("https://x.supabase.co", "k"))

    monkeypatch.setattr(logs, "supabase_env", _Env)
    sent = []

    class _Ok:
        status_code, text = 204, ""

    monkeypatch.setattr(logs, "_log_request",
                        lambda method, *a, **kw: sent.append((method, kw)) or _Ok())
    key = "u|org/m"
    b_began, a_done = threading.Event(), threading.Event()

    def run_a():
        yield {"type": "done", "success": True, "message": "ok"}
        runtime.release(key)
        assert b_began.wait(5)
        logs.record(user_id="u", model_id="org/m", log="A transcript",
                    run_key=key, success=True, record_id="rec-A")
        a_done.set()

    def run_b():
        b_began.set()
        assert a_done.wait(5)
        yield {"type": "done", "success": True, "message": "ok"}

    assert runtime.acquire(key) == "ok"
    runtime.start_job(key, run_a, model_id="org/m", user_id="u")
    deadline = time.time() + 5
    while runtime.acquire(key) != "ok":
        assert time.time() < deadline
        time.sleep(0.01)
    runtime.start_job(key, run_b, model_id="org/m", user_id="u")
    assert a_done.wait(5)
    (method, kw), = [s for s in sent if s[1].get("json", {}).get("log")]
    assert method == "patch"
    assert kw["params"] == {"id": "eq.row-A"}
    assert kw["json"]["record_id"] == "rec-A"
    _wait_job_done(key)
    runtime.release(key)
