"""backend.spec_worker.runtime — cross-VM lock paths.

Two paths to cover:
  • Dev fallback (no SUPABASE env)        — in-memory dict + semaphore.
  • Supabase-backed (prod)                — try_acquire_spec_lock RPC.

The Supabase tests use pytest-httpx to intercept REST calls; nothing
in this file touches a real DB.
"""
from __future__ import annotations

import re

import pytest


SUPABASE_URL = "https://test-spec.supabase.co"
ACQUIRE_URL = re.compile(re.escape(SUPABASE_URL) + r"/rest/v1/rpc/try_acquire_spec_lock_v2")
RELEASE_URL = re.compile(re.escape(SUPABASE_URL) + r"/rest/v1/rpc/release_spec_lock_v2")


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    """Wipe in-memory lock state before each test so cases don't leak."""
    from backend.spec_worker import runtime
    import threading
    runtime._reset_schema_compat_for_tests()
    runtime._model_locks.clear()
    runtime._model_procs.clear()
    runtime._slot_held.clear()
    runtime._lock_epoch.clear()
    runtime._jobs.clear()
    runtime._job_store.clear()
    # Re-create semaphore at default cap to undo leftover acquires.
    runtime.semaphore = threading.Semaphore(runtime._max_concurrent)
    yield
    runtime._model_locks.clear()
    runtime._model_procs.clear()
    runtime._slot_held.clear()
    runtime._jobs.clear()
    runtime._job_store.clear()


@pytest.fixture
def dev_env(monkeypatch):
    """Force the dev fallback by clearing SUPABASE_*."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)


@pytest.fixture
def supabase_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", SUPABASE_URL)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")


# ---------------------------------------------------------------------------
# In-memory dev fallback
# ---------------------------------------------------------------------------

class TestDevFallback:
    def test_acquire_then_duplicate(self, dev_env):
        from backend.spec_worker import runtime
        assert runtime.acquire("m1") == "ok"
        assert runtime.acquire("m1") == "duplicate"

    def test_release_lets_reacquire(self, dev_env):
        from backend.spec_worker import runtime
        runtime.acquire("m1")
        runtime.release("m1")
        assert runtime.acquire("m1") == "ok"

    def test_different_models_independent(self, dev_env):
        from backend.spec_worker import runtime
        assert runtime.acquire("m1") == "ok"
        assert runtime.acquire("m2") == "ok"

    def test_overloaded_when_semaphore_exhausted(self, dev_env, monkeypatch):
        from backend.spec_worker import runtime
        import threading
        # Cap to 2 → 3rd distinct model should report overloaded
        monkeypatch.setattr(runtime, "semaphore", threading.Semaphore(2))
        assert runtime.acquire("m1") == "ok"
        assert runtime.acquire("m2") == "ok"
        assert runtime.acquire("m3") == "overloaded"

    def test_release_idempotent(self, dev_env):
        """Calling release on a never-acquired model_id mustn't crash."""
        from backend.spec_worker import runtime
        runtime.release("never-acquired")  # no error
        # State stays clean
        assert runtime.acquire("never-acquired") == "ok"


# ---------------------------------------------------------------------------
# Supabase-backed path
# ---------------------------------------------------------------------------

class TestSupabaseBacked:
    def test_acquire_calls_try_acquire_rpc(self, supabase_env, httpx_mock):
        from backend.spec_worker import runtime
        captured = {}

        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(200, json=[
                {"acquired": True, "held_by": runtime._host_id(),
                 "boot_id": runtime.boot_id(),
                 "expires_at": "2099-01-01T00:00:00Z"},
            ])

        httpx_mock.add_callback(_capture, method="POST", url=ACQUIRE_URL)
        result = runtime.acquire("model-1")
        assert result == "ok"
        assert captured["body"]["p_model_id"] == "model-1"
        assert captured["body"]["p_host_id"] == runtime._host_id()
        assert captured["body"]["p_boot_id"] == runtime.boot_id()
        assert captured["body"]["p_ttl_seconds"] == runtime._LOCK_TTL_SECONDS

    def test_acquire_duplicate_when_held_by_other(self, supabase_env, httpx_mock):
        from backend.spec_worker import runtime
        httpx_mock.add_response(
            method="POST", url=ACQUIRE_URL,
            json=[{"acquired": False, "held_by": "other-host#999",
                   "expires_at": "2099-01-01T00:00:00Z"}],
        )
        assert runtime.acquire("model-1") == "duplicate"

    def test_release_calls_release_rpc(self, supabase_env, httpx_mock):
        from backend.spec_worker import runtime
        httpx_mock.add_response(
            method="POST", url=ACQUIRE_URL,
            json=[{"acquired": True, "held_by": runtime._host_id(),
                   "boot_id": runtime.boot_id(),
                   "expires_at": "2099-01-01T00:00:00Z"}],
        )
        captured = {}
        def _capture(req):
            import json
            captured["body"] = json.loads(req.content)
            from httpx import Response
            return Response(200, json=True)
        httpx_mock.add_callback(_capture, method="POST", url=RELEASE_URL)

        runtime.acquire("model-1")
        runtime.release("model-1")
        assert captured["body"]["p_model_id"] == "model-1"
        assert captured["body"]["p_host_id"] == runtime._host_id()
        assert captured["body"]["p_boot_id"] == runtime.boot_id()

    def test_acquire_overloaded_when_supabase_unreachable(self, supabase_env, httpx_mock):
        """RPC failure should release the local slot + return overloaded
        so the caller doesn't hold a synthetic lock."""
        from backend.spec_worker import runtime
        httpx_mock.add_response(
            method="POST", url=ACQUIRE_URL,
            status_code=500, text="db down",
        )
        result = runtime.acquire("model-1")
        assert result == "overloaded"
        # The semaphore slot must have been released — proven by the
        # fact that we can fully drain it next.
        # Default cap is 10 → 10 acquires should still succeed
        succ = sum(
            runtime.semaphore.acquire(blocking=False)
            for _ in range(runtime._max_concurrent)
        )
        assert succ == runtime._max_concurrent

    def test_local_overload_short_circuits_before_db(self, supabase_env, monkeypatch, httpx_mock):
        """If the local semaphore is exhausted, we must NOT make a DB call."""
        from backend.spec_worker import runtime
        import threading
        monkeypatch.setattr(runtime, "semaphore", threading.Semaphore(0))
        # No httpx_mock.add_response — if a network call escapes, the test fails.
        assert runtime.acquire("model-1") == "overloaded"

    def test_double_release_does_not_mint_a_phantom_slot(self, dev_env, monkeypatch):
        """release() is idempotent per model_id: releasing the same run twice
        (stop() then the SSE finally; a route release + the runner's release_fn)
        returns its slot exactly once. Needs a concurrent holder + cap≥2 so the
        over-release wouldn't be masked by BoundedSemaphore's at-bound ValueError.
        """
        from backend.spec_worker import runtime
        import threading
        sem = threading.BoundedSemaphore(2)
        monkeypatch.setattr(runtime, "semaphore", sem)
        assert runtime.acquire("m1") == "ok"      # value 1
        assert runtime.acquire("m2") == "ok"      # value 0 (m2 still holds a slot)
        runtime.release("m1")                      # value 1
        runtime.release("m1")                      # double — must be a no-op
        # Exactly ONE real slot is free (m2 still holds the other). A phantom
        # permit from the double-release would let a second acquire through.
        assert sem.acquire(blocking=False) is True
        assert sem.acquire(blocking=False) is False


# ---------------------------------------------------------------------------
# Routing test: which path runs given the env
# ---------------------------------------------------------------------------

class TestSameHostDedup:
    """One VM must not run the same key twice. The cross-VM RPC reports
    `acquired` only when hostname and boot_id both match, so a second
    caller on this process is invisible to it — the local reservation
    refuses the duplicate. A double-clicked review used to start two
    agents that shared one overlay, and the first to finish killed the
    other."""

    def _ours(self):
        from backend.spec_worker import runtime
        return [{"acquired": True, "held_by": runtime._host_id(),
                 "boot_id": runtime.boot_id(),
                 "expires_at": "2099-01-01T00:00:00Z"}]

    def test_second_acquire_same_host_is_duplicate(self, supabase_env, httpx_mock):
        from backend.spec_worker import runtime
        # Exactly one RPC is registered: the second acquire must be refused
        # locally, before it reaches the DB (pytest-httpx fails on an extra call).
        httpx_mock.add_response(method="POST", url=ACQUIRE_URL, json=self._ours())
        assert runtime.acquire("review|u|rec") == "ok"
        assert runtime.acquire("review|u|rec") == "duplicate"

    def test_duplicate_does_not_leak_a_slot(self, supabase_env, httpx_mock):
        from backend.spec_worker import runtime
        httpx_mock.add_response(method="POST", url=ACQUIRE_URL, json=self._ours())
        runtime.acquire("k")
        for _ in range(5):
            assert runtime.acquire("k") == "duplicate"
        # 1 slot held by the live run; the refused callers returned theirs.
        free = sum(runtime.semaphore.acquire(blocking=False)
                   for _ in range(runtime._max_concurrent))
        assert free == runtime._max_concurrent - 1

    def test_failed_cross_vm_lock_frees_the_local_reservation(self, supabase_env, httpx_mock):
        from backend.spec_worker import runtime
        httpx_mock.add_response(method="POST", url=ACQUIRE_URL, status_code=500, text="down")
        httpx_mock.add_response(method="POST", url=ACQUIRE_URL,
                                json=[{"acquired": False, "held_by": "other#1",
                                       "expires_at": "2099-01-01T00:00:00Z"}])
        httpx_mock.add_response(method="POST", url=ACQUIRE_URL, json=self._ours())
        assert runtime.acquire("k") == "overloaded"
        assert runtime.acquire("k") == "duplicate"   # held on another VM
        assert runtime.acquire("k") == "ok"          # neither failure left a local claim

    def test_release_lets_same_host_reacquire(self, supabase_env, httpx_mock):
        from backend.spec_worker import runtime
        httpx_mock.add_response(method="POST", url=ACQUIRE_URL, json=self._ours(), is_reusable=True)
        httpx_mock.add_response(method="POST", url=RELEASE_URL, json=True)
        assert runtime.acquire("k") == "ok"
        runtime.release("k")
        assert runtime.acquire("k") == "ok"

    def test_concurrent_acquires_admit_exactly_one(self, dev_env):
        import threading
        from backend.spec_worker import runtime
        start, results = threading.Barrier(8), []
        def go():
            start.wait()
            results.append(runtime.acquire("k"))
        ts = [threading.Thread(target=go) for _ in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        assert results.count("ok") == 1
        assert results.count("duplicate") == 7

    def test_different_keys_independent(self, supabase_env, httpx_mock):
        from backend.spec_worker import runtime
        httpx_mock.add_response(method="POST", url=ACQUIRE_URL, json=self._ours(), is_reusable=True)
        assert runtime.acquire("review|u|a") == "ok"
        assert runtime.acquire("review|u|b") == "ok"


class TestStaleLocalReservation:
    """A local reservation older than the lock TTL belongs to a run that can
    no longer be going (a caller that never reached release()); it is
    reclaimed like the DB lock self-heals, and its concurrency slot comes back."""

    def test_stale_reservation_is_reclaimed_and_its_slot_returned(self, dev_env):
        import time
        from backend.spec_worker import runtime
        assert runtime.acquire("k") == "ok"          # a run that then leaks
        runtime._model_locks["k"] = time.monotonic() - runtime._LOCK_TTL_SECONDS - 1
        assert runtime.acquire("k") == "ok"          # reclaimed, not duplicate
        free = sum(runtime.semaphore.acquire(blocking=False)
                   for _ in range(runtime._max_concurrent))
        assert free == runtime._max_concurrent - 1   # only the live run holds a slot

    def test_concurrent_callers_reclaim_a_stale_reservation_once(self, dev_env):
        """Many callers racing on the same stale entry: exactly one reclaims and
        runs; the rest see its fresh reservation and are refused."""
        import threading, time
        from backend.spec_worker import runtime
        assert runtime.acquire("k") == "ok"
        runtime._model_locks["k"] = time.monotonic() - runtime._LOCK_TTL_SECONDS - 1
        start, results = threading.Barrier(8), []
        def go():
            start.wait()
            results.append(runtime.acquire("k"))
        ts = [threading.Thread(target=go) for _ in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        assert results.count("ok") == 1 and results.count("duplicate") == 7
        free = sum(runtime.semaphore.acquire(blocking=False)
                   for _ in range(runtime._max_concurrent))
        assert free == runtime._max_concurrent - 1

    def test_fresh_reservation_still_refuses(self, dev_env):
        from backend.spec_worker import runtime
        assert runtime.acquire("k") == "ok"
        assert runtime.acquire("k") == "duplicate"


class TestSupersededRun:
    """A run whose reservation was reclaimed can wake up later. It must not
    act for the key's new owner: its release() is ignored and owns() says no,
    so it neither frees the new run's lock nor starts a second agent."""

    def _stale_and_retake(self, runtime):
        import time
        runtime._model_locks["k"] = time.monotonic() - runtime._LOCK_TTL_SECONDS - 1
        assert runtime.acquire("k") == "ok"          # the new owner

    def test_late_release_from_a_superseded_job_is_ignored(self, dev_env):
        import threading
        from backend.spec_worker import runtime
        assert runtime.acquire("k") == "ok"
        woke, done, seen = threading.Event(), threading.Event(), {}

        def zombie():
            woke.wait(5)
            seen["owns"] = runtime.owns("k")
            runtime.release("k")                    # late
            done.set()
            yield {"type": "done", "success": True}

        runtime.start_job("k", zombie)               # started under the first epoch
        self._stale_and_retake(runtime)
        woke.set(); assert done.wait(5)
        assert seen["owns"] is False
        assert "k" in runtime._model_locks           # new owner's reservation intact
        assert runtime.acquire("k") == "duplicate"
        free = sum(runtime.semaphore.acquire(blocking=False)
                   for _ in range(runtime._max_concurrent))
        assert free == runtime._max_concurrent - 1   # new owner still holds its slot

    def test_the_live_owner_still_releases(self, dev_env):
        import threading
        from backend.spec_worker import runtime
        assert runtime.acquire("k") == "ok"
        done = threading.Event()

        def job():
            runtime.release("k")
            done.set()
            yield {"type": "done", "success": True}

        runtime.start_job("k", job)
        assert done.wait(5)
        assert "k" not in runtime._model_locks
        assert runtime.acquire("k") == "ok"

    def test_a_request_thread_is_not_tied_to_an_epoch(self, dev_env):
        from backend.spec_worker import runtime
        assert runtime.acquire("k") == "ok"
        assert runtime.owns("k") is True


class TestRpcRaising:
    def test_a_non_json_lock_reply_gives_back_reservation_and_slot(self, supabase_env, httpx_mock):
        """_supabase_rpc turns network errors into None, but a body that is not
        JSON raises out of it. acquire must return what it took before the
        error propagates, or the slot leaks for good."""
        from backend.spec_worker import runtime
        httpx_mock.add_response(method="POST", url=ACQUIRE_URL, text="<html>proxy error</html>")
        with pytest.raises(ValueError):
            runtime.acquire("k")
        assert "k" not in runtime._model_locks
        free = sum(runtime.semaphore.acquire(blocking=False)
                   for _ in range(runtime._max_concurrent))
        assert free == runtime._max_concurrent


class TestReleaseOrdersTheDbLock:
    def test_a_new_acquire_waits_for_the_release_rpc(self, dev_env, monkeypatch):
        """release() frees the key locally, then tells the DB. A new caller in
        between must not take the DB row first, or the late release RPC would
        delete it under the new run (this host owns the row either way)."""
        import threading
        import time
        from backend.spec_worker import runtime
        calls, in_release, go = [], threading.Event(), threading.Event()

        def rpc(fn, body, timeout=5.0):
            calls.append(fn)
            if fn == "release_spec_lock_v2":
                in_release.set()
                go.wait(5)
                calls.append("release returned")
                return None
            return [{"acquired": True}]

        monkeypatch.setattr(runtime, "_supabase_lock_enabled", lambda: True)
        monkeypatch.setattr(runtime, "_supabase_rpc", rpc)
        assert runtime.acquire("k") == "ok"
        old = threading.Thread(target=runtime.release, args=("k",))
        old.start()
        assert in_release.wait(5)
        got = {}
        new = threading.Thread(target=lambda: got.setdefault("r", runtime.acquire("k")))
        new.start()
        time.sleep(0.2)
        assert "r" not in got                     # held until the release RPC is done
        go.set()
        old.join(5); new.join(5)
        assert got["r"] == "ok"
        assert calls == ["try_acquire_spec_lock_v2", "release_spec_lock_v2",
                         "release returned", "try_acquire_spec_lock_v2"]
        assert "k" in runtime._model_locks        # the new owner's reservation stands

    def test_a_raising_release_rpc_still_kills_and_returns_the_slot(self, dev_env, monkeypatch):
        from backend.spec_worker import runtime

        class Proc:
            killed = False
            def poll(self): return None
            def kill(self): Proc.killed = True

        def rpc(fn, body, timeout=5.0):
            if fn == "release_spec_lock_v2":
                raise ValueError("not json")
            return [{"acquired": True}]

        monkeypatch.setattr(runtime, "_supabase_lock_enabled", lambda: True)
        monkeypatch.setattr(runtime, "_supabase_rpc", rpc)
        assert runtime.acquire("k") == "ok"
        runtime._model_procs["k"] = Proc()
        with pytest.raises(ValueError):
            runtime.release("k")
        assert Proc.killed
        free = sum(runtime.semaphore.acquire(blocking=False)
                   for _ in range(runtime._max_concurrent))
        assert free == runtime._max_concurrent


class TestRouting:
    def test_supabase_lock_enabled_flag(self, monkeypatch):
        from backend.spec_worker import runtime
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        assert runtime._supabase_lock_enabled() is False

        monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_x")
        assert runtime._supabase_lock_enabled() is True

    def test_partial_env_treats_as_disabled(self, monkeypatch):
        """URL set but no service-role key → can't auth → must fall back."""
        from backend.spec_worker import runtime
        monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        assert runtime._supabase_lock_enabled() is False


# ---------------------------------------------------------------------------
# Release path: must kill any still-running agent subprocess.
#
# Without this, an SSE client disconnect (browser tab close mid-run)
# causes stream_agent's finally to call release(), which clears the
# semaphore + lock dict but leaves the spawned claude/codex CLI alive.
# Each leaked proc holds an API socket; after a few stuck runs the
# machine has multiple zombies and `Too many concurrent tasks` is
# misleading because the *visible* lock count looks fine.
# ---------------------------------------------------------------------------

class TestReleaseKillsChild:
    def test_release_reaps_running_child(self):
        import subprocess
        import time
        from backend.spec_worker import runtime

        proc = subprocess.Popen(["sleep", "30"])
        runtime._model_procs["test_kill"] = proc
        runtime._model_locks["test_kill"] = True
        runtime.semaphore.acquire()
        try:
            runtime.release("test_kill")
            # SIGKILL is async; give the kernel a beat to reap.
            for _ in range(20):
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
            assert proc.poll() is not None, "release() did not reap running child"
            assert "test_kill" not in runtime._model_procs
            assert "test_kill" not in runtime._model_locks
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_release_noop_on_already_exited_child(self):
        """Normal-completion path: stream_agent finally calls release(),
        but the proc has already returned. release() must NOT raise on
        a dead proc.
        """
        import subprocess
        from backend.spec_worker import runtime

        proc = subprocess.Popen(["true"])
        proc.wait()
        runtime._model_procs["test_done"] = proc
        runtime._model_locks["test_done"] = True
        runtime.semaphore.acquire()
        runtime.release("test_done")
        assert "test_done" not in runtime._model_procs

    def test_release_noop_on_no_child(self):
        """Duplicate / overloaded path: acquire failed, no proc was ever
        spawned. release() may still get called (defensive finally) and
        must be a clean no-op.
        """
        from backend.spec_worker import runtime

        runtime._model_locks["test_no_proc"] = True
        runtime.semaphore.acquire()
        runtime.release("test_no_proc")
        assert "test_no_proc" not in runtime._model_locks
