"""Autonomous op promotion is on by default; an explicit flag turns it off."""
from __future__ import annotations

from backend.op_workers.pipeline import coordinator


def _clear_flags(monkeypatch):
    monkeypatch.delenv("OP_PIPELINE_ENABLED", raising=False)
    monkeypatch.delenv("OP_PIPELINE_DISABLED", raising=False)


def test_local_self_host_defaults_enabled(monkeypatch):
    _clear_flags(monkeypatch)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    assert coordinator.pipeline_enabled() is True


def test_service_defaults_enabled(monkeypatch):
    _clear_flags(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    assert coordinator.pipeline_enabled() is True


def test_service_can_explicitly_disable(monkeypatch):
    _clear_flags(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("OP_PIPELINE_ENABLED", "0")
    assert coordinator.pipeline_enabled() is False


def test_explicit_disable_still_works_locally(monkeypatch):
    _clear_flags(monkeypatch)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("OP_PIPELINE_DISABLED", "true")
    assert coordinator.pipeline_enabled() is False


def test_enabled_flag_takes_precedence(monkeypatch):
    monkeypatch.setenv("OP_PIPELINE_ENABLED", "0")
    monkeypatch.setenv("OP_PIPELINE_DISABLED", "0")
    assert coordinator.pipeline_enabled() is False


def test_trace_open_failure_still_releases_proposer_lock(monkeypatch):
    """If the trace file can't be opened (unwritable/full disk) AFTER acquire()
    but BEFORE stream_with_prep runs its own release, the coordinator's finally
    must still return the proposer lock + semaphore slot — otherwise the op is
    stuck 'duplicate' forever and one agent-concurrency permit leaks."""
    import asyncio
    import builtins
    from backend.op_workers import proposal as op_proposal_worker

    released = []
    def _ok(k): return "ok"
    def _rel(k): released.append(k)
    monkeypatch.setattr(op_proposal_worker, "acquire", _ok)
    monkeypatch.setattr(op_proposal_worker, "release", _rel)

    real_open = builtins.open
    def boom(path, *a, **k):
        if "op_pipeline" in str(path) and str(path).endswith(".log"):
            raise OSError("disk full")
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", boom)

    asyncio.run(coordinator._run_pipeline("myop", "/tmp/gap.json", {"op_name": "myop"}))
    assert released == ["myop"]


def test_normal_stream_leaves_release_to_stream_with_prep(monkeypatch):
    """On the happy path the coordinator hands off to stream_with_prep, whose own
    finally releases op_name mid-run (before the review chain). The coordinator
    must NOT also release — a late release could return a slot another run has
    since re-acquired under the same op_name key."""
    import asyncio
    from backend.op_workers import proposal as op_proposal_worker
    import backend.api.routes.op_proposals as opp

    released = []
    def _ok(k): return "ok"
    def _rel(k): released.append(k)
    monkeypatch.setattr(op_proposal_worker, "acquire", _ok)
    monkeypatch.setattr(op_proposal_worker, "release", _rel)

    async def _fake_stream(op_name, prep_fn, *, chain_review):
        yield "data: {\"type\": \"log\", \"text\": \"x\"}\n\n"
    monkeypatch.setattr(opp, "stream_with_prep", _fake_stream)

    asyncio.run(coordinator._run_pipeline("op2", "/tmp/gap.json", {"op_name": "op2"}))
    assert released == []  # coordinator handed off; stream_with_prep owns release
