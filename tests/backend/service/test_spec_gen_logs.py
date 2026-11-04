"""Per-user run-log capture (spec_generation_logs)."""
from __future__ import annotations


def test_truncate_keeps_tail():
    from backend.service import spec_generation_logs as L
    small = "x" * 10
    assert L._truncate(small) == small
    big = "A" * 5 + "B" * L._MAX_LOG_CHARS
    out = L._truncate(big)
    assert out.endswith("B" * 200)      # tail preserved
    assert "truncated" in out
    assert len(out) <= L._MAX_LOG_CHARS + 100


def test_upload_run_log_records_for_user(tmp_path, monkeypatch):
    from backend.spec_worker import worker as W
    from backend.service import spec_generation_logs as L
    logp = tmp_path / "run.log"
    logp.write_text("hello log\n[done] success=True\n")
    calls = []
    monkeypatch.setattr("backend.supabase_env.configured", lambda: True)
    monkeypatch.setattr(L, "record", lambda **kw: calls.append(kw) or True)

    W._upload_run_log("meta/Foo", {"user_id": "u1"}, "u1|meta/Foo", str(logp), True)
    assert len(calls) == 1
    assert calls[0]["user_id"] == "u1"
    assert calls[0]["run_key"] == "u1|meta/Foo"
    assert calls[0]["success"] is True
    assert "hello log" in calls[0]["log"]
    assert calls[0]["record_id"] is None


def test_upload_run_log_names_the_record_the_run_wrote(tmp_path, monkeypatch):
    from backend.spec_worker import worker as W
    from backend.service import spec_generation_logs as L
    logp = tmp_path / "run.log"
    logp.write_text("ok\n")
    calls = []
    monkeypatch.setattr("backend.supabase_env.configured", lambda: True)
    monkeypatch.setattr(L, "record", lambda **kw: calls.append(kw) or True)
    wrote = {"user_id": "u1",
             "terminal": {"record": {"outcome": "recorded", "id": "rec-7"}}}
    W._upload_run_log("meta/Foo", wrote, "u1|meta/Foo", str(logp), True)
    stale = {"user_id": "u1", "terminal": {"record": {"outcome": "stale"}}}
    W._upload_run_log("meta/Foo", stale, "u1|meta/Foo", str(logp), True)
    assert [c["record_id"] for c in calls] == ["rec-7", None]


def test_upload_run_log_records_failure(tmp_path, monkeypatch):
    from backend.spec_worker import worker as W
    from backend.service import spec_generation_logs as L
    logp = tmp_path / "run.log"
    logp.write_text("[verdict] DOWNGRADE\n")
    calls = []
    monkeypatch.setattr("backend.supabase_env.configured", lambda: True)
    monkeypatch.setattr(L, "record", lambda **kw: calls.append(kw) or True)
    W._upload_run_log("meta/Foo", {"user_id": "u1"}, "u1|meta/Foo", str(logp), False)
    assert calls[0]["success"] is False
    assert "DOWNGRADE" in calls[0]["log"]


def test_log_request_retries_once_on_timeout(monkeypatch):
    import httpx
    from backend.service import spec_generation_logs as L
    calls = []

    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **k):
            calls.append("post")
            if len(calls) == 1:
                raise httpx.ReadTimeout("timed out")

            class R:
                status_code = 201
                text = ""
            return R()

    monkeypatch.setattr(L.httpx, "Client", FakeClient)
    r = L._log_request("post", "https://example.invalid")
    assert r.status_code == 201
    assert calls == ["post", "post"]


def test_upload_run_log_skips_anonymous(monkeypatch):
    from backend.spec_worker import worker as W
    from backend.service import spec_generation_logs as L
    calls = []
    monkeypatch.setattr(L, "record", lambda **kw: calls.append(kw))
    # No user_id → returns before importing/reading anything.
    W._upload_run_log("meta/Foo", {}, "meta/Foo", "/nonexistent", None)
    assert calls == []


def test_upload_run_log_keeps_sidecar_when_record_fails(tmp_path, monkeypatch):
    from backend.spec_worker import runtime
    from backend.spec_worker import worker as W
    from backend.service import spec_generation_logs as L
    logp = tmp_path / "run.log"
    logp.write_text("partial\n")
    monkeypatch.setattr(runtime, "_LOG_DIR", str(tmp_path))
    runtime.bind_job_log("u|m", str(logp), user_id="u", model_id="m")
    sidecar = tmp_path / (logp.name + runtime._LOG_SIDECAR_SUFFIX)
    monkeypatch.setattr("backend.supabase_env.configured", lambda: True)
    monkeypatch.setattr(L, "record", lambda **kw: False)
    W._upload_run_log("m", {"user_id": "u"}, "u|m", str(logp), False)
    assert sidecar.is_file()


def test_upload_run_log_none_success_is_false(tmp_path, monkeypatch):
    from backend.spec_worker import worker as W
    from backend.service import spec_generation_logs as L
    logp = tmp_path / "run.log"
    logp.write_text("aborted\n")
    calls = []
    monkeypatch.setattr("backend.supabase_env.configured", lambda: True)
    monkeypatch.setattr(L, "record", lambda **kw: calls.append(kw) or True)
    W._upload_run_log("meta/Foo", {"user_id": "u1"}, "u1|meta/Foo", str(logp), None)
    assert calls[0]["success"] is False


def test_record_patches_the_row_its_run_began(httpx_mock, monkeypatch):
    """record() updates the begin_run row bound to this job thread, by id;
    it neither looks a row up by run_key nor INSERTs a parallel one."""
    import re
    from backend.service import spec_generation_logs as L
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")
    url = re.compile(r"https://example\.supabase\.co/rest/v1/spec_generation_logs")
    httpx_mock.add_response(method="PATCH", url=url, json=[])
    L.bind_run_row("row-1")
    try:
        assert L.record(user_id="u", model_id="org/m", log="transcript",
                        run_key="u|org/m", success=True)
    finally:
        L.bind_run_row(None)
    reqs = httpx_mock.get_requests()
    assert [r.method for r in reqs] == ["PATCH"]
    patch = reqs[0]
    assert patch.url.params["id"] == "eq.row-1"
    import json as _json
    body = _json.loads(patch.content)
    assert "success" not in body
    assert body["log"] == "transcript"
    assert not any(r.method == "POST" for r in reqs)


def test_record_inserts_kind_for_fix_run_key(httpx_mock, monkeypatch):
    """No begin_run row → INSERT still stamps kind=fix from the run_key
    prefix so the account page can attach the transcript."""
    import json as _json
    import re
    from backend.service import spec_generation_logs as L
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")
    url = re.compile(r"https://example\.supabase\.co/rest/v1/spec_generation_logs")
    httpx_mock.add_response(method="POST", url=url, json=[])
    assert L.record(user_id="u", model_id="org/m", log="fix transcript",
                    run_key="fix|u|rec-1", success=True)
    post = next(r for r in httpx_mock.get_requests() if r.method == "POST")
    body = _json.loads(post.content)
    assert body["kind"] == "fix"
    assert body["run_key"] == "fix|u|rec-1"


def test_latest_by_run_key_asks_only_for_state_rows(httpx_mock, monkeypatch):
    import re
    from backend.service import spec_generation_logs as L
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")
    url = re.compile(r"https://example\.supabase\.co/rest/v1/spec_generation_logs")
    httpx_mock.add_response(
        method="GET", url=url,
        json=[{"id": "row-1", "state": "interrupted", "run_key": "u|org/m"}],
    )
    row = L.latest_by_run_key("u|org/m")
    assert row["state"] == "interrupted"
    req = httpx_mock.get_requests()[0]
    assert "state.eq.running" in str(req.url)
    assert "state.eq.interrupted" in str(req.url)
