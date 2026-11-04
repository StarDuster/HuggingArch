"""Why an agent invocation ended, as each CLI reports it.

The strings are the ones DeepSeek-V4.1-Flash's generation runs recorded
(logs/spec_worker/*V4.1-Flash*.log, 2026-10-02..04): of 16 runs, the 13 that
did not finish ended on the infrastructure — a model service refusing the
request, a timer, a kill — not on the spec. Each backend recognizes its own
CLI's report (`AgentBackend.unavailable_errors`); the runner puts the
classification on the terminal `done` as `agent_end`.
"""
from __future__ import annotations

import io
import json
import threading

import pytest

from backend._agent_runner import runner as R
from backend.agents import get_backend
from backend.agents.base import AgentEnd, service_status, signalled
from backend.agents.claude import ClaudeBackend


CLAUDE_524 = (
    'API Error: 524 {"type":"https://developers.cloudflare.com/support/'
    'troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/",'
    '"title":"Error 524: A timeout occurred","status":524,"detail":"The')
CODEX_401 = (
    "unexpected status 401 Unauthorized: Missing bearer or basic "
    "authentication in header, url: https://api.openai.com/v1/responses, "
    "cf-ray: a44906214c8a70f0-SJC, request id: req_780068826ee74c03b959e70ca62a425f")
CODEX_WS_401 = (
    "2026-10-03T03:58:35.435497Z ERROR codex_api::endpoint::responses_websocket: "
    "failed to connect to websocket: HTTP error: 401 Unauthorized, "
    "url: wss://api.openai.com/v1/responses")
KIMI_401 = ("error: failed to run prompt: provider.auth_error: 401 Invalid "
            "Authentication\nSee log: /home/u/.kimi-code/logs/kimi-code.log")
KIMI_WEEKLY = (
    "error: failed to run prompt: provider.auth_error: 403 You've reached "
    "your weekly (7-day) usage limit. Your quota will reset when the current "
    "7-day window ends.")
CLAUDE_EXHAUSTED = (
    "Prompt is too long · automatic compaction failed: " + CLAUDE_524)
CODEX_EXHAUSTED = (
    "Codex ran out of room in the model's context window. Start a new "
    "thread or clear earlier history before retrying.")
KIMI_EXHAUSTED = (
    "error: failed to run prompt: context.overflow: Compaction failed to "
    "bring the context under the model window after 3 attempts.")
KIMI_NO_MODEL = (
    "error: failed to run prompt: No model configured. Run `kimi` and use "
    "/login to sign in, then retry; or set default_model in config.toml.")


def _end(cli: str, *, returncode=1, error_text="", timed_out=False) -> str:
    return get_backend(cli).classify_end(
        returncode=returncode, error_text=error_text, timed_out=timed_out).value


@pytest.mark.parametrize("cli, returncode, text, expected", [
    # Claude Code: the `is_error` result quotes the API's answer.
    ("claude", None, CLAUDE_524, "agent_unavailable"),
    ("claude", None, 'API Error: 401 {"type":"error","error":{"type":'
                     '"authentication_error"}}', "agent_unavailable"),
    ("claude", None, "Invalid API key · Please run /login", "agent_unavailable"),
    ("claude", None, "API Error: Connection error.", "agent_unavailable"),
    ("claude", None, 'API Error: 400 {"type":"error","error":{"type":'
                     '"invalid_request_error","message":"prompt is too long"}}',
     "context_exhausted"),
    # A full session whose compaction met a 524 is full, not unavailable.
    ("claude", None, CLAUDE_EXHAUSTED, "context_exhausted"),
    ("claude", -9, "", "killed"),
    ("claude", 143, "", "killed"),
    # Codex: `turn.failed` and the websocket transport's stderr.
    ("codex", None, CODEX_401, "agent_unavailable"),
    # Killed while it retried a refused request: the refusal explains it.
    ("codex", -9, CODEX_WS_401, "agent_unavailable"),
    ("codex", 1, "exceeded retry limit, last status: 429 Too Many Requests",
     "agent_unavailable"),
    ("codex", 1, "unexpected status 400 Bad Request: unknown parameter",
     "failed"),
    ("codex", None, CODEX_EXHAUSTED, "context_exhausted"),
    # Kimi Code: `error: failed to run prompt: …` on stderr, exit 1.
    ("kimi", 1, KIMI_401, "agent_unavailable"),
    ("kimi", 1, KIMI_WEEKLY, "agent_unavailable"),
    ("kimi", 1, KIMI_NO_MODEL, "agent_unavailable"),
    ("kimi", 1, "error: failed to run prompt: provider.api_error: 400 "
                "context length exceeded", "context_exhausted"),
    ("kimi", 1, KIMI_EXHAUSTED, "context_exhausted"),
    ("kimi", 143, ".mlp.experts`, `self.mlp.shared_expert`. Each named "
                  "submodule should", "killed"),
    ("kimi", 1, "Traceback (most recent call last): ValueError", "failed"),
])
def test_each_backend_classifies_its_own_cli_report(cli, returncode, text, expected):
    assert _end(cli, returncode=returncode, error_text=text) == expected


@pytest.mark.parametrize("cli", ["claude", "codex", "kimi"])
def test_the_runs_own_timer_wins_over_what_the_cli_printed(cli):
    assert _end(cli, returncode=-9, error_text=CODEX_WS_401 + KIMI_401
                + CLAUDE_524, timed_out=True) == "timed_out"


def test_a_backend_does_not_read_another_clis_dialect():
    """Recognition is each CLI's own knowledge: Kimi's provider error means
    nothing to Claude's parser, and a bare status in prose is no report."""
    assert _end("claude", error_text=KIMI_401) == "failed"
    assert _end("kimi", error_text=CLAUDE_524) == "failed"
    assert _end("codex", error_text="the checkpoint has 401 tensors") == "failed"


def test_service_status_and_signals():
    assert all(service_status(s) for s in (401, 402, 403, 408, 429, 500, 502, 524))
    assert not any(service_status(s) for s in (200, 400, 404, 422))
    assert signalled(-9) and signalled(137) and signalled(143)
    assert not signalled(0) and not signalled(1) and not signalled(None)


# --- the runner puts the classification on `done` ----------------------------


class _Proc:
    def __init__(self, lines, returncode, stderr=""):
        self._lines = list(lines)
        self.stderr = io.StringIO(stderr)
        self.pid = 1
        self.returncode = returncode

    @property
    def stdout(self):
        return self

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


class _Claude(ClaudeBackend):
    """Claude Code's own parsing and classification, a scripted process."""

    def __init__(self, proc=None, error=None):
        self._proc, self._error = proc, error

    def spawn(self, prompt, env=None, workspace=None):
        if self._error is not None:
            raise self._error
        return self._proc


def _done(monkeypatch, tmp_path, backend) -> dict:
    monkeypatch.setattr(R, "get_backend", lambda cli: backend)
    frames = list(R.stream_agent_core(
        key="m", prep_result={"prompt": "p"}, agent_cli="claude",
        log_path=str(tmp_path / "log.txt"), log_header="h", launch_banner="B",
        procs_dict={}, locks_lock=threading.Lock(), release_fn=lambda k: None,
        on_success=lambda *a: None))
    events = [json.loads(f[len("data: "):]) for f in frames]
    return [e for e in events if e["type"] == "done"][-1]


def test_claudes_error_result_is_an_unavailable_agent(monkeypatch, tmp_path):
    result = {"type": "result", "subtype": "success", "is_error": True,
              "result": CLAUDE_524}
    done = _done(monkeypatch, tmp_path,
                 _Claude(_Proc([json.dumps(result) + "\n"], 0)))
    assert done["success"] is False
    assert done["agent_end"] == AgentEnd.AGENT_UNAVAILABLE.value


def test_claudes_full_session_is_context_exhausted(monkeypatch, tmp_path):
    result = {"type": "result", "subtype": "success", "is_error": True,
              "result": CLAUDE_EXHAUSTED}
    done = _done(monkeypatch, tmp_path,
                 _Claude(_Proc([json.dumps(result) + "\n"], 1)))
    assert done["agent_end"] == AgentEnd.CONTEXT_EXHAUSTED.value
    assert not AgentEnd(done["agent_end"]).reenterable
    assert all(end.reenterable for end in AgentEnd
               if end is not AgentEnd.CONTEXT_EXHAUSTED)


def test_a_nonzero_exit_is_classified_from_stderr(monkeypatch, tmp_path):
    done = _done(monkeypatch, tmp_path, _Claude(_Proc([], -9, stderr="")))
    assert done["agent_end"] == "killed"


def test_a_cli_that_cannot_start_is_unavailable(monkeypatch, tmp_path):
    done = _done(monkeypatch, tmp_path, _Claude(
        error=FileNotFoundError(2, "No such file or directory", "claude")))
    assert done["agent_end"] == "agent_unavailable"


def test_a_finished_turn_carries_no_agent_end(monkeypatch, tmp_path):
    result = {"type": "result", "subtype": "success", "is_error": False}
    done = _done(monkeypatch, tmp_path,
                 _Claude(_Proc([json.dumps(result) + "\n"], 0)))
    assert done["success"] is True and "agent_end" not in done
