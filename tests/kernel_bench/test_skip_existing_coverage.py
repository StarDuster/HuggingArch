"""`--skip-existing` asks Kernel Bench which cells are measured, a batch of
keys at a time (`driver._kernel_bench_coverage`). One batch the service does
not answer used to turn the skip off for the whole run; now that batch is
retried, and if it still fails only its keys go unasked — counted as not
measured, never as measured — while every other batch's answer holds.

The service here is a stand-in behind the real client
(`backend.measure.lookup`, through an httpx MockTransport), so a timeout is
the transport's own `ReadTimeout`, as on the H100 box.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.measure import lookup
from kernel_bench import driver

IMPL = ("torch", "2.11", "torch_gemm")


def _key(i: int) -> lookup.Key:
    return lookup.Key(signature="gemm", shape_json=json.dumps({"d_out": i}),
                      dtype_env_json="{}", gpu="H100-SXM", phase="decode")


class _Service:
    """Answers `/internal/bench/version` and `/internal/bench/lookup` (every
    key covered by `IMPL`); a lookup naming a key in `times_out` raises
    `ReadTimeout` the first `fails` times it is asked (forever when None)."""

    def __init__(self, times_out=(), fails=None):
        self.times_out = {_key(i).shape_json for i in times_out}
        self.fails = fails
        self.lookups: list = []
        self.timeouts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/bench/version":
            return httpx.Response(200, json={
                "lake_id": "lake-1", "snapshot": 9, "shared_snapshot": 4,
                "rules_rev": "r", "snapshot_content": "t9",
                "shared_snapshot_content": "t4"})
        keys = json.loads(request.content)["keys"]
        self.lookups.append([k["physical_shape"] for k in keys])
        if any(k["physical_shape"] in self.times_out for k in keys) and (
                self.fails is None or self.timeouts < self.fails):
            self.timeouts += 1
            raise httpx.ReadTimeout("timed out", request=request)
        columns = ["framework", "framework_version", "kernel_impl", "origin"]
        return httpx.Response(200, json={
            "lake_id": "lake-1", "snapshot": 9, "columns": columns,
            "values": [*IMPL, "shared"], "excluded_mine": 0,
            "rows": [[[0, 1, 2, 3]] for _ in keys]})


@pytest.fixture
def ask(monkeypatch):
    """Run the coverage question over keys 1..n in batches of 2 against a
    `_Service`: `(covered, result, service)`."""
    monkeypatch.setenv("HUGGINGARCH_BENCH_URL", "http://bench.internal")
    monkeypatch.setenv("HUGGINGARCH_BENCH_SECRET", "s")
    monkeypatch.delenv("HUGGINGARCH_BENCH_OWNER", raising=False)
    monkeypatch.setattr(driver, "COVERAGE_BATCH", 2)
    # raising=False: so the first case reads red, not an error, on a driver
    # that predates retries.
    monkeypatch.setattr(driver, "COVERAGE_BACKOFF_S", 0.0, raising=False)
    # A job is its key: the plan's key is not what is under test.
    monkeypatch.setattr(driver.Invocation, "from_job", staticmethod(lambda job: job))
    monkeypatch.setattr(driver, "physical_key_for", lambda inv, env: _key(inv))
    lookup.forget_rows()

    def run(n: int, service: _Service):
        monkeypatch.setattr(lookup, "_clients", {
            os.getpid(): httpx.Client(transport=httpx.MockTransport(service))})
        result = driver.RunResult()
        got = driver._kernel_bench_coverage(list(range(1, n + 1)), None, result,
                                            lambda m: None)
        return got, result, service
    yield run
    lookup.forget_rows()


def test_a_batch_that_never_answers_leaves_the_other_batches_covered(ask):
    """The H100 case: one batch times out on every attempt. Its keys are
    reported unanswered and left out (measured again); the other batches'
    coverage still skips their cells."""
    got, result, service = ask(5, _Service(times_out=[3]))
    assert set(got) == {_key(1), _key(2), _key(5)}
    assert all(got[k] == {IMPL} for k in got)
    assert _key(3) not in got and _key(4) not in got
    assert service.timeouts == driver.COVERAGE_ATTEMPTS
    (msg,) = result.failures
    assert "PARTIAL" in msg and "2 of 5 keys unanswered" in msg, msg
    assert "1 of 3 batches" in msg and "ReadTimeout" in msg, msg


def test_a_batch_that_times_out_once_is_asked_again(ask):
    got, result, service = ask(5, _Service(times_out=[3], fails=1))
    assert set(got) == {_key(i) for i in range(1, 6)}
    assert result.failures == []
    assert service.timeouts == 1


def test_a_service_that_never_answers_stops_after_two_batches(ask):
    """Down, not busy: two batches in a row exhaust their attempts and the
    rest are not asked — every key counted as not measured, said as such."""
    got, result, service = ask(7, _Service(times_out=range(1, 8)))
    assert got == {}
    assert len(service.lookups) == 2 * driver.COVERAGE_ATTEMPTS
    (msg,) = result.failures
    assert "INACTIVE" in msg and "7 of 7 keys unanswered" in msg, msg
    assert "2 more not asked" in msg, msg


def test_a_failure_between_answers_does_not_stop_the_questions(ask):
    """Only consecutive failures stop the asking: a failed batch followed by
    an answered one resets the count."""
    got, result, _ = ask(7, _Service(times_out=[1, 5]))
    assert set(got) == {_key(3), _key(4), _key(7)}
    (msg,) = result.failures
    assert "4 of 7 keys unanswered" in msg and "2 of 4 batches" in msg, msg
