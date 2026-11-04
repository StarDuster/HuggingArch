"""In-place spec update for the agent-fix flow (service-role PATCH)."""
from __future__ import annotations

import re

import pytest

from backend.service import spec_generations as G
from .conftest import TEST_SUPABASE_URL

_GENS = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generations") + r"(\?.*)?")


def test_update_spec_ok(httpx_mock):
    # return=representation → a non-empty list means a row matched.
    httpx_mock.add_response(method="PATCH", url=_GENS, json=[{"id": "r1"}])
    assert G.update_spec("r1", spec_yaml="match: {}", new_component_yamls=[]) is True


def test_update_spec_carries_a_snapshot_only_when_given(httpx_mock):
    import json
    httpx_mock.add_response(method="PATCH", url=_GENS, json=[{"id": "r1"}])
    httpx_mock.add_response(method="PATCH", url=_GENS, json=[{"id": "r1"}])
    G.update_spec("r1", spec_yaml="x", new_component_yamls=[])
    G.update_spec("r1", spec_yaml="x", new_component_yamls=[], snapshot_json={"config": {}})
    first, second = (json.loads(r.content) for r in httpx_mock.get_requests())
    assert "snapshot_json" not in first
    assert second["snapshot_json"] == {"config": {}}


def test_update_spec_no_match_is_false(httpx_mock):
    # wrong owner / missing record → empty representation → False, not a silent ok
    httpx_mock.add_response(method="PATCH", url=_GENS, json=[])
    assert G.update_spec("r1", spec_yaml="x", new_component_yamls=[],
                         user_id="not-owner") is False


def test_update_spec_http_error_is_false(httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_GENS, status_code=400, text="bad")
    assert G.update_spec("r1", spec_yaml="x", new_component_yamls=[]) is False


_RPC = f"{TEST_SUPABASE_URL}/rest/v1/rpc/"


def _body(req):
    return __import__("json").loads(req.content)


def test_record_is_one_conditional_write(httpx_mock):
    """The success write decides in SQL, under the (user, model) lock: no
    read-then-patch from here."""
    httpx_mock.add_response(method="POST", url=_RPC + "record_generation",
                            json={"outcome": "recorded", "id": "r1"})
    out = G.record(model_id="o/m", user_id="u", spec_yaml="new-y",
                   snapshot_json={"k": 1}, agent_cli="claude",
                   run_started_at="2026-10-01T00:00:00+00:00")
    assert out == {"outcome": "recorded", "id": "r1"}
    (req,) = httpx_mock.get_requests()
    assert req.method == "POST"
    body = _body(req)
    assert body["p_spec_yaml"] == "new-y" and body["p_snapshot_json"] == {"k": 1}
    assert body["p_run_started_at"] == "2026-10-01T00:00:00+00:00"


def test_record_storage_failure_is_the_failed_outcome(httpx_mock):
    httpx_mock.add_response(method="POST", url=_RPC + "record_generation",
                            status_code=503, text="down")
    out = G.record(model_id="o/m", user_id="u", spec_yaml="y")
    assert out["outcome"] == "failed" and "503" in out["reason"]


def test_record_without_a_user_keeps_none(httpx_mock):
    assert G.record(model_id="o/m", user_id="", spec_yaml="y") == {"outcome": "none"}
    assert httpx_mock.get_requests() == []


def test_record_pending_passes_the_gaps_and_engine_version(httpx_mock):
    httpx_mock.add_response(method="POST", url=_RPC + "record_pending_generation",
                            json={"outcome": "stale"})
    gaps = [{"capability": "c", "deployments": ["tp8"], "stages": ["decode"]}]
    assert G.record_pending(model_id="o/m", user_id="u", spec_yaml="y",
                            engine_gaps=gaps, engine_version="ev",
                            run_started_at="t") == {"outcome": "stale"}
    body = _body(httpx_mock.get_requests()[0])
    assert body["p_engine_gaps"] == gaps and body["p_engine_version"] == "ev"


def test_transition_and_promote_name_the_expected_revision(httpx_mock):
    from backend.spec_worker.outcome import GenerationStatus
    httpx_mock.add_response(method="POST", url=_RPC + "transition_pending_generation",
                            json={"outcome": "conflict"})
    httpx_mock.add_response(method="POST", url=_RPC + "transition_pending_generation",
                            json={"outcome": "recorded", "id": "r1"})
    assert G.transition_pending("r1", expected_revision=4,
                                to=GenerationStatus.NEEDS_REPAIR,
                                repairable_errors=["e"])["outcome"] == "conflict"
    assert G.promote_pending("r1", expected_revision=5, spec_yaml="fixed")["outcome"] == "recorded"
    first, second = (_body(r) for r in httpx_mock.get_requests())
    assert (first["p_to"], first["p_expected_revision"], first["p_repairable_errors"]) \
        == ("needs_repair", 4, ["e"])
    assert (second["p_to"], second["p_expected_revision"], second["p_spec_yaml"]) \
        == ("promote", 5, "fixed")


def test_an_rpc_answer_without_an_outcome_is_a_failure(httpx_mock):
    httpx_mock.add_response(method="POST", url=_RPC + "record_generation", json=[])
    assert G.record(model_id="o/m", user_id="u", spec_yaml="y")["outcome"] == "failed"


@pytest.mark.parametrize("rows,served", [
    # a valid record wins over a newer blocked one
    ([{"id": "b", "status": "blocked_by_engine"}, {"id": "g", "status": "closed"}], "g"),
    # with no valid record the blocked one is served
    ([{"id": "r", "status": "needs_repair"}, {"id": "b", "status": "blocked_by_engine"}], "b"),
    # a record that needs repair (or was rejected) is still the user's spec of
    # the model: returned, so the caller knows it is no missing spec, but not
    # among the records it analyses or seeds from
    ([{"id": "r", "status": "needs_repair"}, {"id": "x", "status": "rejected"}], "r"),
    ([], None),
])
def test_serving_generation_prefers_a_valid_record(httpx_mock, rows, served):
    httpx_mock.add_response(method="GET", url=_GENS, json=rows)
    row = G.serving_generation("u", "o/m")
    assert (row or {}).get("id") == served
    assert "status" not in httpx_mock.get_requests()[0].url.params
    if row is not None and row["status"] in ("needs_repair", "rejected"):
        assert row["status"] not in G.SERVED


@pytest.mark.parametrize("answer", [
    {"status_code": 503, "text": "down"},
    {"status_code": 401, "text": "JWT expired"},
])
def test_a_failed_read_of_the_records_is_an_outage_not_no_record(httpx_mock, answer):
    httpx_mock.add_response(method="GET", url=_GENS, **answer)
    with pytest.raises(G.SpecGenerationsUnavailable):
        G.serving_generation("u", "o/m")


def test_a_strict_read_refused_with_401_is_an_outage_not_a_missing_record(httpx_mock):
    httpx_mock.add_response(method="GET", url=_GENS, status_code=401, text="JWT expired")
    with pytest.raises(G.SpecGenerationsUnavailable):
        G.get("rec-1", strict=True)


def test_update_spec_owner_scope_in_params(httpx_mock):
    httpx_mock.add_response(method="PATCH", url=_GENS, json=[{"id": "r1"}])
    G.update_spec("r1", spec_yaml="x", new_component_yamls=[{"path": "p", "body": "b"}],
                  user_id="owner-1")
    req = httpx_mock.get_requests()[-1]
    assert "user_id=eq.owner-1" in str(req.url)
    assert "id=eq.r1" in str(req.url)
