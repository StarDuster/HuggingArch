"""Storage module for agent review reports (service-role, owner-scoped reads)."""
from __future__ import annotations

import re

from backend.service import spec_generation_reviews as R
from .conftest import TEST_SUPABASE_URL

_REVIEWS = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generation_reviews") + r"(\?.*)?")
_GENS = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/spec_generations") + r"(\?.*)?")


def test_record_returns_the_new_row_id(httpx_mock):
    httpx_mock.add_response(method="POST", url=_REVIEWS, status_code=201,
                            json=[{"id": "rev-9"}])
    assert R.record(record_id="r1", model_id="m", reviewer_id="u1",
                    reviewer_role="user", verdict="pass", report="ok") == "rev-9"
    sent = httpx_mock.get_requests()[0]
    assert sent.headers["Prefer"] == "return=representation"
    # the INSERT never names `admission`: a review lands before migration 0038
    assert "admission" not in sent.content.decode()


def test_record_names_its_run_log_and_lands_before_the_column_does(httpx_mock):
    import json
    from backend.service import spec_generation_logs
    httpx_mock.add_response(method="POST", url=_REVIEWS, status_code=400,
                            text='{"code":"PGRST204","message":"Could not find the '
                                 "'log_id' column of 'spec_generation_reviews' in the schema cache\"}")
    httpx_mock.add_response(method="POST", url=_REVIEWS, status_code=201, json=[{"id": "rev-9"}])
    # the run log is the durability row bound to this job thread
    spec_generation_logs.bind_run_row("log-1")
    try:
        assert R.record(record_id="r1", model_id="m", reviewer_id="u1", reviewer_role="user",
                        verdict="pass", report="ok") == "rev-9"
    finally:
        spec_generation_logs.bind_run_row(None)
    first, second = (json.loads(q.content) for q in httpx_mock.get_requests())
    assert first["log_id"] == "log-1" and "log_id" not in second


def test_verdicts_by_log_reads_each_reviews_run_log(httpx_mock):
    httpx_mock.add_response(method="GET", url=_REVIEWS, json=[
        {"log_id": "log-1", "verdict": "error"}, {"log_id": "log-2", "verdict": "pass"}])
    assert R.verdicts_by_log(["r1"]) == {"log-1": "error", "log-2": "pass"}


def test_verdicts_by_log_is_empty_before_the_column(httpx_mock):
    httpx_mock.add_response(method="GET", url=_REVIEWS, status_code=400,
                            text='{"code":"42703","message":"column spec_generation_reviews.log_id does not exist"}')
    assert R.verdicts_by_log(["r1"]) == {}


def test_record_silent_fail_on_http_error(httpx_mock):
    httpx_mock.add_response(method="POST", url=_REVIEWS, status_code=400, text="bad")
    assert R.record(record_id="r1", model_id="m", reviewer_id="u1",
                    reviewer_role="user", verdict="pass", report="ok") is None


def test_list_returns_empty_for_non_owner(httpx_mock):
    # ownership probe returns no row → caller doesn't own it → []
    httpx_mock.add_response(method="GET", url=_GENS, json=[])
    assert R.list_for_record_owner("r1", "u1") == []


def test_list_returns_reviews_for_owner(httpx_mock):
    httpx_mock.add_response(method="GET", url=_GENS, json=[{"id": "r1"}])
    httpx_mock.add_response(method="GET", url=_REVIEWS, json=[
        {"id": "rev1", "verdict": "pass", "report": "good", "reviewer_role": "admin"},
    ])
    out = R.list_for_record_owner("r1", "u1")
    assert len(out) == 1 and out[0]["verdict"] == "pass"
