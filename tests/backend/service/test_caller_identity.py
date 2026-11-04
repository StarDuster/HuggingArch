"""A verified token without a `sub` is refused with 401, and each route keeps
its own detail text."""
from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_uid_is_the_sub():
    from backend.service import auth
    assert auth.uid({"sub": "u-1"}) == "u-1"


@pytest.mark.parametrize("user", [{}, {"sub": ""}, {"sub": None}])
def test_no_sub_is_401(user):
    from backend.service import auth
    with pytest.raises(HTTPException) as e:
        auth.uid(user)
    assert (e.value.status_code, e.value.detail) == (401, "Missing user identity")


@pytest.mark.parametrize("route", ["service_review_status", "service_fix_status"])
def test_review_and_fix_polls_keep_their_detail(route):
    from backend.service.routes import spec
    with pytest.raises(HTTPException) as e:
        getattr(spec, route)(record_id="r", cursor=0, user={})
    assert (e.value.status_code, e.value.detail) == (401, "No user_id in token")


@pytest.mark.parametrize("route, kwargs", [
    ("bench_list", {}),
    ("bench_delete_run", {"run_id": "r"}),
    ("bench_delete", {"upload_id": "x"}),
    ("bench_contribute", {"upload_id": "x"}),
])
def test_bench_route_detail(route, kwargs):
    from backend.service.bench import routes as bench
    with pytest.raises(HTTPException) as e:
        getattr(bench, route)(user={}, **kwargs)
    assert (e.value.status_code, e.value.detail) == (401, "Missing user identity")
