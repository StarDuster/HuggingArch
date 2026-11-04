"""Admin-only is a dependency, not a spelling inside the handler.

`bench.py` used to say it two ways — `Depends(require_admin)` on some routes,
`if not is_admin(...): 403` in the body of others — and the protected set was
whatever a hand-written list of paths in a test happened to name. Now every
admin-only route declares `require_admin`, and this gate enumerates the app's
routes for that dependency: the set is read off the router, so a route that
forgets the dependency is not in the set and a body check cannot stand in.
"""
from __future__ import annotations

import pytest

from backend.service import auth as _auth
from .conftest import mock_admins_no


def _depends_on(route, fn) -> bool:
    dependant = getattr(route, "dependant", None)
    stack = [dependant] if dependant is not None else []
    while stack:
        d = stack.pop()
        if d.call is fn:
            return True
        stack.extend(d.dependencies)
    return False


def _admin_routes(app, require_admin=_auth.require_admin):
    return sorted(
        (sorted(route.methods - {"HEAD", "OPTIONS"})[0], route.path)
        for route in app.routes
        if getattr(route, "methods", None) and _depends_on(route, require_admin)
    )


def test_the_bench_curation_routes_are_declared_admin(bench_app):
    """Served by the Kernel Bench service, whose admin is the bit the HA
    API forwards (`identity.require_admin`)."""
    from backend.service.bench import identity
    declared = set(_admin_routes(bench_app, identity.require_admin))
    for wanted in (("POST", "/api/bench/promote"),
                   ("GET", "/api/bench/nominations"),
                   ("POST", "/api/bench/reject")):
        assert wanted in declared, (wanted, declared)


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/bench/promote"),
    ("GET", "/api/bench/nominations"),
    ("POST", "/api/bench/reject"),
])
def test_every_declared_admin_route_refuses_anonymous_and_non_admin(
        client, bench_proxied, user_token, httpx_mock, method, path):
    anon = client.request(method, path, json={} if method == "POST" else None)
    assert anon.status_code == 401, anon.text
    mock_admins_no(httpx_mock, "user-uuid-1")
    user = client.request(method, path, json={"sets": []} if method == "POST" else None,
                          headers={"Authorization": f"Bearer {user_token}"})
    assert user.status_code == 403, user.text
