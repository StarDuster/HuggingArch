"""Cache headers: hashed /web-assets are immutable, HTML is no-store."""
import pathlib

import pytest
from fastapi.testclient import TestClient

from backend.api import app


def test_hashed_web_assets_are_immutable():
    dist = pathlib.Path("web/dist/web-assets")
    if not dist.is_dir():
        pytest.skip("web/dist missing; run npm run build --prefix web")
    hashed = next(p.name for p in sorted(dist.glob("*.js")) if "-" in p.stem)
    r = TestClient(app).get(f"/web-assets/{hashed}")
    assert r.status_code == 200
    assert "immutable" in r.headers["cache-control"]
    assert "max-age=31536000" in r.headers["cache-control"]


def test_html_stays_no_store():
    r = TestClient(app).get("/inspector.html")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert "no-store" in r.headers["cache-control"]


def test_assets_built_after_the_app_are_served(tmp_path):
    """The asset mount reads its directory per request: before `web/dist` is
    built a file is a 404 (not a 500), and once it is built the same app
    serves it — CI builds it in the test run, after the app is imported."""
    from starlette.applications import Starlette
    from backend.api._app import _BuiltLaterStaticFiles

    assets = tmp_path / "web-assets"
    sub = Starlette()
    sub.mount("/web-assets", _BuiltLaterStaticFiles(directory=str(assets), check_dir=False))
    client = TestClient(sub)
    assert client.get("/web-assets/app-abcdefgh.js").status_code == 404
    assets.mkdir()
    (assets / "app-abcdefgh.js").write_text("export {};\n")
    assert client.get("/web-assets/app-abcdefgh.js").status_code == 200
