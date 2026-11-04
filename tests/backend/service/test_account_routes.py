"""/api/account/avatar — multipart upload + delete.

Validates input (MIME, size), proxies to Storage with service-role,
patches profiles.avatar_url, cleans up older objects."""
from __future__ import annotations

import re

from .conftest import TEST_SUPABASE_URL


# pytest-httpx accepts re.Pattern for prefix-style URL matching.
_AVATAR_UPLOAD_RE = re.compile(
    re.escape(f"{TEST_SUPABASE_URL}/storage/v1/object/avatars/") + r".+"
)
_PROFILES_PATCH_RE = re.compile(
    re.escape(f"{TEST_SUPABASE_URL}/rest/v1/profiles") + r"(\?.*)?"
)


def _stage_storage_mocks(httpx_mock, *, list_existing=None, expect_upload=True):
    """Plant the four httpx calls a successful upload makes:
        1. POST  /storage/v1/object/list/avatars     (list existing)
        2. POST  /storage/v1/object/avatars/<path>   (upload new file)
        3. PATCH /rest/v1/profiles                   (write avatar_url)
        4. DELETE /storage/v1/object/avatars         (cleanup older)
    """
    httpx_mock.add_response(
        method="POST",
        url=f"{TEST_SUPABASE_URL}/storage/v1/object/list/avatars",
        json=list_existing or [],
        is_reusable=True,
    )
    if expect_upload:
        httpx_mock.add_response(
            method="POST", url=_AVATAR_UPLOAD_RE,
            status_code=200, json={"Key": "avatars/test"},
        )
    httpx_mock.add_response(
        method="PATCH", url=_PROFILES_PATCH_RE,
        status_code=204, text="",
    )
    httpx_mock.add_response(
        method="DELETE",
        url=f"{TEST_SUPABASE_URL}/storage/v1/object/avatars",
        status_code=200, json={},
        is_reusable=True,
        is_optional=True,  # only fires when there are older objects to clean up
    )


class TestAvatarGating:
    def test_anonymous_upload_401(self, client):
        r = client.post(
            "/api/account/avatar",
            files={"file": ("a.png", b"\x89PNG", "image/png")},
        )
        assert r.status_code == 401

    def test_anonymous_delete_401(self, client):
        r = client.delete("/api/account/avatar")
        assert r.status_code == 401


class TestAvatarUpload:
    def test_valid_png_upload_writes_avatar_url(self, client, user_token, httpx_mock):
        """Valid PNG → 200 + JSON body containing the new public URL."""
        _stage_storage_mocks(httpx_mock)
        r = client.post(
            "/api/account/avatar",
            files={"file": ("avatar.png", b"\x89PNG\r\n\x1a\n" + b"x" * 100, "image/png")},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["avatar_url"].startswith(
            f"{TEST_SUPABASE_URL}/storage/v1/object/public/avatars/user-uuid-1/"
        )
        assert body["avatar_url"].endswith(".png")

    def test_rejects_unsupported_mime(self, client, user_token):
        r = client.post(
            "/api/account/avatar",
            files={"file": ("a.bmp", b"BM" + b"x" * 50, "image/bmp")},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 415
        assert "unsupported" in r.json()["detail"].lower()

    def test_rejects_empty_upload(self, client, user_token):
        r = client.post(
            "/api/account/avatar",
            files={"file": ("a.png", b"", "image/png")},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 400

    def test_rejects_oversized(self, client, user_token):
        # > 2 MB
        big = b"\x89PNG" + b"x" * (2 * 1024 * 1024 + 1)
        r = client.post(
            "/api/account/avatar",
            files={"file": ("a.png", big, "image/png")},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()

    def test_storage_failure_returns_502(self, client, user_token, httpx_mock):
        # list ok, upload 500 → 502
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/storage/v1/object/list/avatars",
            json=[], is_reusable=True,
        )
        httpx_mock.add_response(
            method="POST", url=_AVATAR_UPLOAD_RE,
            status_code=500, text="storage down",
        )
        r = client.post(
            "/api/account/avatar",
            files={"file": ("a.png", b"\x89PNG" + b"x" * 50, "image/png")},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 502

    def test_user_id_is_namespaced(self, client, user_token, httpx_mock):
        """Upload path must be `<jwt.sub>/<ts>.<ext>`. Header forgery
        from request body must not change which dir the file lands in."""
        captured = {"upload_url": None}

        def _capture(req):
            captured["upload_url"] = str(req.url)
            from httpx import Response
            return Response(200, json={})

        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/storage/v1/object/list/avatars",
            json=[], is_reusable=True,
        )
        httpx_mock.add_callback(
            _capture, method="POST", url=_AVATAR_UPLOAD_RE,
        )
        httpx_mock.add_response(
            method="PATCH", url=_PROFILES_PATCH_RE,
            status_code=204, text="",
        )
        httpx_mock.add_response(
            method="DELETE",
            url=f"{TEST_SUPABASE_URL}/storage/v1/object/avatars",
            status_code=200, json={}, is_reusable=True, is_optional=True,
        )

        client.post(
            "/api/account/avatar",
            files={"file": ("a.png", b"\x89PNG" + b"x" * 30, "image/png")},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        # JWT subject is user-uuid-1; that prefix must appear in storage path.
        assert captured["upload_url"] is not None
        assert "/avatars/user-uuid-1/" in captured["upload_url"]


class TestAvatarDelete:
    def test_clears_avatar_and_storage(self, client, user_token, httpx_mock):
        # PATCH profiles → null
        captured = {}
        def _capture_patch(req):
            import json
            captured["patch"] = json.loads(req.content)
            from httpx import Response
            return Response(204)
        httpx_mock.add_callback(
            _capture_patch, method="PATCH", url=_PROFILES_PATCH_RE,
        )
        # List existing avatars
        httpx_mock.add_response(
            method="POST",
            url=f"{TEST_SUPABASE_URL}/storage/v1/object/list/avatars",
            json=[{"name": "1.png"}, {"name": "2.png"}],
        )
        # Delete call
        httpx_mock.add_response(
            method="DELETE",
            url=f"{TEST_SUPABASE_URL}/storage/v1/object/avatars",
            status_code=200, json={},
        )

        r = client.delete(
            "/api/account/avatar",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 200
        assert r.json() == {"avatar_url": None}
        assert captured["patch"] == {"avatar_url": None}


class TestProfileSave:
    """PATCH /api/account/profile — owner writes their editable fields via
    service-role (authenticated has no direct DML on profiles, migration 0008)."""

    def test_anonymous_401(self, client):
        r = client.patch("/api/account/profile", json={"username": "x"})
        assert r.status_code == 401

    def test_invalid_github_handle_400(self, client, user_token):
        # A bad handle must be rejected before any DB write (no httpx_mock →
        # a REST call would error, proving none is made).
        r = client.patch(
            "/api/account/profile",
            json={"github_handle": "bad--handle"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 400

    def test_valid_save_patches_profiles(self, client, user_token, httpx_mock):
        captured = {}

        def _capture(request):
            import json as _json
            captured["body"] = _json.loads(request.content)
            captured["params"] = str(request.url.params)
            import httpx as _httpx
            return _httpx.Response(204)

        httpx_mock.add_callback(_capture, method="PATCH", url=_PROFILES_PATCH_RE)

        r = client.patch(
            "/api/account/profile",
            json={
                "username": "shenh10",
                "full_name": "Han Shen",
                "bio": "",
                "github_handle": "@shenh10",
            },
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert r.status_code == 200
        # Only allow-listed fields; empty bio → NULL; handle stripped of '@';
        # scoped to the caller's own row.
        assert captured["body"] == {
            "username": "shenh10",
            "full_name": "Han Shen",
            "bio": None,
            "github_handle": "shenh10",
        }
        assert "id=eq.user-uuid-1" in captured["params"]
