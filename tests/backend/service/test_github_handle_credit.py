"""GitHub-handle → contributor-credit logic (Phase 5 credit closure):
handle validation (`profiles.normalize_handle`) and the co-author string
builder (`github.coauthor_for_handle`), including the noreply fallback when
GitHub auth isn't configured.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.service import github, profiles


class TestNormalizeHandle(unittest.TestCase):
    def test_valid(self):
        for h in ("octocat", "torvalds", "a", "user-name", "a1-b2-c3"):
            self.assertEqual(profiles.normalize_handle(h), h)

    def test_strips_at_and_space(self):
        self.assertEqual(profiles.normalize_handle("  @octocat "), "octocat")

    def test_invalid_returns_none(self):
        for bad in ("", None, "-lead", "trail-", "has space", "under_score",
                    "a" * 40, "bad!char"):
            self.assertIsNone(profiles.normalize_handle(bad))


class TestCoauthorForHandle(unittest.TestCase):
    def setUp(self):
        # Ensure no App/PAT is configured → exercises the noreply fallback path.
        self._saved = {k: os.environ.pop(k, None) for k in (
            "HA_GITHUB_APP_ID", "HA_GITHUB_APP_PRIVATE_KEY", "HA_GITHUB_TOKEN")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_empty_handle_is_none(self):
        self.assertIsNone(github.coauthor_for_handle(""))
        self.assertIsNone(github.coauthor_for_handle(None))

    def test_fallback_noreply_when_unconfigured(self):
        # enabled() is False → no API call, plain noreply address.
        self.assertEqual(
            github.coauthor_for_handle("@octocat"),
            "octocat <octocat@users.noreply.github.com>",
        )

    def test_resolves_numeric_id_when_configured(self):
        os.environ["HA_GITHUB_TOKEN"] = "github_pat_x"

        class _Resp:
            status_code = 200
            @staticmethod
            def json():
                return {"id": 583231}

        class _Client:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url): return _Resp()

        with mock.patch.object(github.httpx, "Client", _Client):
            self.assertEqual(
                github.coauthor_for_handle("octocat"),
                "octocat <583231+octocat@users.noreply.github.com>",
            )


if __name__ == "__main__":
    unittest.main()
