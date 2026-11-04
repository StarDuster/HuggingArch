"""GitHub auth for the contribute→PR module (backend/service/github.py).

Covers the two auth modes without touching the network:
  * `enabled()` / `_token()` mode selection (App > PAT > neither),
  * private-key parsing (raw PEM, `\\n`-escaped, base64-wrapped),
  * App JWT signing (RS256, verifiable with the public key),
  * installation-token caching (minted once, reused until expiry),
  * the Co-authored-by trailer derivation.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.service import github


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv, pub


_GH_ENV = (
    "HA_GITHUB_APP_ID", "HA_GITHUB_APP_PRIVATE_KEY",
    "HA_GITHUB_APP_INSTALLATION_ID", "HA_GITHUB_TOKEN", "HA_GITHUB_REPO",
)


class _GitHubEnvBase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _GH_ENV}
        for k in _GH_ENV:
            os.environ.pop(k, None)
        github._inst_token["token"] = None
        github._inst_token["exp"] = 0.0

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        github._inst_token["token"] = None
        github._inst_token["exp"] = 0.0


class TestAuthModeSelection(_GitHubEnvBase):
    def test_disabled_when_nothing_set(self):
        self.assertFalse(github.enabled())
        with self.assertRaises(github.GitHubNotConfigured):
            github._token()

    def test_pat_mode(self):
        os.environ["HA_GITHUB_TOKEN"] = "github_pat_abc"
        self.assertTrue(github.enabled())
        self.assertFalse(github._app_configured())
        self.assertEqual(github._token(), "github_pat_abc")

    def test_app_mode_preferred_over_pat(self):
        priv, _ = _keypair()
        os.environ["HA_GITHUB_APP_ID"] = "111"
        os.environ["HA_GITHUB_APP_PRIVATE_KEY"] = priv
        os.environ["HA_GITHUB_TOKEN"] = "github_pat_should_be_ignored"
        self.assertTrue(github.enabled())
        self.assertTrue(github._app_configured())
        with mock.patch.object(
            github, "_mint_installation_token",
            return_value="ghs_installation_token",
        ) as m:
            self.assertEqual(github._token(), "ghs_installation_token")
            # cached: a second call must NOT mint again.
            self.assertEqual(github._token(), "ghs_installation_token")
            self.assertEqual(m.call_count, 1)

    def test_app_token_refreshes_after_expiry(self):
        priv, _ = _keypair()
        os.environ["HA_GITHUB_APP_ID"] = "111"
        os.environ["HA_GITHUB_APP_PRIVATE_KEY"] = priv
        with mock.patch.object(
            github, "_mint_installation_token",
            side_effect=["token-1", "token-2"],
        ) as m:
            self.assertEqual(github._token(), "token-1")
            # force expiry → next call re-mints.
            github._inst_token["exp"] = time.time() - 1
            self.assertEqual(github._token(), "token-2")
            self.assertEqual(m.call_count, 2)


class TestPrivateKeyParsing(_GitHubEnvBase):
    def test_raw_pem(self):
        priv, _ = _keypair()
        os.environ["HA_GITHUB_APP_PRIVATE_KEY"] = priv
        self.assertIn("-----BEGIN", github._app_private_key())

    def test_backslash_n_escaped(self):
        priv, _ = _keypair()
        os.environ["HA_GITHUB_APP_PRIVATE_KEY"] = priv.replace("\n", "\\n")
        parsed = github._app_private_key()
        self.assertIn("\n", parsed)
        self.assertNotIn("\\n", parsed)

    def test_base64_wrapped(self):
        import base64
        priv, _ = _keypair()
        os.environ["HA_GITHUB_APP_PRIVATE_KEY"] = base64.b64encode(
            priv.encode()).decode()
        self.assertIn("-----BEGIN", github._app_private_key())

    def test_unset_returns_none(self):
        self.assertIsNone(github._app_private_key())


class TestAppJwt(_GitHubEnvBase):
    def test_jwt_is_rs256_and_verifiable(self):
        priv, pub = _keypair()
        os.environ["HA_GITHUB_APP_ID"] = "424242"
        os.environ["HA_GITHUB_APP_PRIVATE_KEY"] = priv
        token = github._app_jwt()
        decoded = jwt.decode(token, pub, algorithms=["RS256"])
        self.assertEqual(decoded["iss"], "424242")
        # exp is in the future, within GitHub's 10-minute ceiling.
        self.assertGreater(decoded["exp"], decoded["iat"])
        self.assertLessEqual(decoded["exp"] - decoded["iat"], 600)


class TestCoauthorTrailer(unittest.TestCase):
    def test_name_and_email(self):
        t = github._coauthor_trailer("Ada Lovelace <ada@example.com>")
        self.assertIn("Co-authored-by: Ada Lovelace <ada@example.com>", t)

    def test_bare_email(self):
        t = github._coauthor_trailer("ada@example.com")
        self.assertIn("Co-authored-by: HuggingArch contributor <ada@example.com>", t)

    def test_non_email_yields_no_trailer(self):
        # An opaque internal user id must NOT become a bogus trailer.
        self.assertEqual(github._coauthor_trailer("a1b2c3d4-uuid"), "")
        self.assertEqual(github._coauthor_trailer(None), "")
        self.assertEqual(github._coauthor_trailer(""), "")


if __name__ == "__main__":
    unittest.main()
