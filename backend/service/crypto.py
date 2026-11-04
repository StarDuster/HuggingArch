"""Symmetric encryption for BYOK agent credentials (Phase 2).

User API keys are encrypted at rest with a server-only secret so Supabase
only ever stores ciphertext. Uses Fernet (AES-128-CBC + HMAC) from the
`cryptography` package.

The secret comes from `HA_CREDS_ENC_KEY`. It may be either a real 32-byte
urlsafe-base64 Fernet key (generate with `Fernet.generate_key()`) or any
arbitrary passphrase — in the latter case we derive the Fernet key by
SHA-256, so operators can set a human-chosen secret without ceremony.

If `HA_CREDS_ENC_KEY` is unset, `encrypt`/`decrypt` raise — BYOK is simply
unavailable until an operator configures it. Callers in the credential
path surface that as a clear error rather than storing plaintext.
"""
from __future__ import annotations

import base64
import functools
import hashlib
import os

from cryptography.fernet import Fernet


class CredentialsKeyMissing(RuntimeError):
    """HA_CREDS_ENC_KEY is not configured — BYOK storage is disabled."""


def enabled() -> bool:
    return bool(os.environ.get("HA_CREDS_ENC_KEY"))


@functools.lru_cache(maxsize=1)
def _fernet_for(secret: str) -> Fernet:
    """Build a Fernet from the secret. Accepts a real Fernet key verbatim;
    otherwise derives a 32-byte key via SHA-256. Cached per secret value."""
    try:
        return Fernet(secret)  # already a valid urlsafe-base64 32-byte key
    except (ValueError, TypeError):
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        return Fernet(base64.urlsafe_b64encode(digest))


def _fernet() -> Fernet:
    secret = os.environ.get("HA_CREDS_ENC_KEY")
    if not secret:
        raise CredentialsKeyMissing(
            "HA_CREDS_ENC_KEY is not set — cannot store BYOK credentials."
        )
    return _fernet_for(secret)


def encrypt(plaintext: str) -> str:
    """Return ciphertext (urlsafe-base64 token) for the given plaintext."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(ciphertext: str) -> str:
    """Recover plaintext from a token produced by `encrypt`."""
    return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
