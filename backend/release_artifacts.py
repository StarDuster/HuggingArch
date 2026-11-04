"""The small index shared by release bootstrap artifacts.

Indexes are mutable pointers; the archives they name are immutable and
digest-addressed. A missing index or pinned object means the cache layer falls
back to runtime warm; only a present-but-invalid artifact fails startup.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional
import urllib.error


INDEX_SCHEMA = 1


def valid_artifact_path(path: object) -> bool:
    return (
        isinstance(path, str)
        and bool(path)
        and not path.startswith("/")
        and ".." not in Path(path).parts
        and path.endswith(".tar.gz")
    )


def read_object(url: str, key: str, timeout: float,
                request: Callable[..., Any]) -> Optional[bytes]:
    """The object's bytes, or ``None`` when storage has no such object.

    Supabase Storage answers a missing object with HTTP 400 and a not-found
    body (``"code":"NoSuchKey"``), not with 404; both mean absent. Any other
    failure raises.
    """
    try:
        with request(url, key, timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        if exc.code == 400:
            body = exc.read().decode(errors="replace")
            if "NoSuchKey" in body or "Object not found" in body:
                return None
        raise


def artifact_to_pull(configured: str, index_path: str, expected_engine: str,
                     base: str, key: str, bucket: str, timeout: float,
                     request: Callable[..., Any]
                     ) -> Optional[tuple[str, str]]:
    """``(path, sha256)`` of the archive to install, ``None`` when absent.

    ``configured`` pins an archive path (its digest is the ``.sha256``
    sibling); empty reads this engine version's index.
    """
    if not configured:
        return selected_artifact(index_path, expected_engine, base, key,
                                 bucket, timeout, request)
    digest = read_object(f"{base}/storage/v1/object/{bucket}/{configured}.sha256",
                         key, timeout, request)
    if digest is None:
        return None
    return configured, digest.decode().strip().split()[0]


def selected_artifact(index_path: str, expected_engine: str, base: str,
                      key: str, bucket: str, timeout: float,
                      request: Callable[..., Any]
                      ) -> Optional[tuple[str, str]]:
    raw = read_object(f"{base}/storage/v1/object/{bucket}/{index_path}", key,
                      timeout, request)
    if raw is None:
        return None
    index = json.loads(raw)
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError("release artifact index has an unknown schema")
    if index.get("engine_version") != expected_engine:
        raise ValueError("release artifact index names another engine version")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("release artifact index has no artifacts")
    selected = artifacts[0]
    path, digest = selected.get("path"), selected.get("sha256")
    if not valid_artifact_path(path) or not isinstance(digest, str):
        raise ValueError("release artifact index has an invalid artifact")
    return path, digest
