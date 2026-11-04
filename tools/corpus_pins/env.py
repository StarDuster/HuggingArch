"""The environment an admission measures in, and whether it may write.

Two trees with the same spec can fingerprint differently under two
interpreters or two dependency sets, so a write is only as reproducible as
its environment. The digest names it: the interpreter's major.minor, the
`uv.lock` sha256, and a sha256 of the runtime distributions actually
installed (`name==version`, sorted). Dev dependencies are left out: CI
installs them and the deployed image does not, and the two must agree.

A run may write only when the interpreter is `ADMIT_PYTHON` and the installed
runtime distributions are exactly what the lock resolves to for this
interpreter and platform. The lock's resolution is read from
`uv export --frozen --no-dev` (no environment is created or changed),
normalized to the `name==version` entries whose markers hold here. Every
other run is a preview.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tools.corpus_pins.corpus import ROOT

#: The interpreter the admitter writes under. CI (`uv python install`) and
#: the deployed image (`FROM python:<ver>-slim`) run the same one; the
#: registry test holds all three equal.
ADMIT_PYTHON = "3.13"


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def lock_sha256(root: Path = ROOT) -> str:
    return hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()


def lock_sha256_at(sha: str, root: Path = ROOT) -> Optional[str]:
    """sha256 of `uv.lock` as committed at `sha`; None when git cannot show it."""
    out = subprocess.run(["git", "show", f"{sha}:uv.lock"], cwd=root,
                         capture_output=True, check=False)
    return hashlib.sha256(out.stdout).hexdigest() if out.returncode == 0 else None


def locked_runtime(root: Path = ROOT) -> Optional[Dict[str, str]]:
    """`{name: version}` the lock resolves to for this interpreter and
    platform, without dev dependencies; None when `uv` is not available."""
    from packaging.markers import Marker
    uv = shutil.which("uv")
    if uv is None:
        return None
    out = subprocess.run(
        [uv, "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project",
         "--format", "requirements-txt"],
        cwd=root, capture_output=True, text=True, check=True)
    pins: Dict[str, str] = {}
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        requirement, _, marker = line.partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate():
            continue
        name, _, version = requirement.strip().partition("==")
        pins[normalize(name)] = version.strip()
    return pins


def installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def installed(names: List[str]) -> Dict[str, str]:
    return {name: installed_version(name) for name in names}


def digest(pins: Dict[str, str]) -> str:
    return hashlib.sha256("\n".join(
        f"{name}=={version}" for name, version in sorted(pins.items())).encode()).hexdigest()


@dataclass(frozen=True)
class Environment:
    python: str
    uv_lock_sha256: str
    #: sha256 of the installed runtime distributions; None without `uv`
    dists_sha256: Optional[str]
    #: sha256 the lock resolves to here; None without `uv`
    locked_dists_sha256: Optional[str]

    @classmethod
    def here(cls, root: Path = ROOT) -> "Environment":
        locked = locked_runtime(root)
        return cls(
            python=python_version(),
            uv_lock_sha256=lock_sha256(root),
            dists_sha256=digest(installed(sorted(locked))) if locked is not None else None,
            locked_dists_sha256=digest(locked) if locked is not None else None,
        )

    def write_refusals(self, base_lock_sha256: Optional[str]) -> List[str]:
        """Why this environment may not write against a base whose `uv.lock`
        hashes to `base_lock_sha256`; empty when it may."""
        out = []
        if base_lock_sha256 is None:
            out.append("the base's uv.lock cannot be read")
        elif base_lock_sha256 != self.uv_lock_sha256:
            out.append("uv.lock differs from the base's uv.lock")
        if self.python != ADMIT_PYTHON:
            out.append(f"interpreter {self.python} is not ADMIT_PYTHON {ADMIT_PYTHON}")
        if self.dists_sha256 is None:
            out.append("uv is not available to read the lock's resolution")
        elif self.dists_sha256 != self.locked_dists_sha256:
            out.append("installed runtime distributions differ from the lock's resolution")
        return out

    def summary(self) -> Dict[str, Optional[str]]:
        return {"python": self.python, "uv_lock_sha256": self.uv_lock_sha256,
                "dists_sha256": self.dists_sha256}
