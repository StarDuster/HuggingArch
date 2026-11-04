"""Verify the core / service boundary stays clean.

The OSS analyzer (`backend.{api,arch,inference,rendering,hf,agents,
spec_worker}` + `analyzer.py` + `_types.py` + `custom_model.py`) must
never import anything from `backend.service`. Conversely, the service
layer freely imports core (the shared FastAPI app, _types, etc.) — that
direction is allowed.

The single intentional exception is `backend/api/__init__.py`, which
opt-in imports `backend.service.ha_api` to register service routes against
the shared app. Drop that line + the two service directories for a
no-service build.

Mechanism: walk every `*.py` under the core packages with `ast`, look
for `Import` / `ImportFrom` nodes whose target starts with
`backend.service`. Anything we find that isn't on the allowlist is a
boundary violation.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Allowlist: file path (relative to repo root) → set of allowed
# `backend.service…` import targets. Empty set = no service imports
# allowed; presence = exactly these are allowed (other ones still fail).
#
# Each entry below is an INTENTIONAL weak dependency: the importing
# core file falls back gracefully when the service layer is absent
# (i.e. when SUPABASE_* env isn't configured). Without that fallback
# pattern, the import would belong outside core entirely.
SERVICE_IMPORT_ALLOWLIST: dict[str, set[str]] = {
    # Opt-in registration of service routes against the shared FastAPI
    # app. Without this line the dashboard / auth pages 404 silently.
    "backend/api/__init__.py": {"backend.service.ha_api"},

    # Write-through to the same cache after a successful agent run, plus
    # the per-user generation record (Phase 1 multi-user). Both silent-fail
    # when service is unconfigured, so spec generation still works without
    # the service layer — the writes are just skipped.
    "backend/spec_worker/runtime.py": {
        "backend.service.spec_generation_logs",
    },

    "backend/spec_worker/worker.py": {
        "backend.service.spec_generation_logs",
    },

    # Where a run's generation record is kept: `spec_generations` when
    # Supabase is configured, the host-local store otherwise. The chooser
    # lives in core because `backend.service` is the optional half — a
    # no-service build must still keep the local record. Same lazy,
    # supabase_env-guarded import as the writes above.
    "backend/spec_worker/records.py": {
        "backend.service.spec_generations",
    },

    # The agent review runner persists its verdict + report the same way
    # worker.py persists run logs: a lazy, supabase_env-guarded write that
    # silent-fails when the service layer is unconfigured.
    "backend/spec_worker/reviewer.py": {
        "backend.service.spec_generation_reviews",
    },

}


# Core packages that must stay service-clean (besides the allowlist file).
CORE_DIRS = [
    "backend/api/_app.py",
    "backend/api/_models.py",
    "backend/api/_helpers.py",
    "backend/api/__main__.py",
    "backend/api/gallery_overview.py",
    "backend/api/sweep_runtime.py",
    "backend/api/routes",
    "backend/arch",
    "backend/inference",
    "backend/measure",
    "backend/rendering",
    "backend/hf",
    "backend/agents",
    "backend/spec_worker",
    "backend/analyzer.py",
    "backend/custom_id.py",
    "backend/custom_model.py",
    "backend/_types.py",
]


def _iter_py(path: pathlib.Path):
    if path.is_file() and path.suffix == ".py":
        yield path
    elif path.is_dir():
        for p in path.rglob("*.py"):
            if "__pycache__" not in p.parts:
                yield p


def _service_imports(py_file: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(line_number, snippet)] for every `backend.service…` import
    found in the file. Snippet is rendered as the fully-qualified
    target so the allowlist can be specific (e.g. allow
    `backend.service.shared_spec_cache` without granting all of
    `backend.service`).
    """
    text = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(py_file))
    except SyntaxError:
        return []
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "backend.service":
                # `from backend.service import X, Y` → emit one entry
                # per imported name as `backend.service.X`, etc.
                found.extend((node.lineno, f"backend.service.{alias.name}")
                             for alias in node.names)
            elif module.startswith("backend.service."):
                # `from backend.service.foo import bar` → record
                # `backend.service.foo` (the submodule itself).
                found.append((node.lineno, module))
            elif module == "backend":
                # `from backend import service` is the opt-in pattern
                found.extend((node.lineno, f"backend.{alias.name}")
                             for alias in node.names
                             if alias.name == "service" or alias.name.startswith("service"))
        elif isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names
                         if alias.name == "backend.service"
                         or alias.name.startswith("backend.service."))
    return found


def test_core_does_not_import_service():
    """Core packages must not import `backend.service` (except the one
    opt-in line in `backend/api/__init__.py`)."""
    violations: list[str] = []
    for rel in CORE_DIRS:
        for py in _iter_py(REPO_ROOT / rel):
            rel_path = py.relative_to(REPO_ROOT).as_posix()
            allowed = SERVICE_IMPORT_ALLOWLIST.get(rel_path, set())
            for lineno, snippet in _service_imports(py):
                # Snippet is the fully-qualified target name. Allowlist
                # entries match exactly OR are a prefix that matches
                # the dotted path (so `backend.service.shared_spec_cache` in
                # the allowlist permits `backend.service.shared_spec_cache`
                # but not `backend.service.routes`).
                if snippet in allowed:
                    continue
                violations.append(f"{rel_path}:{lineno}  {snippet}")
    if violations:
        pytest.fail(
            "Service imports leaked into core packages:\n  "
            + "\n  ".join(violations)
        )


def test_api_init_has_opt_in_service_import():
    """`backend/api/__init__.py` is the *one* place that imports
    `backend.service`. If this line ever disappears, service routes won't
    register against the shared app and the dashboard will silently 404.
    """
    init_file = REPO_ROOT / "backend" / "api" / "__init__.py"
    assert init_file.is_file()
    text = init_file.read_text(encoding="utf-8")
    assert "from backend.service import ha_api" in text, (
        "Lost the opt-in `from backend.service import ha_api` line in "
        "backend/api/__init__.py — service endpoints will silently 404."
    )


def test_service_routes_register_on_app():
    """Boot the app, assert all three service routes show up in app.routes."""
    from backend.api import app

    paths = {r.path for r in app.routes if hasattr(r, "path")}
    expected = {"/site-config.js", "/analytics/event", "/api/admin/health"}
    missing = expected - paths
    assert not missing, f"Service routes failed to register: {missing}"


def test_auth_pages_are_spa_not_static_mounts():
    """Auth/account/admin HTML is the React SPA, not StaticFiles on frontend/service."""
    from backend.api import app

    mount_names = {getattr(r, "name", None) for r in app.routes}
    leftover = {"service_auth", "service_account", "service_admin", "service_js", "service_css"} & mount_names
    assert not leftover, f"vanilla service static mounts must be gone: {leftover}"


def _bench_imports(py_file: pathlib.Path) -> list[tuple[int, str]]:
    """`(line, target)` of every import of the Kernel Bench service's package
    (`backend.service.bench…`) in the file, at any depth."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "backend.service" and any(a.name == "bench" for a in node.names):
                found.append((node.lineno, "backend.service.bench"))
            elif module == "backend.service.bench" or module.startswith("backend.service.bench."):
                found.append((node.lineno, module))
        elif isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names
                         if alias.name.startswith("backend.service.bench"))
    return found


def test_only_the_kernel_bench_service_opens_the_lake():
    """The lake belongs to the Kernel Bench service's own process
    (`backend.service.bench.server`, docs/measured_calibration.md §5.1).
    The HA API, its compute workers (which import `backend.api`) and the
    estimator reach it over HTTP only: nothing under `backend/` outside the
    service's package imports it — the lake, DuckDB, the routes."""
    violations = []
    for py in _iter_py(REPO_ROOT / "backend"):
        rel = py.relative_to(REPO_ROOT).as_posix()
        if rel.startswith("backend/service/bench/"):
            continue
        violations += [f"{rel}:{line} imports {target}" for line, target in _bench_imports(py)]
    assert violations == [], "\n".join(violations)


#: The modules of the service layer and the HA API, as the entry points and
#: the tests import them. Each must be importable FIRST in a fresh process:
#: `backend/api/__init__.py` imports `backend.service.ha_api` at its end, so a
#: service module that imports anything of `backend.api`'s at module level
#: re-enters that package init and the wiring runs against a half-built
#: module (once, `bench_proxy` imported `health_detail` and
#: `backend.service.bench_proxy` stopped being importable on its own, which
#: is why `install` takes the registrar from its caller).
_IMPORT_FIRST = (
    "backend.api",
    "backend.api.routes.meta",
    "backend.service",
    "backend.service.ha_api",
    "backend.service.bench_proxy",
    "backend.service.bench.server",
    "backend.service.routes.account",
)

#: Importing these builds no HA API app: the Kernel Bench service runs in its
#: own process and must not (docs/measured_calibration.md §5.1).
_NO_APP = ("backend.service", "backend.service.bench_proxy", "backend.service.bench.server")


@pytest.mark.parametrize("module", _IMPORT_FIRST)
def test_any_module_can_be_imported_first(module):
    """In a fresh process: the import succeeds, the app ends up wired once
    (one `/api/bench/*` route, however the cycle was entered), and the
    service-only modules built no app on their own."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys, importlib; importlib.import_module({module!r});"
         "built = 'backend.api._app' in sys.modules;"
         "import backend.api;"
         "n = sum(1 for r in backend.api.app.routes"
         " if getattr(r, 'path', '') == '/api/bench/{path:path}');"
         "print(built, n)"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-3000:]
    built, routes = out.stdout.split()[-2:]
    assert routes == "1", f"{module}: {routes} bench routes"
    assert built == str(module not in _NO_APP), f"{module}: app built = {built}"
