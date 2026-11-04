#!/usr/bin/env python3
"""Read-only gate: can every stored private spec still load?

Schema changes ship against the YAML in git. `spec_generations` /
`cached_specs` / `~/.huggingarch/user_specs` keep the bytes the agent
wrote, and extra=forbid will refuse a renamed field. This tool materializes
each row the same way serving does (the loader, not a second rule set) and
prints why a row fails.

It never writes the database. Default overlays write the stored bytes so
the report is the row as saved. `--migrate-preview` runs the pure rewrite
and also the serving path (`normalize_bundle`) to show whether that would
make the row load.

Usage (from repo root; do not use `uv run`):

    PYTHONPATH=. ./.venv/bin/python tools/check_stored_specs.py
    PYTHONPATH=. ./.venv/bin/python tools/check_stored_specs.py --local-only
    PYTHONPATH=. ./.venv/bin/python tools/check_stored_specs.py --db-json dump.json
    PYTHONPATH=. ./.venv/bin/python tools/check_stored_specs.py --migrate-preview
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


logging.disable(logging.WARNING)


@dataclass
class CheckRow:
    source: str
    model_id: str
    status: str = ""
    latest: bool = False
    load: str = ""
    message: str = ""
    migrate_notes: tuple[str, ...] = ()
    migrate_blockers: tuple[str, ...] = ()
    migrate_load: str = ""
    migrate_message: str = ""
    created_at: str = ""
    record_id: str = ""
    user_id: str = ""


def check_overlay(root: str, *, model_id: str = "") -> tuple[str, str]:
    """Load one overlay with the real loader. Returns (LOAD_OK|LOAD_FAIL, msg)."""
    from backend.arch.loader import SpecLoader, overlay_kwargs

    loader = SpecLoader(**overlay_kwargs(root))
    root_s = os.path.abspath(root)
    fails = [
        (p, str(e)) for (p, _t, _i, e) in loader._model_load_failures
        if os.path.abspath(str(p)).startswith(root_s)
    ]
    if fails:
        msg = fails[0][1].replace("\n", " | ")
        return "LOAD_FAIL", msg
    return "LOAD_OK", ""


def _bundle_for_check(model_id: str, spec_yaml: str, comps, *, migrate: bool):
    from backend.spec_worker.bundle import (
        BundleFile, FormalSource, NormalizedBundle, normalize_bundle,
    )

    if migrate:
        return normalize_bundle(
            model_id, spec_yaml, comps or [], FormalSource.from_deployment())
    serving = []
    for comp in comps or []:
        if not isinstance(comp, dict):
            continue
        path = (comp.get("path") or "").strip()
        body = comp.get("body")
        if path and isinstance(body, str) and body:
            serving.append(BundleFile(path=path, body=body))
    return NormalizedBundle(
        model_id=model_id,
        spec_yaml=spec_yaml,
        serving_files=tuple(serving),
        proposal_files=(),
        notes=(),
        formal_source=FormalSource(models={}, components={}),
    )


def _materialize_record(user_id: str, model_id: str, spec_yaml: str,
                        comps, *, migrate: bool, tmp: Path) -> str:
    from backend.spec_worker import overlay as ov

    dest = tmp / ("migrated" if migrate else "raw")
    dest.mkdir(parents=True, exist_ok=True)
    bundle = _bundle_for_check(model_id, spec_yaml, comps, migrate=migrate)
    with _override_user_specs(dest):
        return ov.materialize_user_spec(user_id, model_id, bundle)


def _override_user_specs(dest: Path):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        prev = os.environ.get("HA_USER_SPECS_ROOT")
        os.environ["HA_USER_SPECS_ROOT"] = str(dest)
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("HA_USER_SPECS_ROOT", None)
            else:
                os.environ["HA_USER_SPECS_ROOT"] = prev
    return _ctx()


def _load_failure_line(exc: BaseException) -> tuple[str, str]:
    from backend.arch.loader import SpecLoadError
    from backend.spec_worker.bundle import SpecBundleError

    if isinstance(exc, SpecLoadError):
        return "LOAD_FAIL", str(exc).replace("\n", " | ")
    if isinstance(exc, SpecBundleError):
        first = exc.errors[0] if exc.errors else str(exc)
        return "LOAD_FAIL", first.replace("\n", " | ")
    return "LOAD_FAIL", f"{exc.__class__.__name__}: {exc}".replace("\n", " | ")


def _migrate_serving_load(user_id: str, model_id: str, spec_yaml: str,
                          comps, tmp_path: Path) -> tuple[str, str]:
    """Serving-path preview: never raise; a missing base is a LOAD_FAIL line."""
    try:
        root_m = _materialize_record(
            user_id, model_id, spec_yaml, comps,
            migrate=True, tmp=tmp_path)
        return check_overlay(root_m, model_id=model_id)
    except Exception as exc:
        return _load_failure_line(exc)


def check_bundle(model_id: str, spec_yaml: str, comps=None, *,
                 user_id: str = "check", migrate_preview: bool = False) -> CheckRow:
    """Materialize one stored bundle into a temp overlay and load it."""
    from backend.arch.spec_migrate import migrate_yaml_text

    row = CheckRow(source="record", model_id=model_id, user_id=user_id)
    with tempfile.TemporaryDirectory(prefix="check_stored_") as tmp:
        tmp_path = Path(tmp)
        try:
            root = _materialize_record(
                user_id, model_id, spec_yaml, comps,
                migrate=False, tmp=tmp_path)
            row.load, row.message = check_overlay(root, model_id=model_id)
        except Exception as exc:
            row.load, row.message = _load_failure_line(exc)

        if migrate_preview:
            migrated = migrate_yaml_text(spec_yaml)
            row.migrate_notes = migrated.notes
            row.migrate_blockers = migrated.blockers
            if not migrated.ok:
                row.migrate_load = "UNMIGRATABLE"
                row.migrate_message = "; ".join(migrated.blockers)
            else:
                row.migrate_load, row.migrate_message = _migrate_serving_load(
                    user_id, model_id, spec_yaml, comps, tmp_path)
    return row


def iter_local_overlays(root: Optional[Path] = None) -> Iterable[CheckRow]:
    from backend.spec_worker.overlay import user_specs_root

    base = Path(root) if root else user_specs_root()
    if not base.is_dir():
        return
    for user_dir in sorted(base.iterdir()):
        if not user_dir.is_dir():
            continue
        for model_dir in sorted(user_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            models = model_dir / "models"
            if not models.is_dir():
                continue
            model_id = model_dir.name.replace("__", "/", 1)
            load, msg = check_overlay(str(model_dir), model_id=model_id)
            yield CheckRow(
                source=f"local:{user_dir.name}",
                model_id=model_id,
                user_id=user_dir.name,
                load=load,
                message=msg,
            )


def _page_table(url: str, headers: dict, table: str, select: str) -> list[dict]:
    import httpx

    out: list[dict] = []
    total: int | None = None
    page = 200
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        while True:
            lo = len(out)
            r = client.get(
                f"{url}/rest/v1/{table}",
                params={"select": select, "order": "created_at.desc"},
                headers={**headers, "Range-Unit": "items",
                         "Range": f"{lo}-{lo + page - 1}",
                         "Prefer": "count=exact"},
            )
            r.raise_for_status()
            chunk = r.json() or []
            out.extend(chunk)
            rng = r.headers.get("Content-Range", "")
            if total is None and "/" in rng:
                tail = rng.rsplit("/", 1)[1]
                total = int(tail) if tail.isdigit() else None
            if not chunk or (total is not None and len(out) >= total):
                break
    return out


def iter_db_records() -> Iterable[dict]:
    from backend import supabase_env, supabase_rest

    if not supabase_env.configured():
        return
    url, key = supabase_env.service_role()
    headers = supabase_rest.service_headers(key, json_body=False)
    yield from _page_table(
        url, headers, "spec_generations",
        "id,user_id,model_id,spec_yaml,new_component_yamls,status,created_at",
    )


def iter_cached_records() -> Iterable[dict]:
    from backend import supabase_env, supabase_rest

    if not supabase_env.configured():
        return
    url, key = supabase_env.service_role()
    headers = supabase_rest.service_headers(key, json_body=False)
    yield from _page_table(
        url, headers, "cached_specs",
        "model_id,spec_yaml,new_component_yamls,created_at",
    )


def _mark_latest(rows: list[CheckRow]) -> None:
    """Newest-first input: first (source, user, model) is latest. Display only."""
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.source, row.user_id, row.model_id)
        if key not in seen:
            row.latest = True
            seen.add(key)


def _fmt(row: CheckRow, *, migrate_preview: bool) -> str:
    flag = "LATEST" if row.latest else "older "
    msg = row.message.replace("\n", " | ")[:300]
    line = (
        f"{row.created_at[:10] or '----------'} {flag} "
        f"{row.source:22s} {row.model_id:45s} "
        f"{(row.status or '-'):12s} {row.load:12s} {msg}"
    )
    if migrate_preview:
        extra = row.migrate_load or "-"
        detail = row.migrate_message or (", ".join(row.migrate_notes) if row.migrate_notes else "")
        line += f"\n           migrate -> {extra:12s} {detail[:300]}"
    return line


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Load stored private specs with the current loader.")
    ap.add_argument("--local-only", action="store_true",
                    help="Skip Supabase; only scan HA_USER_SPECS_ROOT")
    ap.add_argument("--db-json", type=Path,
                    help="spec_generations dump {spec_generations: [...]} "
                         "as produced by a read-only export")
    ap.add_argument("--migrate-preview", action="store_true",
                    help="Also run the pure rewrite in memory (no DB writes)")
    ap.add_argument("--user-specs-root", type=Path, default=None)
    args = ap.parse_args(argv)

    rows: list[CheckRow] = []

    if args.db_json:
        payload = json.loads(args.db_json.read_text(encoding="utf-8"))
        records = payload.get("spec_generations") or payload
        for rec in records:
            if rec.get("status") == "rejected":
                continue
            uid = rec.get("user_id") or "check"
            row = check_bundle(
                rec.get("model_id") or "", rec.get("spec_yaml") or "",
                rec.get("new_component_yamls"),
                user_id=uid,
                migrate_preview=args.migrate_preview,
            )
            row.source = "db-json"
            row.status = rec.get("status") or ""
            row.created_at = rec.get("created_at") or ""
            row.record_id = rec.get("id") or ""
            row.user_id = uid
            rows.append(row)

    if not args.local_only and not args.db_json:
        from backend import supabase_env
        if not supabase_env.configured():
            print("# spec_generations failed: supabase is not configured "
                  "(pass --local-only or --db-json)", file=sys.stderr)
            return 1
        try:
            db_rows = list(iter_db_records() or [])
        except Exception as exc:
            print(f"# spec_generations failed: {exc}", file=sys.stderr)
            return 1
        for rec in db_rows:
            if rec.get("status") == "rejected":
                continue
            uid = rec.get("user_id") or ""
            row = check_bundle(
                rec.get("model_id") or "", rec.get("spec_yaml") or "",
                rec.get("new_component_yamls"),
                user_id=uid or "check",
                migrate_preview=args.migrate_preview,
            )
            row.source = "spec_generations"
            row.status = rec.get("status") or ""
            row.created_at = rec.get("created_at") or ""
            row.record_id = rec.get("id") or ""
            row.user_id = uid
            rows.append(row)
        try:
            cached = list(iter_cached_records() or [])
        except Exception as exc:
            print(f"# cached_specs failed: {exc}", file=sys.stderr)
            return 1
        for rec in cached:
            row = check_bundle(
                rec.get("model_id") or "", rec.get("spec_yaml") or "",
                rec.get("new_component_yamls"),
                user_id="_cached_specs",
                migrate_preview=args.migrate_preview,
            )
            row.source = "cached_specs"
            row.created_at = rec.get("created_at") or ""
            row.user_id = ""
            rows.append(row)

    local_root = args.user_specs_root
    for row in iter_local_overlays(local_root):
        if args.migrate_preview:
            _preview_local(row, local_root)
        rows.append(row)

    rows.sort(key=lambda r: r.created_at or "", reverse=True)
    _mark_latest(rows)
    fails = 0
    latest_fails = 0
    for row in rows:
        print(_fmt(row, migrate_preview=args.migrate_preview))
        if row.load != "LOAD_OK":
            fails += 1
            if row.latest:
                latest_fails += 1
    print(f"# {len(rows)} checked, {fails} load failures "
          f"({latest_fails} among latest-per-key)")
    return 1 if fails else 0


def _preview_local(row: CheckRow, local_root: Optional[Path]) -> None:
    from backend.arch.spec_migrate import migrate_yaml_text
    from backend.spec_worker.overlay import _safe, user_specs_root

    base = Path(local_root) if local_root else user_specs_root()
    user = row.source.split(":", 1)[1]
    ov = base / user / _safe(row.model_id)
    models_dir = ov / "models"
    if not models_dir.is_dir():
        return
    identity = models_dir / f"{_safe(row.model_id)}.yaml"
    yamls = [identity] if identity.is_file() else sorted(models_dir.glob("*.yaml"))
    if not yamls:
        return
    text = yamls[0].read_text(encoding="utf-8")
    migrated = migrate_yaml_text(text)
    row.migrate_notes = migrated.notes
    row.migrate_blockers = migrated.blockers
    if not migrated.ok:
        row.migrate_load = "UNMIGRATABLE"
        row.migrate_message = "; ".join(migrated.blockers)
        return
    with tempfile.TemporaryDirectory(prefix="check_mig_") as tmp:
        row.migrate_load, row.migrate_message = _migrate_serving_load(
            user, row.model_id, text, _local_drafts(ov), Path(tmp))


def _local_drafts(ov: Path) -> list[dict]:
    drafts = ov / "components" / "drafts"
    if not drafts.is_dir():
        return []
    return [
        {
            "path": f"backend/arch/components/drafts/{path.name}",
            "body": path.read_text(encoding="utf-8"),
        }
        for path in sorted(drafts.glob("*.yaml"))
    ]


if __name__ == "__main__":
    sys.exit(main())
