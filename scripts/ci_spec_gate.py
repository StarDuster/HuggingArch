"""CI gate for a spec-only change.

A contributed spec is model YAML + a snapshot (+ optional draft
components). That cannot break FastAPI, the web UI, or kernel_bench, so
the full pytest suite is the wrong job. This file is the right one:

  classify  every path in the diff is spec data (or it is not); a diff that
            adds a snapshot is never spec-only, because a new model owes a
            key to every corpus pin and the full suite is what compares them
  run       load + validate each changed snapshot (the publish gate)

Used by `.github/workflows/test.yml`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# A change that stays inside this set is a spec contribution. Anything else
# (Python, tests, web, library components, dims.yaml, ops_registry, …)
# runs the full suite. Library components and dims.yaml are in the engine's
# data plane: editing them can break every model, so they are not spec-only.
SPEC_ONLY_PREFIXES = (
    "backend/arch/models/",
    "backend/arch/snapshots/",
    "backend/arch/components/drafts/",
)
SPEC_ONLY_FILES = frozenset({
    "backend/arch/supported_models.txt",
})

# Schema / retired-field parsing changes must re-load stored private specs.
STORED_SPEC_CHECK_PREFIXES = (
    "backend/arch/schema/",
)
STORED_SPEC_CHECK_FILES = frozenset({
    "backend/arch/spec_parsing.py",
})


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def is_spec_path(path: str) -> bool:
    p = _norm(path)
    if p in SPEC_ONLY_FILES:
        return True
    return any(p.startswith(prefix) for prefix in SPEC_ONLY_PREFIXES)


def adds_a_snapshot(added: list[str]) -> bool:
    """A snapshot new to the corpus (added, or renamed to a new name) owes a
    key to every corpus pin, and only the full suite compares them all."""
    return any(_norm(p).startswith("backend/arch/snapshots/") and _norm(p).endswith(".json")
               for p in added if p.strip())


def is_spec_only(paths: list[str], added: list[str] = ()) -> bool:
    files = [_norm(p) for p in paths if p.strip()]
    if not files or adds_a_snapshot(list(added)):
        return False
    return all(is_spec_path(p) for p in files)


def requires_stored_spec_check(paths: list[str]) -> bool:
    """True when the diff touches schema or the retired-field parsing table."""
    for raw in paths:
        p = _norm(raw)
        if p in STORED_SPEC_CHECK_FILES:
            return True
        if any(p.startswith(prefix) for prefix in STORED_SPEC_CHECK_PREFIXES):
            return True
    return False


def models_to_validate_for_paths(paths: list[str], root: Path = ROOT) -> list[str]:
    """Model ids to revalidate after a spec, snapshot, draft, or base change.

    Shared names have referers beyond the path in the diff: a draft is opted in
    by every identity that lists it, and a base may sit in another identity's
    inheritance chain. Those models join the direct targets before the
    snapshot-on-disk filter.
    """
    changed_models: set[str] = set()
    changed_drafts: set[str] = set()
    for raw in paths:
        p = _norm(raw)
        if ((p.startswith("backend/arch/snapshots/") and p.endswith(".json"))
                or (p.startswith("backend/arch/models/")
                    and p.endswith((".yaml", ".yml")))):
            changed_models.add(Path(p).stem)
        elif (p.startswith("backend/arch/components/drafts/")
              and p.endswith((".yaml", ".yml"))):
            changed_drafts.add(Path(p).stem)

    documents: dict[str, dict] = {}
    models_dir = root / "backend" / "arch" / "models"
    for path in sorted([*models_dir.glob("*.yaml"), *models_dir.glob("*.yml")]):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            documents[path.stem] = raw

    selected: set[str] = set(changed_models)
    if changed_drafts:
        for stem, raw in documents.items():
            drafts = raw.get("drafts") or []
            if not isinstance(drafts, list):
                drafts = [drafts]
            if any(draft in changed_drafts for draft in drafts):
                selected.add(stem)

    if changed_models:
        for stem, raw in documents.items():
            chain: list[str] = []
            base = raw.get("base")
            while base:
                if base in chain:
                    break
                chain.append(base)
                if base in changed_models:
                    selected.add(stem)
                    break
                parent = documents.get(base)
                if parent is None:
                    break
                base = parent.get("base")

    ids: list[str] = []
    for stem in sorted(selected):
        snap = root / "backend" / "arch" / "snapshots" / f"{stem}.json"
        if snap.is_file():
            ids.append(stem.replace("__", "/", 1))
    return ids


def model_ids_to_validate(paths: list[str], root: Path = ROOT) -> list[str]:
    """Compatibility name for the referer-aware CI selection."""
    return models_to_validate_for_paths(paths, root=root)


def _read_paths(path_file: str | None) -> list[str]:
    if path_file:
        text = Path(path_file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _classify(paths: list[str], added: list[str]) -> int:
    kind = "spec" if is_spec_only(paths, added) else "full"
    print(kind)
    return 0


def _parse_changed_yaml(paths: list[str], root: Path) -> None:
    import yaml
    for raw in paths:
        p = _norm(raw)
        if not p.endswith((".yaml", ".yml")):
            continue
        path = root / p
        if not path.is_file():
            continue
        yaml.safe_load(path.read_text(encoding="utf-8"))
        print(f"  parsed {p}")


def _run(paths: list[str], root: Path = ROOT) -> int:
    sys.path.insert(0, str(root))
    ids = models_to_validate_for_paths(paths, root=root)
    if not ids:
        missing = []
        for raw in paths:
            p = _norm(raw)
            if p.startswith("backend/arch/models/") and p.endswith((".yaml", ".yml")):
                stem = Path(p).stem
                snap = root / "backend" / "arch" / "snapshots" / f"{stem}.json"
                if not snap.is_file():
                    missing.append(p)
        if missing:
            print("spec YAML with no snapshot on disk:", file=sys.stderr)
            for p in missing:
                print(f"  {p}", file=sys.stderr)
            print("A contributed spec needs backend/arch/snapshots/<stem>.json.",
                  file=sys.stderr)
            return 1
        print("No snapshots in this diff; checking YAML parses.")
        _parse_changed_yaml(paths, root)
        return 0

    from backend.inference.validate import validate

    rc = 0
    for model_id in ids:
        print(f"→ validate {model_id}")
        result = validate(model_id)
        storage = result.get("model_storage_bytes") or {}
        diff = storage.get("diff")
        valid = bool(result.get("valid"))
        spec_name = result.get("spec_name") or "?"
        print(f"  spec={spec_name}  valid={valid}  bytes_diff={diff}")
        if valid:
            continue
        rc = 1
        for err in result.get("errors") or []:
            print(f"  - {err}", file=sys.stderr)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="run",
        choices=("classify", "run", "stored-spec-check"),
        help="classify prints 'spec' or 'full'; run validates changed snapshots; "
             "stored-spec-check prints 'yes' or 'no'")
    parser.add_argument(
        "--paths-file", default=None,
        help="Changed paths, one per line (stdin if omitted)")
    parser.add_argument(
        "--added-file", default=None,
        help="classify: paths the diff adds or renames to, one per line")
    args = parser.parse_args(argv)
    paths = _read_paths(args.paths_file)
    if args.command == "classify":
        added = _read_paths(args.added_file) if args.added_file else []
        return _classify(paths, added)
    if args.command == "stored-spec-check":
        print("yes" if requires_stored_spec_check(paths) else "no")
        return 0
    return _run(paths)


if __name__ == "__main__":
    sys.exit(main())
