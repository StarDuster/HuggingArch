"""Promote an op-proposal artifact into the registry — library form.

Both the CLI (`scripts/promote_op_proposal.py`) and the
`op_workers.review` route consume this. The pre-existing CLI logic lives
here so the reviewer can insert/pytest/revert without shelling out.

Pipeline:
  1. Read `backend/arch/ops_proposals/<op_name>/`
  2. Insert `patch.NEW_DIM_SYMBOLS` into `_CANONICAL_DIMS` in ops_base.py
  3. Append `patch.OP_REGISTRATIONS` to ops_registry.py
  4. Merge optional `dims.yaml` entries into the dim registry
     `backend/arch/dims.yaml`
  5. Insert optional `tests.py` methods into `TestNewPrimitivesRegistered`
     in `tests/backend/arch/test_multi_output_ops.py`
  6. Run pytest against the broadened target list
  7. Revert on test failure; archive to `_promoted/<op_name>/` on success

Anchors are hard-coded by design — if someone refactors the target
files we want a noisy failure, not a silent mis-insert.

Distinct from `backend/arch/promote.py`, which promotes draft
*components* (yaml) to the library tier.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import importlib.util
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from backend.arch.authored_yaml import AuthoredYaml
from backend.arch.dims import parse_row, validate_fragment


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROPOSALS_DIR = REPO_ROOT / "backend" / "arch" / "ops_proposals"
# _CANONICAL_DIMS (with the `# Misc` anchor) lives in ops_base.py; the
# register_op(...) blocks live in ops_registry.py alongside the _dim / _zero
# helpers a registration snippet references.
OPS_BASE_PATH = REPO_ROOT / "backend" / "arch" / "ops_base.py"
OPS_REGISTRY_PATH = REPO_ROOT / "backend" / "arch" / "ops_registry.py"
REGISTRY_PATH = REPO_ROOT / "backend" / "arch" / "dims.yaml"
TESTS_PATH = REPO_ROOT / "tests" / "backend" / "arch" / "test_multi_output_ops.py"

OPS_DIMS_ANCHOR = "        # Misc"
TESTS_CLASS_ANCHOR = "class TestNewPrimitivesRegistered(unittest.TestCase):"

# Broadened test target list for auto-merge. Catches breakage in
# engine/calibration paths a single-suite run would miss.
DEFAULT_PYTEST_TARGETS = [
    "tests/backend/arch/test_multi_output_ops.py",
    "tests/backend/inference/test_op_calibration.py",
]


@dataclass
class PromoteResult:
    """Outcome of `run_promote`. `ok=False` means engine failure (insert
    threw, snapshot/restore failed). `ok=True` covers all expected
    pipeline outcomes including 'pytest failed, reverted' — the system
    did its job, the proposal just didn't qualify."""
    ok: bool
    stage: str          # "read" | "insert" | "test" | "archive" | "done"
    message: str = ""
    error: str = ""
    pytest_stdout: str = ""
    pytest_stderr: str = ""
    artifacts: dict = field(default_factory=dict)


def _load_patch_module(patch_py: Path):
    """Load patch.py as a module so we can read its attributes."""
    spec = importlib.util.spec_from_file_location("op_proposal_patch", patch_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _insert_dims(ops_text: str, section_comment: str, symbols: list[str]) -> str:
    """Insert a new section above `# Misc` in _CANONICAL_DIMS.

    Mirrors existing section style:
        # <section comment>
        "<sym1>", "<sym2>", ...,
    """
    if not symbols:
        return ops_text
    idx = ops_text.find(OPS_DIMS_ANCHOR)
    if idx < 0:
        raise RuntimeError(
            f"Could not find OPS_DIMS_ANCHOR '{OPS_DIMS_ANCHOR}' in ops_base.py"
        )
    line_start = ops_text.rfind("\n", 0, idx) + 1
    indent = ops_text[line_start:idx]
    quoted = ", ".join(f'"{s}"' for s in symbols)
    new_section = f"{indent}# {section_comment}\n{indent}{quoted},\n"
    return ops_text[:line_start] + new_section + ops_text[line_start:]


def _insert_registration(ops_text: str, registration: str) -> str:
    """Append a `register_op(OpSpec(...))` block at EOF with exactly one
    blank-line separation."""
    if not registration.strip():
        return ops_text
    rstripped = ops_text.rstrip()
    return f"{rstripped}\n\n{registration.rstrip()}\n"


def _insert_registry(registry_text: str, entries_yaml: str) -> str:
    """Append new symbol entries to the dim registry.

    Appended rather than spliced at an anchor: `symbols:` is a mapping, so
    where an entry sits does not change what it says, and the section a new
    symbol belongs to is its own `group:` field. The old glossary was prose
    with a hardcoded `## Per-arch one-offs` anchor, and a proposal landing
    under the wrong heading was a documentation bug nothing could catch.

    A symbol the registry already names is a hard error. Two classifications
    for one symbol is the exact failure the registry exists to prevent, and
    a proposal is not the place to re-adjudicate one.
    """
    if not entries_yaml.strip():
        return registry_text
    new = load_yaml(entries_yaml) or {}
    if not isinstance(new, dict):
        raise RuntimeError(
            "dims.yaml artifact must be a mapping of symbol -> entry, got "
            f"{type(new).__name__}"
        )
    existing = load_yaml(registry_text) or {}
    known = set((existing.get("symbols") or {}))
    try:
        validate_fragment(new, known=known)
    except ValueError as e:
        raise RuntimeError(str(e)) from e
    body = yaml.dump(new, sort_keys=False, width=84, allow_unicode=True,
                     default_flow_style=False)
    merged = registry_text.rstrip("\n") + "\n" + textwrap.indent(body, "  ")

    # Appending assumes `symbols:` is the file's last top-level block, which
    # is true and is not self-evident. Re-parse rather than assume: a
    # reordered registry would otherwise turn the new entries into members of
    # whatever block happened to be last.
    landed = (load_yaml(merged) or {}).get("symbols") or {}
    astray = sorted(set(new) - set(landed))
    if astray:
        raise RuntimeError(
            f"appended entries {astray} did not land under `symbols:` in "
            f"dims.yaml — is `symbols:` still the last top-level block?"
        )
    return merged


def _insert_tests(tests_text: str, methods_src: str) -> str:
    """Indent free-standing `def test_*(self):` methods by 4 spaces and
    insert at end of `TestNewPrimitivesRegistered`."""
    if not methods_src.strip():
        return tests_text
    class_idx = tests_text.find(TESTS_CLASS_ANCHOR)
    if class_idx < 0:
        raise RuntimeError(
            f"Could not find TESTS_CLASS_ANCHOR in {TESTS_PATH.name}"
        )
    search_start = class_idx + len(TESTS_CLASS_ANCHOR)
    next_class_idx = tests_text.find("\nclass ", search_start)
    if next_class_idx < 0:
        next_class_idx = len(tests_text)
    insertion_point = next_class_idx
    while insertion_point > 0 and tests_text[insertion_point - 1] == "\n":
        insertion_point -= 1

    indented = textwrap.indent(methods_src.strip() + "\n", "    ")
    return (
        tests_text[:insertion_point]
        + "\n\n"
        + indented
        + tests_text[insertion_point:]
    )


def read_artifacts(op_dir: Path) -> dict:
    """Validate the artifact directory and read its contents."""
    patch_py = op_dir / "patch.py"
    registry_yaml = op_dir / "dims.yaml"
    tests_py = op_dir / "tests.py"
    if not patch_py.is_file():
        raise FileNotFoundError(f"missing {patch_py}")

    patch = _load_patch_module(patch_py)
    section_comment = getattr(patch, "DIM_SECTION_COMMENT", "")
    symbols = list(getattr(patch, "NEW_DIM_SYMBOLS", []) or [])
    registration = getattr(patch, "OP_REGISTRATIONS", "")

    registry_entries = ""
    if registry_yaml.is_file():
        registry_entries = registry_yaml.read_text(encoding="utf-8")

    tests_src = ""
    if tests_py.is_file():
        tests_src = tests_py.read_text(encoding="utf-8")

    return {
        "section_comment": section_comment,
        "symbols": symbols,
        "registration": registration,
        "registry_entries": registry_entries,
        "tests_src": tests_src,
    }


@dataclass(frozen=True)
class PromotedSymbols:
    """A model spec and the dim registry after its `dim_symbols` moved into
    the registry. `registry` is None when the spec declares none."""
    spec_yaml: str
    registry: "str | None"


def promote_spec_symbols(model_id: str, spec_yaml: str,
                         registry_text: str) -> PromotedSymbols:
    """Move a spec's own `dim_symbols:` rows into `registry_text` (the
    dims.yaml of the tree the PR is built on) and drop the block from the
    spec. Both texts keep their comments (`AuthoredYaml`).

    Per symbol, against that registry:
      * absent → appended, `used_by: <repo name>` (the part of the model id
        after `/`, the form existing rows use);
      * present with the same row — both through `_parse_symbol`, `used_by`
        ignored, `shardable_by` as a set → not appended; the repo name joins
        the existing row's `used_by`;
      * present with a different row → ValueError naming the fix.
    """
    doc = load_yaml(spec_yaml) or {}
    local = doc.get("dim_symbols") if isinstance(doc, dict) else None
    if not local:
        return PromotedSymbols(spec_yaml=spec_yaml, registry=None)
    if not registry_text.strip():
        raise ValueError(
            "the dim registry (backend/arch/dims.yaml) of the tree this PR is "
            "built on was not read, so this spec's dim_symbols cannot be "
            "promoted into it")
    registry = load_yaml(registry_text) or {}
    rows = dict(registry.get("symbols") or {})
    user = model_id.rsplit("/", 1)[-1]

    def comparable(name: str, entry) -> tuple:
        sym = parse_row(name, entry, registry)
        return (sym.kind, frozenset(sym.shardable_by), sym.group,
                sym.meaning.strip(), sym.config, sym.binding, sym.derived, sym.default, sym.at_most,
                sym.engine_supplied)

    for name, entry in local.items():
        name = str(name)
        mine = comparable(name, entry)
        if name not in rows:
            rows[name] = {**dict(entry), "used_by": user}
            continue
        if comparable(name, rows[name]) != mine:
            held = rows[name]
            raise ValueError(
                f"dim_symbols.{name}: backend/arch/dims.yaml now registers "
                f"{name!r} as kind {held.get('kind')}, shardable_by "
                f"{held.get('shardable_by') or []}, meaning: "
                f"{str(held.get('meaning', '')).strip()}. Delete this spec's "
                f"declaration and bind the registered symbol if it is the "
                f"same quantity; otherwise rename this spec's symbol.")
        users = [u.strip() for u in str(rows[name].get("used_by") or "").split(",")
                 if u.strip()]
        if user not in users:
            rows[name] = {**rows[name], "used_by": ", ".join([*users, user])}
    new_registry = AuthoredYaml(registry_text).rewrite({**registry, "symbols": rows})
    landed = (load_yaml(new_registry) or {}).get("symbols") or {}
    for name in local:
        comparable(str(name), landed[str(name)])
    stripped = {k: v for k, v in doc.items() if k != "dim_symbols"}
    return PromotedSymbols(spec_yaml=AuthoredYaml(spec_yaml).rewrite(stripped),
                           registry=new_registry)


def spliced_files(artifacts: dict) -> dict[str, str]:
    """Repo-relative path → new file text. Does not write.

    Used by the GitHub PR landing path so a serving process never mutates
    its own checkout. `run_promote` still writes, for local CLI `--apply`.
    """
    files: dict[str, str] = {}
    base_text = OPS_BASE_PATH.read_text(encoding="utf-8")
    new_base = _insert_dims(
        base_text, artifacts["section_comment"], artifacts["symbols"]
    )
    if new_base != base_text:
        files[str(OPS_BASE_PATH.relative_to(REPO_ROOT))] = new_base

    registry_ops = OPS_REGISTRY_PATH.read_text(encoding="utf-8")
    new_ops = _insert_registration(registry_ops, artifacts["registration"])
    if new_ops != registry_ops:
        files[str(OPS_REGISTRY_PATH.relative_to(REPO_ROOT))] = new_ops

    if artifacts["registry_entries"].strip():
        registry_text = REGISTRY_PATH.read_text(encoding="utf-8")
        files[str(REGISTRY_PATH.relative_to(REPO_ROOT))] = _insert_registry(
            registry_text, artifacts["registry_entries"]
        )

    if artifacts["tests_src"].strip():
        tests_text = TESTS_PATH.read_text(encoding="utf-8")
        files[str(TESTS_PATH.relative_to(REPO_ROOT))] = _insert_tests(
            tests_text, artifacts["tests_src"]
        )
    return files


def _apply(artifacts: dict) -> None:
    """Apply all four inserts. Reads from disk, writes to disk."""
    for rel, body in spliced_files(artifacts).items():
        (REPO_ROOT / rel).write_text(body, encoding="utf-8")


def _snapshot_files() -> dict[Path, str]:
    return {
        p: p.read_text(encoding="utf-8")
        for p in (OPS_BASE_PATH, OPS_REGISTRY_PATH, REGISTRY_PATH, TESTS_PATH)
    }


def _restore(snapshot: dict[Path, str]) -> None:
    for p, body in snapshot.items():
        p.write_text(body, encoding="utf-8")


def _run_tests(targets: list[str]) -> tuple[bool, str, str]:
    result = subprocess.run(
        ["uv", "run", "pytest", *targets, "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stdout, result.stderr


def run_promote(
    op_name: str,
    *,
    pytest_targets: list[str] | None = None,
    skip_tests: bool = False,
) -> PromoteResult:
    """Insert, test, and archive an op proposal.

    On any failure after inserting, files are reverted from snapshot.
    Returns a `PromoteResult` describing the terminal stage.
    """
    op_dir = PROPOSALS_DIR / op_name
    if not op_dir.is_dir():
        return PromoteResult(
            ok=False, stage="read",
            error=f"artifact dir not found: {op_dir}",
        )

    try:
        artifacts = read_artifacts(op_dir)
    except Exception as e:
        return PromoteResult(
            ok=False, stage="read", error=str(e),
        )

    snapshot = _snapshot_files()
    try:
        _apply(artifacts)
    except Exception as e:
        _restore(snapshot)
        return PromoteResult(
            ok=False, stage="insert", error=str(e),
            artifacts=artifacts,
        )

    if skip_tests:
        return PromoteResult(
            ok=True, stage="insert",
            message="insert applied; pytest skipped (NOT recommended)",
            artifacts=artifacts,
        )

    targets = pytest_targets or DEFAULT_PYTEST_TARGETS
    passed, stdout, stderr = _run_tests(targets)
    if not passed:
        _restore(snapshot)
        return PromoteResult(
            ok=True, stage="test",
            message="pytest failed; inserts reverted",
            pytest_stdout=stdout, pytest_stderr=stderr,
            artifacts=artifacts,
        )

    promoted_dir = PROPOSALS_DIR / "_promoted" / op_name
    promoted_dir.parent.mkdir(parents=True, exist_ok=True)
    if promoted_dir.exists():
        shutil.rmtree(promoted_dir)
    shutil.move(str(op_dir), str(promoted_dir))

    return PromoteResult(
        ok=True, stage="done",
        message=(
            f"promoted; artifacts archived at "
            f"{promoted_dir.relative_to(REPO_ROOT)}"
        ),
        pytest_stdout=stdout, pytest_stderr=stderr,
        artifacts=artifacts,
    )


def preview(op_name: str) -> dict:
    """Read-only inspection — CLI preview mode + review SSE use this."""
    op_dir = PROPOSALS_DIR / op_name
    if not op_dir.is_dir():
        raise FileNotFoundError(f"artifact dir not found: {op_dir}")
    return read_artifacts(op_dir)
