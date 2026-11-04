"""Every closed per-model set in the tests has an owner in the registry.

`tools/corpus_pins` registers the pins (closed tables a new model extends),
the corpus checks, the decisions a person owns, and the curated subsets that
do not grow with the corpus. A table that is none of those is a table the
admitter cannot see, and the first model added after it turns main red. These
checks turn the common ways such a table is written into "register it or go
red". They are a ratchet over syntax, not a proof; what they cannot see is
listed in `tools/corpus_pins/curated.py` and left to review.
"""
from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.corpus_pins import ADMIT_PYTHON, REGISTRY
from tools.corpus_pins import curated
from tools.corpus_pins.pin import Pin
from tools.corpus_pins.pins.readme_counts import README_COUNTS

GOLDEN = PROJECT_ROOT / "tests" / "golden"


def _owner_modules(entries) -> set:
    return {test.split("::", 1)[0] for entry in entries for test in entry.owner_tests}


PIN_MODULES = _owner_modules(REGISTRY.pins)
CHECK_MODULES = _owner_modules(REGISTRY.checks)
CURATED_MODULES = {entry.module for entry in REGISTRY.curated}
WALKING = [path for path in curated.test_module_paths() if curated.walks_corpus(path)]


def _golden_owners():
    """`{file: {owner}}`: a pin or a decision by id; curated sets as one owner
    (the lane cohort's two readers share one golden)."""
    owners = {}
    for pin in REGISTRY.pins:
        for artifact in pin.artifacts():
            owners.setdefault(artifact, set()).add(pin.id)
    for decision in REGISTRY.decisions:
        for artifact in decision.artifacts:
            owners.setdefault(artifact, set()).add(decision.id)
    for entry in REGISTRY.curated:
        if entry.artifact:
            owners.setdefault(entry.artifact, set()).add("curated")
    return owners


def _ownership_problems(files):
    owners = _golden_owners()
    return ([f"{f}: no owner" for f in files if f not in owners]
            + [f"{f}: owners {sorted(owners[f])}" for f in files
               if f in owners and len(owners[f]) != 1])


def test_every_golden_file_has_exactly_one_owner():
    files = sorted(str(p.relative_to(PROJECT_ROOT)) for p in GOLDEN.rglob("*") if p.is_file())
    assert _ownership_problems(files) == []


def test_every_script_golden_is_compared_by_a_test():
    """A golden that only a script compares (gate.sh, a workflow) drifts on
    main without a pull request going red."""
    texts = [PROJECT_ROOT / "scripts" / "gate.sh",
             *sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yml"))]
    referenced = {m for path in texts
                  for m in re.findall(r"tests/golden/[\w./-]+", path.read_text())}
    by_artifact = {a: pin for pin in REGISTRY.pins for a in pin.artifacts()}
    loose = sorted(r for r in referenced
                   if r not in by_artifact or not by_artifact[r].owner_tests)
    assert loose == [], f"goldens only a script compares: {loose}"


def test_every_corpus_walking_module_is_claimed():
    claimed = PIN_MODULES | CHECK_MODULES | CURATED_MODULES
    loose = sorted(curated.module_name(p) for p in WALKING
                   if curated.module_name(p) not in claimed)
    assert loose == [], (
        "test modules that walk the corpus with no pin, check or curated set "
        f"registered for them in tools/corpus_pins: {loose}")


def test_corpus_literals_are_exactly_the_manifest():
    """A module- or class-level container naming a snapshot is a closed
    per-model set. Each is in the manifest, and each manifest row is a
    registered curated set; regenerate with
    `python -m tools.corpus_pins.write_manifest` and register the new set."""
    hits = curated.literal_hits(curated.module_paths(), curated.corpus_identifiers())
    scanned = curated.render_manifest([hit.row("curated") for hit in hits])
    assert scanned.decode() == curated.MANIFEST.read_text(encoding="utf-8")
    unregistered = sorted(f"{h.module}:{h.name}" for h in hits
                          if curated.find_curated(h.module, h.name) is None)
    assert unregistered == []


def test_numbers_in_corpus_walking_modules_are_classified():
    """A module-level number in a module that walks the corpus is either a
    pin's data or a parameter with a reason; a global count pinned as a bare
    integer (how the refusal pin used to be kept) would otherwise escape
    every per-model check."""
    known = ({(p.module, p.name) for p in curated.PARAMETERS}
             | {(m, n) for m, n, _ in curated.PIN_DATA}
             | {(c.module, c.name) for c in REGISTRY.curated})
    loose = sorted(f"{curated.module_name(path)}:{name}" for path in WALKING
                   for name in curated.numeric_constants(path)
                   if (curated.module_name(path), name) not in known)
    assert loose == [], f"unclassified numbers in corpus-walking modules: {loose}"


def test_built_identifiers_declare_where_their_keys_come_from():
    declared = curated.DYNAMIC_KEY_SOURCES
    building = {curated.module_name(p) for p in WALKING if curated.dynamic_identifiers(p)}
    assert sorted(building - set(declared)) == []
    assert sorted(set(declared) - building) == []
    wrong = sorted(
        module for module, source in declared.items()
        if not (source == "constant" and module in CURATED_MODULES)
        and not (source == "corpus" and module in PIN_MODULES | CHECK_MODULES))
    assert wrong == [], (
        "a module whose keys come from a constant needs a curated set; one "
        f"whose keys come from the corpus needs a pin or a check: {wrong}")


@pytest.mark.parametrize("pin", [p for p in REGISTRY.pins if isinstance(p, Pin)],
                         ids=lambda p: p.id)
def test_every_pin_file_round_trips(pin):
    assert pin.round_trips(pin.read()), f"{pin.artifact} does not round-trip"


def test_the_readme_cells_each_read_exactly_once():
    _values, problems = README_COUNTS.cells(README_COUNTS.texts())
    assert problems == []


def test_every_owner_test_is_collected():
    owners = sorted({t for entry in (*REGISTRY.pins, *REGISTRY.checks)
                     for t in entry.owner_tests})
    modules = sorted({t.split("::", 1)[0] for t in owners})
    out = subprocess.run(
        # `-m ""`: an owner deselected by default (`regression`) is still a
        # test pytest collects.
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:randomly",
         "-p", "no:cacheprovider", "-m", "", *modules],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    collected = set(out.stdout.split())
    missing = [t for t in owners
               if not any(c == t or c.startswith((t + "[", t + "::"))
                          for c in collected)]
    assert missing == [], f"owner tests pytest does not collect: {missing}\n{out.stderr[-2000:]}"


def test_the_registry_imports_without_pytest():
    code = textwrap.dedent("""
        import sys
        sys.modules["pytest"] = None
        import tools.corpus_pins
        import tools.corpus_admit
    """)
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr[-2000:]


def test_admit_python_is_the_ci_and_image_interpreter():
    installs = {(path.name, version) for path in
                sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yml"))
                for version in re.findall(r"uv python install (\S+)", path.read_text())}
    images = set(re.findall(r"^FROM python:(\S+?)-slim",
                            (PROJECT_ROOT / "Dockerfile").read_text(), re.M))
    assert installs and images
    assert {v for _, v in installs} == {ADMIT_PYTHON}, installs
    assert images == {ADMIT_PYTHON}, images


# ── the detectors fire ────────────────────────────────────────────────────

def _module(tmp_path, body: str) -> Path:
    path = tmp_path / "test_mutant.py"
    path.write_text(textwrap.dedent(body))
    return path


def test_a_write_needs_the_base_uv_lock():
    from tools.corpus_pins.env import Environment, lock_sha256_at
    env = Environment(python=ADMIT_PYTHON, uv_lock_sha256="a" * 64,
                      dists_sha256="d", locked_dists_sha256="d")
    assert env.write_refusals("a" * 64) == []
    assert env.write_refusals("b" * 64) == ["uv.lock differs from the base's uv.lock"]
    assert env.write_refusals(None) == ["the base's uv.lock cannot be read"]
    assert re.fullmatch(r"[0-9a-f]{64}", lock_sha256_at("HEAD") or "")
    assert lock_sha256_at("0" * 40) is None


def test_a_dict_naming_one_snapshot_is_a_literal_hit(tmp_path):
    stem = curated.corpus_identifiers().pop()
    path = _module(tmp_path, f"CASES = {{{stem!r}: 1}}\n")
    assert [h.identifiers for h in curated.literal_hits([path], curated.corpus_identifiers())] == [(stem,)]


def test_a_list_naming_only_a_candidate_is_a_literal_hit(tmp_path):
    """The scan reads the identifiers of the tree it runs on, so on a tree
    with the candidate's snapshot its id alone is a hit."""
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "org__candidate.json").write_text("{}")
    path = _module(tmp_path, 'MODELS = ["org/candidate"]\n')
    ids = curated.corpus_identifiers(snapshots)
    assert [h.name for h in curated.literal_hits([path], ids)] == ["MODELS"]


def test_an_unregistered_number_in_a_corpus_module_is_found(tmp_path):
    path = _module(tmp_path, """
        import pytest
        pytestmark = pytest.mark.regression
        PINNED_TOTAL = 197
    """)
    assert curated.walks_corpus(path)
    assert curated.numeric_constants(path) == ["PINNED_TOTAL"]


def test_an_unowned_golden_file_is_found():
    assert _ownership_problems(["tests/golden/new_table.json"]) == [
        "tests/golden/new_table.json: no owner"]


def test_the_scan_reads_class_level_literals(tmp_path):
    stem = curated.corpus_identifiers().pop()
    path = _module(tmp_path, f"class T:\n    MODELS = ({stem!r},)\n")
    assert [h.name for h in curated.literal_hits([path], curated.corpus_identifiers())] == ["T.MODELS"]


def test_detectors_ignore_function_locals(tmp_path):
    stem = curated.corpus_identifiers().pop()
    path = _module(tmp_path, f"def f():\n    return [{stem!r}]\n")
    assert curated.literal_hits([path], curated.corpus_identifiers()) == []


def test_the_pinned_fact_lane_has_a_comparer_for_every_written_file():
    """The nightly `pins` lane is derived from this registry, so an entry that
    owns a file and names nothing to compare it is silently unguarded.

    That is the hole the night of 2026-10-29 fell through at a larger scale:
    the four measuring goldens each HAD a comparer and no lane ran it, which
    `.github/workflows/pins.yml` now fixes. This check covers the other half
    — an entry whose file nothing claims to compare at all — because the lane
    cannot run a test that was never named, and a lane that silently runs 24
    targets instead of 25 looks exactly as green.
    """
    from tools.corpus_pins.lane import lane_targets

    unguarded = [f"{pin.id} owns {pin.artifact} and names no owner_tests"
                 for pin in REGISTRY.pins
                 if any(pin.artifacts()) and not pin.owner_tests]
    unguarded += [f"curated {entry.name} owns {entry.artifact} and names no module"
                  for entry in REGISTRY.curated
                  if entry.artifact and not entry.module]
    assert unguarded == []

    targets = lane_targets()
    absent = [t for t in targets
              if not (PROJECT_ROOT / t.split("::", 1)[0]).is_file()]
    assert absent == [], f"lane targets whose file is gone: {absent}"

    # The control: the surface the lane was built for must be in it. Taken
    # from the pin's own declaration, not re-listed here — so this fails if
    # the derivation stops reaching pins, and cannot pass by agreeing with a
    # copy of the answer.
    sol = REGISTRY.pin("P2")
    assert set(sol.owner_tests) <= set(targets)
