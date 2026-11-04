"""The corpus: every snapshot under `backend/arch/snapshots`, and what each
one resolves to.

A `CorpusModel` is one snapshot. It answers the questions every pin asks of
it (its config, its checkpoint metadata, the spec it resolves to, the files
that define that spec) and nothing about any pin. `tests/snapshots.py`
re-exports the directory and the shared loader from here, so a test and the
admitter read the corpus the same way.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
ARCH_DIR = ROOT / "backend" / "arch"
SNAPSHOT_DIR = ARCH_DIR / "snapshots"
MODELS_DIR = ARCH_DIR / "models"
COMPONENTS_DIR = ARCH_DIR / "components"
GOLDEN_DIR = ROOT / "tests" / "golden"
PINS_DIR = GOLDEN_DIR / "pins"


def _require_backend_from_root() -> None:
    """Refuse to measure this tree's snapshots with another tree's code.

    The pins read snapshots and goldens from `ROOT` but import `backend`
    from wherever `sys.path` finds it first. The venv's editable install
    puts the main checkout on `sys.path`, so a script run from a worktree
    (or from /tmp) silently measured the worktree's snapshots against the
    main checkout's specs and loader — a different, uncommitted tree."""
    import backend
    found = Path(backend.__file__).resolve().parents[1]
    if found != ROOT.resolve():
        raise ImportError(
            f"tools.corpus_pins reads the corpus under {ROOT} but imported "
            f"`backend` from {found}; put {ROOT} first on sys.path")


_require_backend_from_root()


@functools.cache
def shared_loader() -> Any:
    """The one `SpecLoader` over the repo tree, built once per process.

    Building a loader parses every component and model yaml (seconds each
    time). Over an unchanged tree a loader answers `match` / `get` /
    `component` the same whether it is fresh or has answered before. A
    caller that writes specs, attaches an overlay, loads a loose yaml or
    edits a loaded spec builds its own `SpecLoader()`.

    `match` still returns a fresh `ResolvedSpec` each call. What a caller
    over THIS loader shares is one step further on, in `CorpusModel`:
    `resolved()` / `program()` below."""
    from backend.arch.loader import SpecLoader
    return SpecLoader()


# -- what a process pays once --------------------------------------------
#
# WHY THESE ARE SHARED, AND WHAT HAD TO BE TRUE FIRST
#
# `shared_loader` stopped at the yaml. Everything after it was paid again by
# every caller, and over a test suite that is most of the bill: `match`
# hands back a NEW `ResolvedSpec` whose six memo slots start cold, so the
# first `iter_tree` on it expands the whole model graph again (136 ms for
# DeepSeek-V4-Flash), and the walk that follows is 900 ms of sympy. One
# model picked by one test to check one behaviour was resolved, expanded and
# walked from scratch dozens of times per run.
#
# Three facts make the sharing sound. The first two are held by
# `tests/meta/test_shared_corpus_layer.py`; the third turned out to be TRUE
# OF THE SHELL ONLY, so read it before trusting the word "frozen" anywhere
# in this file.
#
# 1. A `ResolvedSpec`'s facts are written once, in `__init__`. Nothing in
#    `backend/` or `tools/` assigns `spec`, `dim_values`, `config`,
#    `view_values`, `site`, `declared_by` or `checkpoint` afterwards; every
#    reader copies what it needs (`dict(resolved.dim_values)`).
# 2. Its six remaining slots (`_spec_tree_blocks_cache`,
#    `_spec_tree_ir_cache`, `_spec_tree_cache`, `_appended_cache`,
#    `_call_sites_cache`, `_sampled_head_cache`) are memos of those facts
#    and nothing else -- each is written by exactly one
#    `if slot is None: slot = compute(self)`. Sharing the resolution is
#    therefore sharing the memos, which is the point.
# 3. `Program` and `Deployment` are frozen dataclasses, and `Deployment`'s
#    builders (`at` / `during` / `on` / `with_dims` / `fused` /
#    `with_precision`) all `replace`, so a deployment is a value.
#    BUT `frozen` FREEZES THE SHELL. `Program.ir` is a tree of ordinary
#    mutable `SpecIRNode`s, so a frozen `Program` is not an immutable
#    graph, and a shared one is a graph every later holder can have
#    rewritten under them. Two tests did exactly that and the damage landed
#    on strangers: `test_eval_traffic`'s cache term reported 0 instead of
#    71303168 bytes, and `test_ep_as_count` before `test_program_moe`'s
#    corpus sweep turned the sweep red while the reverse order was green.
#
#    Neither test needed it. Both were asserting something about a SPEC —
#    that a graph declaring no `cache:` reads none, that a `repeat:` over a
#    routed expert bank is refused — and were producing it by editing a
#    resolved model instead of declaring it. Rewritten against specs of
#    their own, they assert the proposition they name and nothing mutable
#    is shared. The durable guarantee is a frozen `SpecIRNode` with the
#    walk building through a builder; that is engine work -- the walk builds
#    its nodes by assigning to them, so the builder has to come first -- and
#    is not done here, so what holds today
#    is that no test needs to write to a node — checked by
#    `tests/meta/test_shared_corpus_layer.py`, which reds if a file that
#    takes `shared_program` assigns to a node field.
#
# `Analysis` is NOT frozen, so sharing one carries an OBLIGATION rather
# than a guarantee: a consumer must treat a finished analysis as read-only,
# and one that will not must take a walk of its own. That rule is what is
# stated here, deliberately without describing how any particular caller
# obtains its private walk -- a sentence about mechanism has to be
# maintained against the mechanism, and six of this branch's review rounds
# were spent correcting sentences of exactly that kind. The rule holds
# however the route changes.
#
# `tests/meta/test_shared_corpus_layer.py` enforces it, and states its own
# limits; the two files that carry the exception say why at the call site
# (`tests.snapshots.own_walk`), which is the only place that cannot go
# stale.


def _shared_resolution_key(loader: Optional[Any]) -> bool:
    """Whether `loader` is the shared one, and so whether its resolutions may
    be shared. A caller holding its own loader (an overlay, a written spec, a
    loose yaml) is outside this boundary and gets a fresh resolution."""
    return loader is None or loader is shared_loader()


@functools.cache
def _resolution(stem: str) -> Any:
    resolved = shared_loader().match(
        _payload(SNAPSHOT_DIR / f"{stem}.json")["config"],
        model_id=stem.replace("__", "/", 1))
    if resolved is None:
        raise LookupError(f"{stem}: snapshot does not resolve to a spec")
    return resolved


@functools.cache
def _program(stem: str) -> Any:
    from backend.arch.eval.program import Program
    program = Program.of(_resolution(stem))
    _SHARED_PROGRAMS.add(id(program))
    return program


#: The ids of the programs handed out by `_program`, which are the only ones
#: `shared_walk` memoises a walk of. `functools.cache` holds every one for
#: the life of the process, so these ids cannot be reused by another object.
_SHARED_PROGRAMS: set = set()


def _walk_key(value: Any) -> Any:
    """`value` as something hashable and structural, so a deployment's knobs
    can key a memo.

    The rule is one sentence: a value is its fields. A frozen dataclass with
    generated equality holds all of its state in its fields, so it is keyed
    by them recursively (`Workload` has to be -- its `media` is a dict, and
    `hash` on it raises, which would make every `at(...)` a distinct key and
    the memo never hit). Mappings and sequences are keyed by their items.

    Anything else that cannot be hashed is keyed by IDENTITY. That direction
    is safe: a wrong identity key can only ever cost a cache MISS -- two
    equivalent objects built separately get two walks -- where a wrong
    structural key would hand back the wrong analysis. Nothing is dropped
    from the key, so a knob this does not understand cannot be ignored."""
    import dataclasses
    from collections.abc import Mapping, Sequence
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        params = value.__dataclass_params__
        if params.frozen and params.eq:
            return (type(value).__qualname__, tuple(
                _walk_key(getattr(value, field.name))
                for field in dataclasses.fields(value)))
    if isinstance(value, Mapping):
        return ("map", tuple(sorted(
            (str(key), _walk_key(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))):
        return ("seq", tuple(_walk_key(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return ("id", id(value))
    return value


def walk_key(deployment: Any) -> Any:
    """What makes two deployments the same walk.

    `Deployment` is a frozen dataclass whose builders (`at` / `during` /
    `on` / `with_dims` / `fused` / `with_precision`) all `replace`, and
    `analyze()` reads nothing but its fields -- so its fields ARE its
    identity as a walk. The key is therefore every field, read off
    `dataclasses.fields`, so a field added to `Deployment` later is keyed
    without anybody remembering to come here.

    `program` is the exception and the anchor: it is the GRAPH, not a knob,
    and the shared layer hands out exactly one per model
    (`CorpusModel.program`), so identity is its equality within a process.

    THE OBLIGATION THAT COMES WITH THAT, and it is on the CALLER: because
    the key contains `id(deployment.program)`, it identifies that program
    only while the program is ALIVE. A consumer that keeps keys and lets
    the programs go will have a dead program's id handed to the next
    object allocated there, and two different deployments will collapse
    onto one key. So anything that retains a `walk_key` must retain a
    reference to what it keyed.

    `WalkGroup` satisfies this by storing the deployment beside the
    analysis, which is stated where it stores it. `tools/suite_timing.py`
    did not, and its `distinct` column was silently allocator-dependent --
    one file read 85/63 alone, 85/74 in a batch and 85/85 once the
    deployments were retained. Nothing said the obligation existed, which
    is why it cost four rounds of re-measurement to find; it is written
    here now so the next consumer reads it rather than measures it."""
    import dataclasses
    return (id(deployment.program), tuple(
        _walk_key(getattr(deployment, field.name))
        for field in dataclasses.fields(deployment)
        if field.name != "program"))


class WalkGroup:
    """The walks a GROUP of tests shares, for as long as that group runs.

    One object, so the thing that holds analyses has a name and a lifetime
    instead of being a module-level dict somebody has to remember to clear.
    `of(deployment)` walks once per distinct deployment and hands the same
    analysis to every later caller in the group.

    THE LIFETIME IS THE POINT, and what bounds it is the CORPUS rather
    than the group. A group that sweeps every snapshot holds one analysis
    per model until it ends: measured at 237 MiB peak for all 62, against
    136 MiB for the same sweep dropping each one. So a group is bounded by
    the corpus, not by being small; what group scope buys is that the bound
    is paid once per module instead of once per run, which is why
    `tests/conftest.py` makes one group per test MODULE and no other -- for
    the structural reason recorded there, not for a memory one.

    WHY IT IS WORTH HAVING: `test_eval_fusion.py` asks for the same walk
    of the same model many times over and went 230 s -> 75 s. No other
    converted file does, and the per-file duplication table that used to
    sit here is deliberately gone -- it was corrected four times without
    anything about this class changing, and no decision rested on it (see
    the note in `tests/conftest.py`). `tools/suite_timing.py` measures it
    on demand.

    WHAT A MEMBER MUST NOT DO. `Analysis` is not frozen, so every member of
    a group gets the SAME object and must treat it as read-only. Two things
    write to one, and a consumer that does either needs its own walk
    (`tests.snapshots.own_walk`, which names the reason at the call site):

      * `Analysis.assumed()` writes the assumption ledger back when its
        scope closes (`self.assumptions = tuple(ledger.held)`). Called
        directly by `test_eval_symbolic_workload`, and inside
        `inference.serving` / `inference.step_eval` /
        `inference.activation_peak`, so a test driving those mutates too;
      * assigning a layer on the analysis --
        `test_layout_materializer` overwrites `input_references` to build a
        deliberately broken graph.

    `tests/meta/test_shared_corpus_layer.py` is what keeps the first of
    those from being reintroduced by reading the test files rather than by
    trusting this paragraph.

    There is no lock: a pytest worker is a process and no group is asked
    from two threads. Under `pytest-xdist` each worker builds its own
    groups, which composes with the sharing rather than conflicting with it
    -- though the default `--dist load` splits a module's tests across
    workers, so `--dist loadfile` is what matches a module-scoped group.
    """

    def __init__(self, what: str = "a group of tests") -> None:
        #: `{walk_key: (deployment, analysis)}`. The deployment is held
        #: beside its analysis so the program its key is the id of cannot be
        #: collected and have that id handed out again to another object.
        self._walks: Dict[Any, Tuple[Any, Any]] = {}
        self.what = what

    def __len__(self) -> int:
        return len(self._walks)

    def of(self, deployment: Any) -> Any:
        """`deployment.analyze()`, once for this group.

        The caller builds the deployment it wants, so this adds no
        vocabulary of its own for workloads, phases, meshes or precisions.

        A deployment over a program the shared layer did NOT hand out -- a
        spec the caller edited (`with_spec`), a loose yaml it loaded, an
        isolated component -- is simply walked and not kept. Nobody else can
        be holding that program, so there is no second caller to share with,
        and keeping it would pin a one-off graph and its analysis for the
        life of the group."""
        if id(deployment.program) not in _SHARED_PROGRAMS:
            return deployment.analyze()
        key = walk_key(deployment)
        if key not in self._walks:
            self._walks[key] = (deployment, deployment.analyze())
        return self._walks[key][1]

    def clear(self) -> int:
        """Drop every analysis held, returning how many. The resolutions and
        programs under them are not dropped: they are bounded by the corpus
        (62 models, ~57 MiB measured, no `Analysis` among them) and their
        reuse is the genuinely cross-file part."""
        held = len(self._walks)
        self._walks.clear()
        return held


#: The group a bare `shared_walk` lands in. `tests/conftest.py` installs a
#: module-scoped one for the duration of each test module and takes it away
#: again, which is what gives the walks their lifetime.
#:
#: WHY THIS IS AMBIENT AND NOT AN ARGUMENT. A pytest fixture reaches a test
#: by being named in its signature, and many of the files that share walks
#: are `unittest.TestCase` classes, whose methods cannot take one. Their
#: walks are built by module-level helpers (`_walked(name)`,
#: `_program(name)`) that no `cls` or fixture argument reaches either. So the
#: group is installed around the module rather than passed into it. A caller
#: outside a test session -- a pin, `tools/corpus_admit.py` -- finds no group
#: and is simply walked, which is the right answer for a one-shot script.
_CURRENT: Optional[WalkGroup] = None


def use_group(group: Optional[WalkGroup]) -> Optional[WalkGroup]:
    """Install `group` as the one `shared_walk` lands in; return the one it
    replaced, so a caller can put it back."""
    global _CURRENT
    previous, _CURRENT = _CURRENT, group
    return previous


def shared_walk(deployment: Any) -> Any:
    """This deployment's analysis, shared with the rest of the current group
    (`WalkGroup.of`). Walked, and not kept, when there is no group."""
    return (_CURRENT.of(deployment) if _CURRENT is not None
            else deployment.analyze())


@functools.cache
def _payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CorpusModel:
    """One snapshot of the corpus, named by its file stem (`org__model`)."""

    stem: str

    @property
    def filename(self) -> str:
        return f"{self.stem}.json"

    @property
    def model_id(self) -> str:
        return self.stem.replace("__", "/", 1)

    @property
    def path(self) -> Path:
        return SNAPSHOT_DIR / self.filename

    @property
    def payload(self) -> Dict[str, Any]:
        return _payload(self.path)

    @property
    def config(self) -> Dict[str, Any]:
        return self.payload["config"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.payload["metadata"]

    @property
    def header(self) -> Any:
        """The checkpoint's safetensors header (`hf.header`)."""
        from backend.hf.header import SafetensorsHeader
        return SafetensorsHeader.from_json(self.metadata["header"])

    def resolved(self, loader: Optional[Any] = None) -> Any:
        """The snapshot resolved against its spec. A snapshot that does not
        resolve raises: it is a broken corpus, not an absent model.

        Over the shared loader this is the process's ONE resolution of this
        snapshot, memos and all (see the note above `_resolution`). Pass
        your own loader to get a fresh one."""
        if _shared_resolution_key(loader):
            return _resolution(self.stem)
        resolved = loader.match(self.config, model_id=self.model_id)
        if resolved is None:
            raise LookupError(f"{self.stem}: snapshot does not resolve to a spec")
        return resolved

    def program(self) -> Any:
        """`Program.of` this snapshot's shared resolution, once per process.
        Free after the first call: a program does not copy the graph, and the
        expansion it reads is the resolution's memo."""
        return _program(self.stem)

    def spec_file(self, loader: Optional[Any] = None) -> Path:
        """The model yaml the snapshot resolves to."""
        return Path(self.resolved(loader).spec.source_path).resolve()

    def bundle(self, loader: Optional[Any] = None) -> Tuple[Path, ...]:
        """The files this model's admission brings: its snapshot, its model
        yaml and the drafts that yaml lists. A `base:` family it names is
        already in the corpus and is not part of it."""
        spec = self.spec_file(loader)
        doc = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
        return (self.path.resolve(), spec, *(
            (COMPONENTS_DIR / "drafts" / f"{name}.yaml").resolve()
            for name in doc.get("drafts") or ()))

    def spec_files(self, loader: Optional[Any] = None) -> Tuple[Path, ...]:
        """The files that define this model's spec: its model yaml, the
        `base:` chain under it, and the drafts any of them lists."""
        out: List[Path] = []
        path: Optional[Path] = self.spec_file(loader)
        while path is not None:
            out.append(path)
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            out.extend(COMPONENTS_DIR / "drafts" / f"{name}.yaml"
                       for name in doc.get("drafts") or ())
            base = doc.get("base")
            path = MODELS_DIR / f"{base}.yaml" if base else None
        return tuple(dict.fromkeys(out))


class Corpus:
    """The snapshots checked out under `SNAPSHOT_DIR`, in file-name order."""

    @staticmethod
    def stems() -> List[str]:
        return [path.stem for path in sorted(SNAPSHOT_DIR.glob("*.json"))]

    @classmethod
    def models(cls) -> List[CorpusModel]:
        return [CorpusModel(stem) for stem in cls.stems()]

    @staticmethod
    def component_files() -> List[Path]:
        return sorted(COMPONENTS_DIR.rglob("*.yaml"))


def relative(path: Path) -> str:
    """`path` relative to `backend/arch`, the form the pins key files by."""
    return str(Path(path).resolve().relative_to(ARCH_DIR))
