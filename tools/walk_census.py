"""Count how often a test session asks for the same answer twice.

Analysing a model means resolving it, building a `Program` and walking it.
Resolution is memoised (`tools.corpus_pins.corpus.shared_loader`); the walk is
not. Most test files pick one or two real models to check one behaviour and
walk each from scratch, so the suspicion is that a directory pays for the same
walk many times over.

This plugin wraps `Deployment.analyze`, attributes every call to the test file
that was running, and prints, per file, how many calls it made against how many
DISTINCT deployments it made them for. The gap is what a file-scoped fixture
can remove, split into the duplicates one test makes inside what it drives
(dup/within, no fixture can reach those) and the duplicates between a file's
tests (dup/across, exactly what a fixture removes). The gap between a file's
distinct count and the session's is what a conftest fixture can remove.

WHAT IT FOUND, so nobody has to guess again. Downstream of `arch`, the walk is
mostly NOT where the time goes, and the measurement has to be taken at the
level the tests actually drive:

  tests/backend/service     42 walks in the whole directory, 7.5 s. Nothing.
  tests/backend/rendering   25 s of walks in 318 s — but 280 s inside the
                            `/inference/compute-graph` handler, asked 57 times
                            for 31 distinct requests by ONE file.
  tests/backend/playground  235 s of walks in 902 s, and of 1602 duplicate
                            walks, 988 are repeats a SINGLE test makes inside
                            the code it drives. Taken at `bench_component`,
                            `test_module_bench` is 101 calls for 82 distinct.
  tests/backend/inference   walks are 9% of the directory. Taken at
                            `pipeline.analyze` over its sixteen heaviest
                            files, 975 of 1192 seconds — and only 142 of 666
                            of those calls are duplicates. Two argument sets
                            out of 200 are wanted by more than one file.

So `--walk-census-wrap` is not a convenience: for these directories it is the
only setting that measures the right thing.

Run it over a directory:

    ./.venv/bin/python -m pytest tests/backend/inference -p tools.walk_census -q

Four knobs:

    --walk-census-top=N     show only the N files with the most duplicate walks
    --walk-census-keys=N    also show the N most-repeated argument sets
    --walk-census-json=PATH write the whole table as JSON
    --walk-census-wrap=M:A  census `A` of module `M` instead of the walk, keyed
                            on its arguments — the same question one level up,
                            for a test file that drives an endpoint (an answer
                            that holds a walk inside it, and repeats for free
                            once shared)

The deployment key is a repr of everything `Deployment.analyze` reads
(`dataclasses.fields`) with the workload reduced to the symbolic form
`analyze` itself reduces it to, so two calls that differ only in batch size or
token counts — which `analyze` answers from one walk and a `Point` — count as
one deployment, as they should.
"""
from __future__ import annotations

import collections
import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class _FileCensus:
    """One test file's walks, and each of its tests' separately.

    A file's duplicates split in two, and only one half is the file's to
    remove. A walk a SINGLE test makes twice is repeated inside the code
    that test drives — no fixture of the file's can reach it. A walk two of
    the file's tests each make is the file paying twice for one answer, and
    that is what a module- or class-scoped fixture removes."""

    calls: int = 0
    seconds: float = 0.0
    #: Wall clock spent running this file's tests, so the census is also the
    #: before/after table: what a file costs, and how much of it is the walk.
    wall: float = 0.0
    keys: collections.Counter = field(default_factory=collections.Counter)
    #: nodeid -> that test's own keys.
    tests: Dict[str, collections.Counter] = field(default_factory=dict)

    @property
    def distinct(self) -> int:
        return len(self.keys)

    @property
    def duplicates(self) -> int:
        return self.calls - self.distinct

    @property
    def within_a_test(self) -> int:
        """Duplicates a single test makes — inside what it drives."""
        return sum(sum(keys.values()) - len(keys) for keys in self.tests.values())

    @property
    def across_tests(self) -> int:
        """Duplicates between the file's tests — a fixture's to remove."""
        return self.duplicates - self.within_a_test

    def row(self) -> Dict[str, object]:
        return {"wall": round(self.wall, 2), "calls": self.calls,
                "distinct": self.distinct, "duplicates": self.duplicates,
                "within_a_test": self.within_a_test,
                "across_tests": self.across_tests,
                "seconds": round(self.seconds, 2)}


class WalkCensus:
    """The session's census, and the wrapper that fills it."""

    def __init__(self) -> None:
        self.files: Dict[str, _FileCensus] = collections.defaultdict(_FileCensus)
        self.current: Optional[Tuple[str, str]] = None
        self.session_keys: collections.Counter = collections.Counter()
        self._unwrap = None
        self.label = "model walks"

    # -- the key ------------------------------------------------------------

    @staticmethod
    def key_of(deployment) -> str:
        """What this deployment walks, as a string.

        The workload is taken in the symbolic form `analyze` reduces it to,
        so deployments that differ only in their counts share a key."""
        try:
            over, _point, _free = deployment._over_counts()
        except Exception:
            over = deployment
        parts: List[str] = []
        for f in fields(over):
            value = getattr(over, f.name)
            if f.name == "program":
                parts.append(f"program={_program_key(value)}")
            else:
                parts.append(f"{f.name}={value!r}")
        return "|".join(parts)

    # -- wrapping -----------------------------------------------------------

    def install(self, target: Optional[str] = None) -> None:
        """Census `Deployment.analyze`, or the `module:attr` named by
        `target` keyed on that call's arguments."""
        if target is None:
            from backend.arch.eval import transforms
            self._wrap(transforms.Deployment, "analyze",
                       lambda args, kwargs: self.key_of(args[0]))
            return
        module_name, _, attr = target.partition(":")
        if not attr:
            raise ValueError(
                f"--walk-census-wrap wants module:attr, got {target!r}")
        import importlib
        self.label = target
        owner = importlib.import_module(module_name)
        for part in attr.split(".")[:-1]:
            owner = getattr(owner, part)
        self._wrap(owner, attr.split(".")[-1], _arguments_key)

    def _wrap(self, owner, name: str, key_of) -> None:
        census = self
        original = getattr(owner, name)

        def censused(*args, **kwargs):
            key = key_of(args, kwargs)
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                census.record(key, time.perf_counter() - started)

        censused.__doc__ = original.__doc__
        censused.__name__ = getattr(original, "__name__", name)
        setattr(owner, name, censused)
        self._unwrap = lambda: setattr(owner, name, original)

    def remove(self) -> None:
        if self._unwrap is not None:
            self._unwrap()
            self._unwrap = None

    def record(self, key: str, seconds: float) -> None:
        self.session_keys[key] += 1
        where, nodeid = self.current or ("<outside a test>", "")
        entry = self.files[where]
        entry.calls += 1
        entry.seconds += seconds
        entry.keys[key] += 1
        entry.tests.setdefault(nodeid, collections.Counter())[key] += 1

    # -- reporting ----------------------------------------------------------

    @property
    def calls(self) -> int:
        return sum(e.calls for e in self.files.values())

    @property
    def seconds(self) -> float:
        return sum(e.seconds for e in self.files.values())

    def table(self) -> List[Tuple[str, _FileCensus]]:
        return sorted(self.files.items(),
                      key=lambda kv: (-kv[1].duplicates, -kv[1].calls,
                                      -kv[1].wall, kv[0]))

    def hottest(self, limit: int) -> List[Tuple[str, int, int]]:
        """The most-repeated arguments, with how many files asked for them.

        A key many FILES ask for is what a session-scoped conftest fixture
        can cover; a key one file asks for many times is that file's own
        fixture."""
        askers: Dict[str, int] = collections.Counter()
        for entry in self.files.values():
            for key in entry.keys:
                askers[key] += 1
        return [(key, count, askers[key])
                for key, count in self.session_keys.most_common(limit)]

    def payload(self) -> Dict[str, object]:
        return {
            "wall": round(sum(e.wall for e in self.files.values()), 2),
            "calls": self.calls,
            "distinct": len(self.session_keys),
            "duplicates": self.calls - len(self.session_keys),
            "seconds": round(self.seconds, 2),
            "files": {name: entry.row() for name, entry in self.table()},
            "hottest": [{"key": key, "calls": calls, "files": files}
                        for key, calls, files in self.hottest(200)],
        }


def _arguments_key(args, kwargs) -> str:
    """One call's arguments, by content.

    An object's default `repr` carries its ADDRESS, so two calls that pass
    equivalent resolutions would never count as one; `_stable` names what
    each argument IS instead."""
    return "|".join([*(_stable(a) for a in args),
                     *(f"{name}={_stable(value)}"
                       for name, value in sorted(kwargs.items()))])


def _stable(value, depth: int = 4) -> str:
    """One argument, named without ever `repr`ing an unknown object.

    `repr` is only safe on things whose size is known. An argument of a
    serving call can be a walked analysis holding the whole graph, and the
    first version of this censused `pipeline.analyze` by `repr`ing its
    arguments and was killed for memory. So: scalars by their repr, a
    dataclass by its fields, a mapping or sequence by its items, and
    anything else by what it resolves (`_resolution_key`) or by its type —
    which costs the census nothing worse than telling two of them apart it
    could not tell apart anyway."""
    import dataclasses
    from collections.abc import Mapping, Sequence
    if value is None or isinstance(value, (bool, int, float, complex, str,
                                           bytes)):
        return repr(value)
    if depth <= 0:
        return f"<{type(value).__name__}>"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return "%s(%s)" % (type(value).__name__, ",".join(
            f"{f.name}={_stable(getattr(value, f.name), depth - 1)}"
            for f in dataclasses.fields(value)))
    if isinstance(value, Mapping):
        return "{%s}" % ",".join(
            f"{key!r}:{_stable(item, depth - 1)}"
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0])))
    if isinstance(value, (set, frozenset)):
        return "{%s}" % ",".join(sorted(_stable(i, depth - 1) for i in value))
    if isinstance(value, Sequence):
        return "[%s]" % ",".join(_stable(item, depth - 1) for item in value)
    return _resolution_key(value)


def _resolution_key(value) -> str:
    """What an object with no repr of its own describes: the model it
    resolves, the dimensions it resolved to and the call it isolates. An
    object that answers none of those is named by its type, which keeps the
    census honest (it cannot tell two of them apart) rather than wrong."""
    resolved = getattr(value, "resolved", value)
    spec = getattr(resolved, "spec", None)
    name = getattr(spec, "name", None)
    if name is None:
        return f"<{type(value).__name__}>"
    dims = sorted((getattr(resolved, "dim_values", None) or {}).items())
    return f"<{type(value).__name__} {name} {getattr(resolved, 'site', None)!r} {dims}>"


def _program_key(program) -> str:
    """What a `Program` walks, by content rather than by object identity.

    `SpecLoader.match` hands every caller a fresh `ResolvedSpec`, so two
    tests that resolve the same snapshot hold different objects describing
    the same model. The question this census asks is whether one walk could
    have served both, so the key is the spec's name, the dimensions it
    resolved to, the call site it isolates (a component-only spec) and the
    schedule — everything that decides what the walk produces."""
    resolved = getattr(program, "resolved", None)
    spec = getattr(resolved, "spec", None)
    name = getattr(spec, "name", None) or type(program).__name__
    dims = sorted((getattr(resolved, "dim_values", None) or {}).items())
    site = getattr(resolved, "site", None)
    return f"{name}/{getattr(program, 'schedule', None)}/{site!r}/{dims}"


# -- the plugin -------------------------------------------------------------

def pytest_addoption(parser) -> None:
    group = parser.getgroup("walk-census")
    group.addoption("--walk-census-top", type=int, default=25,
                    help="show only the N files with the most duplicate walks")
    group.addoption("--walk-census-json", default=None,
                    help="write the census to this path as JSON")
    group.addoption("--walk-census-keys", type=int, default=0,
                    help="also show the N most-repeated argument sets, and "
                         "how many files asked for each")
    group.addoption("--walk-census-wrap", default=None, metavar="MODULE:ATTR",
                    help="census this callable, keyed on its arguments, "
                         "instead of Deployment.analyze")


#: The session's census. `pytest_runtest_logreport` is handed a report and no
#: config, so the one census of the one session is held here as well as on
#: the config that owns it.
_CENSUS: List[Optional[WalkCensus]] = [None]


def pytest_configure(config) -> None:
    census = WalkCensus()
    census.install(config.getoption("--walk-census-wrap"))
    config._walk_census = _CENSUS[0] = census


def pytest_unconfigure(config) -> None:
    census = getattr(config, "_walk_census", None)
    if census is not None:
        census.remove()
    _CENSUS[0] = None


def pytest_runtest_protocol(item, nextitem):
    census = getattr(item.config, "_walk_census", None)
    if census is not None:
        census.current = (item.nodeid.split("::")[0], item.nodeid)
    return None


def pytest_runtest_logreport(report):
    """Each file's wall clock, from the same durations `--durations` reports
    (setup, call and teardown of every one of its tests)."""
    census = _CENSUS[0]
    if census is not None:
        census.files[report.nodeid.split("::")[0]].wall += report.duration


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    census = getattr(config, "_walk_census", None)
    if census is None or not census.files:
        return
    write = terminalreporter.write_line
    write("")
    write("%s: %d calls for %d distinct arguments (%d duplicates, %.1f s "
          "inside, over %.1f s of tests)"
          % (census.label, census.calls, len(census.session_keys),
             census.calls - len(census.session_keys), census.seconds,
             sum(e.wall for e in census.files.values())))
    write("%-50s %8s %6s %8s %11s %10s %8s"
          % ("file", "wall", "calls", "distinct", "dup/across", "dup/within",
             "seconds"))
    for name, entry in census.table()[:config.getoption("--walk-census-top")]:
        write("%-50s %8.1f %6d %8d %11d %10d %8.2f"
              % (name[-50:], entry.wall, entry.calls, entry.distinct,
                 entry.across_tests, entry.within_a_test, entry.seconds))
    keys = config.getoption("--walk-census-keys")
    if keys:
        write("")
        write("%6s %6s  %s" % ("calls", "files", "arguments"))
        for key, calls, askers in census.hottest(keys):
            write("%6d %6d  %s" % (calls, askers, key[:150]))
    path = config.getoption("--walk-census-json")
    if path:
        Path(path).write_text(json.dumps(census.payload(), indent=2),
                              encoding="utf-8")
        write(f"census written to {path}")
