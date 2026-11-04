"""How much a test file pays to get its models, counted rather than timed.

Wall clock on a shared box is not evidence: the same file measured twice an
hour apart on this machine came back 523 s and 312 s because other sessions
were building. So this plugin counts the operations a test cannot make
cheaper and can only stop repeating. Every column, and what each one is:

  resolutions  `SpecLoader.resolve` calls -- one checkpoint config against
               one spec. ~2 ms each; the cost is not the call, it is that
               every call hands back a NEW `ResolvedSpec` whose spec-tree
               memos (`_spec_tree_ir_cache` and the five beside it) start
               cold, so the next column is paid again.
  trees        cold `iter_tree` builds -- the expansion of the whole model
               graph. ~140 ms for DeepSeek-V4-Flash. A warm call is
               memoised on the resolution and is not counted.
  asks         calls to `Deployment.analyze` and to `WalkGroup.of`, which
               are the two ways a walk is normally requested. NOT every
               request: a caller that drives `Evaluator(...).run(...)`
               itself goes through neither.
  distinct     how many of the asks were for genuinely different
               deployments, keyed by `corpus.walk_key`. Reported as `-` in
               a tree that has no `walk_key` (before the shared layer),
               because there is no sound way to count it there.
  walks        `Evaluator.run` calls -- one symbolic evaluation of one
               program at one binding. ~900 ms for DeepSeek-V4-Flash, and
               it is where the suite's time actually is (sympy, 2.4 s of a
               3.2 s profile).

Each is attributed to the test file that was running. Seconds are reported
beside the counts, and the load average beside the total, because a reader
wants them -- but the load on this box swings the times by 3-5x, so the
counts are the only part worth comparing between runs.

    python -m tools.suite_timing "-m=not nothing" --out table.json <files>

(`-m` needs the `=` spelling: this script's own argument parser would
otherwise swallow it.)

No table of its output is committed. One was, and it was corrected four
times in a night while the code it measured never changed; `tests/conftest.py`
records why that number is no longer published. Run this when a specific
decision needs it, and read what `asks` does and does not instrument first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

#: what is counted, and the seconds column each count sums into
KINDS = {"resolutions": "resolve_s", "trees": "tree_s", "walks": "walk_s",
         "asks": "ask_s"}


class Tally:
    """Counts and seconds per test file, for the paid operations.

    `asks` counts calls to `Deployment.analyze` and `WalkGroup.of`; `walks`
    counts `Evaluator.run`s that actually happened. WHERE EVERY WALK IS
    REQUESTED THROUGH ONE OF THOSE TWO, the gap between them is what the
    sharing removed, and `asks - distinct` is the headroom that is left.

    Neither statement is unconditional, and the condition is not a corner
    case. A caller that drives `Evaluator(...).run(...)` itself goes
    through neither entry point, so it raises `walks` without raising
    `asks` -- measured on a file with no sharing in use at all:
    `test_eval_transforms.py::TestPhaseAlsoChangesTheGraph::
    test_the_weights_are_the_same_weights` reports `asks=0, walks=2`. So
    `walks` can exceed `asks` with or without a shared layer, and the two
    columns being unequal is not by itself evidence of anything."""

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {**{name: 0 for name in KINDS},
                     **{column: 0.0 for column in KINDS.values()},
                     "distinct": 0, "wall_s": 0.0, "tests": 0})
        self.keys: Dict[str, set] = defaultdict(set)
        #: The deployments whose keys are in `self.keys`, held for exactly
        #: the reason `corpus.walk_key` states: the key contains
        #: `id(deployment.program)`, so it only identifies that program
        #: while the program is alive. Without this list a temporary
        #: `Program` is collected, its id is handed to the next one, and
        #: two genuinely different asks collapse into one `distinct` row.
        #: It is allocator-dependent, so it under-reported `distinct` by a
        #: different amount every run: `test_include_aux_weights` read
        #: 85/63 alone, 85/74 in a batch, and 85/85 once retained.
        self.keyed_deployments: List[Any] = []
        self.current = "<collection>"

    #: True while a `WalkGroup.of` call is on the stack. The group calls
    #: `Deployment.analyze` itself on a miss, and both are wrapped, so
    #: without this a miss counted as two asks and one walk -- the tool
    #: over-reporting the very quantity it exists to report.
    inside_group: bool = False

    def charge(self, kind: str, seconds: float) -> None:
        row = self.rows[self.current]
        row[kind] += 1
        row[KINDS[kind]] += seconds

    #: False when this tree has no `corpus.walk_key` to key a deployment
    #: by -- a tree from before the shared layer, which is exactly the tree
    #: a `before` measurement runs in. `distinct` is then UNAVAILABLE and
    #: printed as `-`, not 0: the first version of this reported 0, which
    #: read as "no walk was distinct" and so as infinite headroom.
    keyed: bool = True

    def note_key(self, key: Any, deployment: Any) -> None:
        self.keys[self.current].add(key)
        self.keyed_deployments.append(deployment)

    # -- pytest hooks ----------------------------------------------------

    def pytest_runtest_protocol(self, item: Any) -> None:
        self.current = str(Path(item.fspath).resolve().relative_to(ROOT))
        self.rows[self.current]["tests"] += 1
        self._started = time.perf_counter()

    def pytest_runtest_logfinish(self) -> None:
        self.rows[self.current]["wall_s"] += time.perf_counter() - self._started

    def pytest_sessionstart(self) -> None:
        _instrument(self)

    def table(self) -> List[Dict[str, Any]]:
        for name, row in self.rows.items():
            row["distinct"] = len(self.keys[name]) if self.keyed else None
        return sorted(
            ({"file": name, **{k: round(v, 2) if isinstance(v, float) else v
                               for k, v in row.items()}}
             for name, row in self.rows.items()),
            key=lambda row: -row["wall_s"])


def _instrument(tally: Tally) -> None:
    """Wrap the three operations in place.

    `resolve` and `run` are methods, so one assignment is seen by every
    caller. `iter_tree` is a module-level function that most callers
    imported by name, so the wrapper is also written over every binding of
    it already in `sys.modules` -- otherwise the count would be of the two
    modules that call it through its module."""
    from backend.arch.loader import SpecLoader
    from backend.arch.eval.evaluator import Evaluator
    from backend.arch.eval.transforms import Deployment
    from backend.arch.spec_tree import build
    from tools.corpus_pins import corpus

    def wrap(original: Any, kind: str, charge: Any = None,
             note: Any = None, outermost: bool = False) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if kind == "asks" and tally.inside_group and not outermost:
                return original(*args, **kwargs)
            # `iter_tree` is memoised on the resolution it is given, so a
            # warm call costs nothing and is not an expansion. Charge the
            # builds, which is what a shared resolution removes.
            charged = charge is None or charge(*args)
            if note is not None:
                tally.note_key(note(*args), args[-1])
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if charged:
                    tally.charge(kind, time.perf_counter() - started)
        wrapper._suite_timing = True  # type: ignore[attr-defined]
        return wrapper

    for owner, name, kind in ((SpecLoader, "resolve", "resolutions"),
                              (Evaluator, "run", "walks")):
        original = getattr(owner, name)
        if not getattr(original, "_suite_timing", False):
            setattr(owner, name, wrap(original, kind))

    # Every walk ASKED for, whether it ran or came from the memo, so the
    # `asks` and `distinct` columns mean the same thing before and after a
    # conversion: an unconverted file asks `Deployment.analyze`, a converted
    # one asks its group.
    #
    # ASKED THROUGH THE METHOD, NOT THE MODULE FUNCTION. `WalkGroup.of` is
    # wrapped rather than `corpus.shared_walk`, because `shared_walk` is
    # imported BY NAME (`tests/snapshots.py` does
    # `from ...corpus import shared_walk`), so rebinding the module
    # attribute is invisible to every caller -- which is exactly what the
    # first version of this did, and why its `asks` column read zero for
    # the converted files it was supposed to measure. A method is looked up
    # on the class at call time, so one assignment covers everybody.
    #
    # `WalkGroup` / `walk_key` are absent in a tree from before the shared
    # layer existed, which is exactly the tree a `before` measurement runs
    # in. There every ask goes through `Deployment.analyze`, so `asks` and
    # `walks` are right -- but there is no key to tell two deployments
    # apart, so `distinct` is UNAVAILABLE and reported as `-`. It is not
    # computed here from a key of this tool's own: that would be a second
    # implementation of `walk_key` to drift from the one being measured.
    key = getattr(corpus, "walk_key", None)
    tally.keyed = key is not None
    if not getattr(Deployment.analyze, "_suite_timing", False):
        Deployment.analyze = wrap(Deployment.analyze, "asks", note=key)
    group = getattr(corpus, "WalkGroup", None)
    if group is not None and not getattr(group.of, "_suite_timing", False):
        asked = wrap(group.of, "asks", note=lambda _self, d: key(d),
                     outermost=True)   # `args[-1]` is the deployment

        def of(self: Any, deployment: Any) -> Any:
            # The group's own `analyze` on a miss is not a second ask.
            was, tally.inside_group = tally.inside_group, True
            try:
                return asked(self, deployment)
            finally:
                tally.inside_group = was

        of._suite_timing = True  # type: ignore[attr-defined]
        group.of = of

    if not getattr(build.iter_tree, "_suite_timing", False):
        original = build.iter_tree
        wrapper = wrap(original, "trees", charge=(
            lambda resolved: resolved._spec_tree_ir_cache is None))
        for module in list(sys.modules.values()):
            if getattr(module, "iter_tree", None) is original:
                module.iter_tree = wrapper


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="test files to measure")
    parser.add_argument("--out", help="write the table here as JSON")
    args, extra = parser.parse_known_args(argv)

    import pytest

    sys.path.insert(0, str(ROOT))
    tally = Tally()
    started = time.perf_counter()
    status = pytest.main(["-q", "-p", "no:randomly", "--no-header",
                          *args.paths, *extra], plugins=[tally])
    elapsed = time.perf_counter() - started

    table = tally.table()
    report = {"load_average": os.getloadavg(), "total_wall_s": round(elapsed, 1),
              "exit_status": int(status), "files": table}
    columns = ("resolutions", "trees", "tree_s", "asks", "distinct",
               "walks", "walk_s", "wall_s")
    print("\n" + "".join(f"{name:>12}" for name in columns) + "  file")
    print("\n".join("".join(
        f"{'-' if row[name] is None else row[name]:>12}" for name in columns)
        + f"  {row['file']}" for row in table))
    print(f"\ntotal wall {elapsed:.1f}s   load {os.getloadavg()[0]:.1f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
