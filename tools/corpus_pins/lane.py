"""Which tests must run before anyone claims a golden did not move.

The four measuring goldens — the SOL surface, the capacity baseline, the
Scaling payload, the spec-match fingerprints — are each `regression`-marked,
`pyproject.toml` deselects that mark, and the manual `regression` workflow was
the only thing that ran them. On 2026-10-29/30 six commits changed the engine
and left all four behind: 56 SOL cells over 29 models, 26 capacity fields, 145
Scaling points, and a spec-match field still carrying a 435-commit-old hash.
Every one of those commits truthfully said no number moved, because no check
that ran could have said otherwise. A golden nothing runs is indistinguishable
from a golden that passes.

So this names the lane, and it names it by DERIVING it rather than listing it.
`REGISTRY` already declares, once, who owns each file under `tests/golden/`:
a pin and a decision name the node ids that compare them (`owner_tests`), and
a curated set names the module that keeps them (`module`). A second list of
"the tests that guard a pinned fact" would be a second table in the §1.4
sense — it would drift the first time someone added a golden, and nothing
would go red. `tests/meta/test_corpus_pin_registry.py` already asserts every
file under `tests/golden/` has exactly one owner, so coverage here holds by
construction: a new golden cannot exist without an owner, and an owner cannot
exist without appearing below.

    python -m tools.corpus_pins.lane          # node ids / modules, one a line

`-m ""` is required when running them: the default `addopts` would deselect
the very tests this prints.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

from tools.corpus_pins import REGISTRY

def _targets() -> Iterator[Tuple[str, ...]]:
    """Every registry entry's comparing tests, in registry order.

    A pin's node ids, or a curated set's module when it declares none — a
    curated set names the module that keeps its golden and no node ids, and
    running the module is the only honest reading of "this is who compares
    it". Both forms are what `pytest` takes as an argument.

    The condition is that the entry OWNS A FILE, not where that file lives.
    A measurement written into a file is the thing that can go stale while
    every test stays green, wherever it sits — the README's component counts
    drift the moment a component is added and are not under `tests/golden/`,
    so a directory test would have been a proxy for "is this a measured
    fact" and would have missed them. Of the 101 curated sets only 6 own a
    file; the other 95 are lists of deliberately-excluded models living in
    the test module that reads them, and a list in the code it is read by
    cannot fall out of step with itself.
    """
    for pin in REGISTRY.pins:
        if any(pin.artifacts()):
            yield pin.owner_tests
    for curated in REGISTRY.curated:
        if curated.artifact:
            yield (curated.module,)
    # Decisions are deliberately absent. D1 and D2 own files under
    # `tests/golden/pins/`, but a decision is a record a PERSON writes — the
    # TP split chosen for a model, the copy contract agreed for a tensor —
    # and the engine moving cannot make one stale, only disagree with it.
    # That disagreement is what the admitter's `judge` reports, and it is not
    # a golden comparison. A decision declares no comparing test for the same
    # reason, so there is nothing here to run.


def lane_targets() -> List[str]:
    """Sorted pytest targets: every registered pinned fact's comparer."""
    targets = set()
    for declared in _targets():
        targets.update(declared)
    return sorted(targets)


if __name__ == "__main__":
    for target in lane_targets():
        print(target)
