"""Every scoreboard counter stays at or under its baseline.

`tools/scoreboard.py` defines the counters; `scoreboard_baseline.json` next
to this file declares the ceiling each is held to. The counters match code
by pattern, a proxy for the rules they track, so this is a report rather
than a gate: the module runs with the `regression` marker (the manual
`regression` workflow), not on every push. A new site is a question for
review (`python -m tools.scoreboard --list <counter>`), not a red build.
A commit that removes sites may lower the baseline (`--rebaseline`); one
that does not is not red for it.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from tools import scoreboard as sb

pytestmark = pytest.mark.regression  # a report on pattern counts, not a gate

BASELINE = sb.load_baseline()


@pytest.fixture(scope="module")
def measured():
    """Walked on first use, not at import: every xdist worker imports this
    module to collect it, and only the worker that runs it needs the walk."""
    return sb.measure()


@pytest.mark.parametrize("counter", sb.COUNTERS, ids=[c.key for c in sb.COUNTERS])
def test_counter_holds_its_baseline(counter, measured):
    now = len(measured[counter.key])
    floor = BASELINE[counter.key]["baseline"]
    assert now <= floor, (
        f"{counter.title}: {now} sites, baseline {floor}. The baseline keeps "
        "only a count, so find the new ones with `python -m tools.scoreboard "
        f"--list {counter.key}`; the last five by path:\n  "
        + "\n  ".join(measured[counter.key][-5:])
        + f"\n({counter.definition})")


@pytest.mark.parametrize("fact", sb.FACTS, ids=[f.key for f in sb.FACTS])
def test_fact_rederivations_do_not_grow(fact, measured):
    key = f"fact:{fact.key}"
    now = len(measured[key])
    floor = BASELINE[key]["baseline"]
    assert now <= floor, (
        f"{fact.title}: {now} restatements, baseline {floor}. "
        f"Declared at {fact.declared or 'nowhere yet'}; ask it instead. The last "
        "five by path (`python -m tools.scoreboard --list {key}` shows all):\n  "
        + "\n  ".join(measured[key][-5:]))

