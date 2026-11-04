"""The admission state machine as a truth table: constructed facts in, key
states and run status out, no engine. `tools/corpus_pins/states.py` states
the rules; every case below also holds `blocked_by` to the failed global
gates whatever the key's state."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.corpus_pins.states import (
    BlockedKind, KeyFacts, KeyState, RunFacts, Status, key_state, overall)

S = KeyState


@pytest.mark.parametrize("facts,failed,state", [
    # branch 1: a conditional pin whose predicate is false
    (KeyFacts(applies=False, present=True, waits_on=("G4",)), ["G1"], S.NOT_APPLICABLE),
    # branch 2: present wins over every gate and dependency
    (KeyFacts(present=True), ["G1"], S.PRESENT),
    (KeyFacts(present=True, waits_on=("G4",)), ["G2"], S.PRESENT),
    # branch 3: withheld, with G1 / G2 failing beside it
    (KeyFacts(waits_on=("G4",)), ["G1"], S.WITHHELD),
    (KeyFacts(waits_on=("G4",), own_g1_failed=True), ["G1"], S.WITHHELD),
    (KeyFacts(waits_on=("G4", "D4")), ["G2"], S.WITHHELD),
    # branch 4: its own pin drifted, and G2 failed too
    (KeyFacts(own_g1_failed=True), ["G1", "G2"], S.BASE_DRIFT),
    # branch 5: eligible but behind a global gate (another pin's G1, G0, G2)
    (KeyFacts(), ["G1"], S.WITHHELD_BY_GLOBAL),
    (KeyFacts(), ["G0"], S.WITHHELD_BY_GLOBAL),
    (KeyFacts(aggregate=True, delta_zero=True), ["G2"], S.WITHHELD_BY_GLOBAL),
    # branch 6: an aggregate cell this admission does not move
    (KeyFacts(aggregate=True, delta_zero=True), [], S.NO_OP),
    # an aggregate cell is never "present": it is moved or it is a no-op
    (KeyFacts(aggregate=True, present=True, delta_zero=False), [], S.READY),
    # branch 7
    (KeyFacts(), [], S.READY),
])
def test_key_state(facts, failed, state):
    verdict = key_state(facts, failed)
    assert verdict.state is state
    assert verdict.blocked_by == tuple(g for g in ("G0", "G1", "G2") if g in failed)
    assert verdict.eligible == (not facts.waits_on)


def test_a_write_turns_ready_into_written():
    assert key_state(KeyFacts(), [], written=True).state is S.WRITTEN
    assert key_state(KeyFacts(waits_on=("G4",)), [], written=True).state is S.WITHHELD


def _run(**kw):
    base = dict(g0_ok=True, g1_failures=(), g2_ok=True, keys=(), decisions_open=False)
    base.update(kw)
    return overall(RunFacts(**base))


@pytest.mark.parametrize("kw,expected", [
    (dict(g0_ok=False, g1_failures=("P9",), g2_ok=False), (Status.UNAVAILABLE, None)),
    (dict(g1_failures=("P9",), g2_ok=False), (Status.BLOCKED, BlockedKind.BASE)),
    (dict(g2_ok=False, keys=[(False, S.PRESENT)]), (Status.BLOCKED, BlockedKind.CANDIDATE_OR_RULE)),
    # an admitted model meeting a drifted base is blocked, not "nothing to do"
    (dict(g1_failures=("P10",), keys=[(False, S.PRESENT), (True, S.NO_OP)]),
     (Status.BLOCKED, BlockedKind.BASE)),
    (dict(keys=[(False, S.PRESENT), (False, S.NOT_APPLICABLE), (True, S.NO_OP)]),
     (Status.ALREADY_ADMITTED, None)),
    (dict(keys=[(False, S.WITHHELD), (False, S.READY)]), (Status.NEEDS_REVIEW, None)),
    (dict(keys=[(False, S.READY)], decisions_open=True), (Status.NEEDS_REVIEW, None)),
    # every aggregate cell a no-op, every keyed key ready
    (dict(keys=[(False, S.READY), (True, S.NO_OP), (True, S.NO_OP)]), (Status.READY, None)),
    # the Qwen3-VL shape: two aggregate cells move, ten do not
    (dict(keys=[(False, S.READY)] * 9 + [(True, S.READY)] * 2 + [(True, S.NO_OP)] * 10),
     (Status.READY, None)),
    # after the write the same run is still ready
    (dict(keys=[(False, S.WRITTEN), (True, S.WRITTEN), (True, S.NO_OP)]), (Status.READY, None)),
    # a partly admitted model: present keys count as done
    (dict(keys=[(False, S.PRESENT), (False, S.READY), (True, S.NO_OP)]), (Status.READY, None)),
])
def test_overall(kw, expected):
    assert _run(**kw) == expected


def test_blocked_by_lists_every_failed_gate_whatever_the_state():
    for facts in (KeyFacts(present=True), KeyFacts(waits_on=("G4",)),
                  KeyFacts(own_g1_failed=True), KeyFacts(), KeyFacts(applies=False)):
        assert key_state(facts, ["G0", "G1", "G2"]).blocked_by == ("G0", "G1", "G2")
