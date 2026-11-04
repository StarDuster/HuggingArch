"""The admission state machine: one state per pin key, one status per run.

Gates come in two kinds. Global gates hold for the whole run, and any one of
them failing means nothing is written:
  G0  the inputs are fixed (base SHA, bundle, snapshot, environment);
  G1  every pin the run might write reads back byte for byte from its file,
      and the gate surfaces and README counts agree with the base;
  G2  the corpus checks hold for the candidate and nothing it measured
      raised.
Item dependencies hold per key: G4 (the TP-split and copy-contract decisions
are written and agree with the measurement), D3 (a checkpoint residual's
reason), D4 (a new refusal family's note). A key whose dependencies are met
is eligible; an eligible key behind no failed global gate is written.

The key state is the first branch that holds:
  1. a conditional pin's predicate is false              -> n/a
  2. the key is already in the pin's file                 -> present
  3. it waits on a dependency                             -> withheld
  4. its own pin failed G1                                -> base_drift
  5. G0 or G2 failed, or another pin failed G1            -> withheld_by_global
  6. an aggregate cell whose delta is zero                -> no_op
  7. otherwise ready (written, once a write has run)
`blocked_by` lists every failed global gate whatever the state, and
`eligible` is "waits on nothing".

The run's status is the first row that holds, gates before anything else (a
re-run of an admitted model that meets a drifted base or a failed check is
blocked, not "nothing to do"):
  unavailable        G0 failed
  blocked (base)     a pin failed G1
  blocked (candidate_or_rule)  G2 failed
  already_admitted   every key the model owns in a keyed pin is present
  needs_review       a key is withheld, or a decision is pending or
                     inconsistent
  ready              every keyed key is ready (or present), every aggregate
                     cell ready or no_op
The status is decided before a write; a write then turns ready keys into
written ones and leaves the status ready.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple


class KeyState(str, enum.Enum):
    READY = "ready"
    WRITTEN = "written"
    WITHHELD = "withheld"
    WITHHELD_BY_GLOBAL = "withheld_by_global"
    BASE_DRIFT = "base_drift"
    PRESENT = "present"
    NOT_APPLICABLE = "n/a"
    NO_OP = "no_op"


class Status(str, enum.Enum):
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"
    ALREADY_ADMITTED = "already_admitted"
    NEEDS_REVIEW = "needs_review"
    READY = "ready"


class BlockedKind(str, enum.Enum):
    BASE = "base"
    CANDIDATE_OR_RULE = "candidate_or_rule"


GLOBAL_GATES = ("G0", "G1", "G2")


@dataclass(frozen=True)
class KeyFacts:
    """What decides one key's state."""

    #: false when a conditional pin's predicate does not hold
    applies: bool = True
    #: the key is already in the pin's file (keyed pins only)
    present: bool = False
    #: the unmet dependencies (G4, D3, D4)
    waits_on: Tuple[str, ...] = ()
    #: this key's own pin failed G1
    own_g1_failed: bool = False
    #: an aggregate cell (P8) rather than a model's key
    aggregate: bool = False
    #: an aggregate cell this admission does not move
    delta_zero: bool = False


@dataclass(frozen=True)
class KeyVerdict:
    state: KeyState
    eligible: bool
    waits_on: Tuple[str, ...]
    blocked_by: Tuple[str, ...]

    def as_dict(self) -> dict:
        return {"state": self.state.value, "eligible": self.eligible,
                "waits_on": list(self.waits_on), "blocked_by": list(self.blocked_by)}


def key_state(facts: KeyFacts, failed_gates: Iterable[str], written: bool = False) -> KeyVerdict:
    """The key's state (the seven branches above, first that holds)."""
    blocked_by = tuple(g for g in GLOBAL_GATES if g in set(failed_gates))
    eligible = not facts.waits_on
    if not facts.applies:
        state = KeyState.NOT_APPLICABLE
    elif facts.present and not facts.aggregate:
        state = KeyState.PRESENT
    elif facts.waits_on:
        state = KeyState.WITHHELD
    elif facts.own_g1_failed:
        state = KeyState.BASE_DRIFT
    elif blocked_by:
        state = KeyState.WITHHELD_BY_GLOBAL
    elif facts.aggregate and facts.delta_zero:
        state = KeyState.NO_OP
    else:
        state = KeyState.WRITTEN if written else KeyState.READY
    return KeyVerdict(state, eligible, tuple(facts.waits_on), blocked_by)


@dataclass(frozen=True)
class RunFacts:
    g0_ok: bool
    g1_failures: Tuple[str, ...]
    g2_ok: bool
    #: (aggregate?, state) for every applicable key the model owns
    keys: Sequence[Tuple[bool, KeyState]] = field(default_factory=tuple)
    #: a decision is pending or inconsistent
    decisions_open: bool = False


def overall(run: RunFacts) -> Tuple[Status, Optional[BlockedKind]]:
    """The run's status (the table above, first row that holds)."""
    if not run.g0_ok:
        return Status.UNAVAILABLE, None
    if run.g1_failures:
        return Status.BLOCKED, BlockedKind.BASE
    if not run.g2_ok:
        return Status.BLOCKED, BlockedKind.CANDIDATE_OR_RULE
    keyed = [state for aggregate, state in run.keys
             if not aggregate and state is not KeyState.NOT_APPLICABLE]
    if keyed and all(state is KeyState.PRESENT for state in keyed):
        return Status.ALREADY_ADMITTED, None
    if run.decisions_open or any(state is KeyState.WITHHELD for _, state in run.keys):
        return Status.NEEDS_REVIEW, None
    done_keyed = {KeyState.READY, KeyState.WRITTEN, KeyState.PRESENT, KeyState.NOT_APPLICABLE}
    done_aggregate = {KeyState.READY, KeyState.WRITTEN, KeyState.NO_OP}
    if all(state in (done_aggregate if aggregate else done_keyed)
           for aggregate, state in run.keys):
        return Status.READY, None
    return Status.NEEDS_REVIEW, None


def failed_gates(g0_ok: bool, g1_failures: Sequence[str], g2_ok: bool) -> List[str]:
    return [gate for gate, ok in (("G0", g0_ok), ("G1", not g1_failures), ("G2", g2_ok))
            if not ok]
