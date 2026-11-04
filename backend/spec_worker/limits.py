"""Agent-run budgets. One module so generate / fix / review / model_info
cannot drift from each other.

Environment knobs override the defaults. Fix uses the same default as
generate (`SPEC_AGENT_RUN_TIMEOUT_SECONDS`); set `HA_SPEC_FIX_TIMEOUT_SECONDS`
to pick a different cap.

A review's budget scales with the spec it reads (`review_budget_sec`),
between `REVIEW_AGENT_SEC` and `REVIEW_AGENT_MAX_SEC`.

Generate and fix run in rounds: the agent's first turn, then up to
`max_repair_rounds()` repair turns when the verdict gate rejects its output.
Each round has its own timer, `ROUND_BUDGET_SEC`, inside the run's overall
deadline (`GENERATE_AGENT_SEC` / `FIX_AGENT_SEC`, 5 hours; a round gets 2 hours): a round stops at
`min(round start + round budget, overall deadline)` (`stream_agent_core`).
Op-continue under the same generate lock shares that overall deadline.

A generate run whose agent's service is unavailable (auth, quota, no model
configured, 5xx, no connection) moves on to the next agent backend that has
credentials, inside the same overall deadline; `HA_SPEC_AGENT_FAILOVER`
orders the backends it tries after the run's own (`failover_order`). A round
stopped by its timer with a draft on disk continues from that draft while the
overall deadline lasts.

Lock TTL covers every operation that holds the spec lock — a generate or fix
of `1 + repair rounds` full rounds, its overall deadline, review at its
largest budget, op-continue — plus validate/publish grace, so the lock does
not expire under a live round. `SPEC_AGENT_LOCK_TTL_SECONDS` can raise the
TTL, not lower it below that required budget.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


GENERATE_TIMEOUT_KNOB = "SPEC_AGENT_RUN_TIMEOUT_SECONDS"
FIX_TIMEOUT_KNOB = "HA_SPEC_FIX_TIMEOUT_SECONDS"
REVIEW_TIMEOUT_KNOB = "HA_SPEC_REVIEW_TIMEOUT_SECONDS"
REVIEW_MAX_KNOB = "HA_SPEC_REVIEW_MAX_SECONDS"
MODEL_INFO_TIMEOUT_KNOB = "HA_MODEL_INFO_TIMEOUT_SECONDS"
LOCK_TTL_KNOB = "SPEC_AGENT_LOCK_TTL_SECONDS"
VALIDATE_TIMEOUT_KNOB = "HA_SPEC_VALIDATE_TIMEOUT_SECONDS"
OP_CONTINUE_TIMEOUT_KNOB = "HA_SPEC_OP_CONTINUE_SECONDS"
ROUND_TIMEOUT_KNOB = "HA_SPEC_ROUND_TIMEOUT_SECONDS"
REPAIR_ROUNDS_KNOB = "HA_SPEC_REPAIR_ROUNDS"
FAILOVER_KNOB = "HA_SPEC_AGENT_FAILOVER"


def failover_order(own_cli: str, known: Sequence[str]) -> tuple[str, ...]:
    """The agent CLIs a run tries, in order: its own first, then the ones it
    moves on to when an agent's service is unavailable. `known` is every
    registered CLI.

    Default (knob unset): the others in `known` order.
    `HA_SPEC_AGENT_FAILOVER=kimi,claude`: those, in that order (the run's own
    CLI is skipped where it appears). Set but empty: no failover. Only CLIs
    with credentials for the run are tried — that filter is the caller's."""
    raw = os.environ.get(FAILOVER_KNOB)
    if raw is None:
        others = [cli for cli in known if cli != own_cli]
    else:
        others = [cli.strip().lower() for cli in raw.split(",") if cli.strip()]
        unknown = sorted(set(others) - set(known))
        if unknown:
            # A typo'd knob must not silently drop a backend from the order.
            raise ValueError(
                f"{FAILOVER_KNOB} names unknown agent CLI(s) {unknown}; "
                f"expected some of {', '.join(known)}")
    order = [own_cli]
    for cli in others:
        if cli not in order:
            order.append(cli)
    return tuple(order)


def max_repair_rounds() -> int:
    """Verdict-gate repair attempts per run. Default 2; HA_SPEC_REPAIR_ROUNDS=0
    disables the loop (each repair round is a fresh agent invocation, i.e.
    real token spend on BYOK accounts)."""
    raw = os.environ.get(REPAIR_ROUNDS_KNOB)
    if raw is None:
        return 2
    try:
        return max(0, int(raw))
    except ValueError as e:
        # A typo'd knob must not silently run the default number of paid
        # repair rounds — surface it so the operator knows it didn't take.
        raise ValueError(
            f"{REPAIR_ROUNDS_KNOB} must be an integer, got {raw!r}") from e


#: The whole run's wall-clock deadline, every round included.
GENERATE_AGENT_SEC = _int_env(GENERATE_TIMEOUT_KNOB, 5 * 60 * 60)
#: One round's own timer (the first turn, or one repair turn).
ROUND_BUDGET_SEC = _int_env(ROUND_TIMEOUT_KNOB, 2 * 60 * 60)
FIX_AGENT_SEC = _int_env(FIX_TIMEOUT_KNOB, GENERATE_AGENT_SEC)
REVIEW_AGENT_SEC = _int_env(REVIEW_TIMEOUT_KNOB, 20 * 60)
REVIEW_AGENT_MAX_SEC = max(_int_env(REVIEW_MAX_KNOB, 3 * REVIEW_AGENT_SEC),
                           REVIEW_AGENT_SEC)
MODEL_INFO_SEC = _int_env(MODEL_INFO_TIMEOUT_KNOB, 120)
LOCK_GRACE_SEC = 15 * 60
VALIDATE_SEC = _int_env(VALIDATE_TIMEOUT_KNOB, 10 * 60)
OP_CONTINUE_SEC = _int_env(OP_CONTINUE_TIMEOUT_KNOB, GENERATE_AGENT_SEC)


#: Review budget terms: a fixed part for reading the prompt, snapshot and
#: validate output and for writing the report, plus reading time per KiB of
#: staged forward source and per component file the record brings (each new
#: component is checked line by line against the source).
REVIEW_BASE_SEC = 5 * 60
REVIEW_SEC_PER_SOURCE_KIB = 6
REVIEW_SEC_PER_COMPONENT = 90


def review_budget_sec(source_bytes: int, n_components: int, *,
                      floor: int = REVIEW_AGENT_SEC,
                      ceiling: int = REVIEW_AGENT_MAX_SEC) -> int:
    """Wall-clock budget for one review of a spec:

        base + 6 s per KiB of forward source + 90 s per component file,

    never below `floor` (`HA_SPEC_REVIEW_TIMEOUT_SECONDS`, 20 min) and never
    above `ceiling` (`HA_SPEC_REVIEW_MAX_SECONDS`, 3x the floor). A dense
    transformers model (~20-30 KiB of source, no new components) gets the
    floor; DeepSeek-V4.1-Flash (103 KiB across five files, 13 components)
    gets 300 + 618 + 1170 = 2088 s."""
    estimate = (REVIEW_BASE_SEC
                + REVIEW_SEC_PER_SOURCE_KIB * (max(source_bytes, 0) // 1024)
                + REVIEW_SEC_PER_COMPONENT * max(n_components, 0))
    return int(min(max(estimate, floor), max(ceiling, floor)))


def rounds_hold_sec(round_budget: int, repair_rounds: int) -> int:
    """The longest a run of rounds can run: `1 + repair_rounds` rounds, each
    at its full budget."""
    return round_budget * (1 + max(0, repair_rounds))


def lock_hold_budget_sec(generate: int, fix: int, review: int,
                         op_continue: int, *, round_budget: int = 0,
                         repair_rounds: int = 0) -> int:
    """Seconds the lock must cover the agent itself — the longest holder: a
    generate / fix at its overall deadline, or its rounds at their full
    budgets, review, op-continue."""
    return max(generate, fix, review, op_continue,
               rounds_hold_sec(round_budget, repair_rounds))


def default_lock_ttl_sec(
        generate: int = GENERATE_AGENT_SEC,
        fix: int = FIX_AGENT_SEC,
        review: int = REVIEW_AGENT_MAX_SEC,
        op_continue: int = OP_CONTINUE_SEC,
        validate: int = VALIDATE_SEC,
        grace: int = LOCK_GRACE_SEC,
        round_budget: int = ROUND_BUDGET_SEC,
        repair_rounds: Optional[int] = None) -> int:
    """Lock TTL = longest holder + verdict subprocess + metadata/publish grace."""
    if repair_rounds is None:
        repair_rounds = max_repair_rounds()
    return lock_hold_budget_sec(
        generate, fix, review, op_continue, round_budget=round_budget,
        repair_rounds=repair_rounds) + validate + grace


def resolve_lock_ttl_sec(
        requested: Optional[int] = None,
        generate: int = GENERATE_AGENT_SEC,
        fix: int = FIX_AGENT_SEC,
        review: int = REVIEW_AGENT_MAX_SEC,
        op_continue: int = OP_CONTINUE_SEC,
        validate: int = VALIDATE_SEC,
        grace: int = LOCK_GRACE_SEC,
        round_budget: int = ROUND_BUDGET_SEC,
        repair_rounds: Optional[int] = None) -> int:
    """Floor an explicit TTL at the budget the lock must actually cover."""
    required = default_lock_ttl_sec(
        generate=generate, fix=fix, review=review, op_continue=op_continue,
        validate=validate, grace=grace, round_budget=round_budget,
        repair_rounds=repair_rounds)
    if requested is None:
        raw = os.environ.get(LOCK_TTL_KNOB)
        if raw is None or raw == "":
            return required
        requested = int(raw)
    return max(requested, required)


LOCK_TTL_SEC = resolve_lock_ttl_sec()
