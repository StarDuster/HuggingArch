"""Generate / fix / review / model_info budgets share one module."""
from __future__ import annotations

from backend.spec_worker.limits import (
    FIX_AGENT_SEC,
    GENERATE_AGENT_SEC,
    LOCK_GRACE_SEC,
    MODEL_INFO_SEC,
    OP_CONTINUE_SEC,
    OP_CONTINUE_TIMEOUT_KNOB,
    REVIEW_AGENT_MAX_SEC,
    REVIEW_AGENT_SEC,
    ROUND_BUDGET_SEC,
    VALIDATE_SEC,
    default_lock_ttl_sec,
    lock_hold_budget_sec,
    max_repair_rounds,
    resolve_lock_ttl_sec,
    review_budget_sec,
    rounds_hold_sec,
)
from backend.spec_worker.runtime import _AGENT_HARD_CAP_SEC, _LOCK_TTL_SECONDS


def test_fix_and_op_continue_match_generate():
    assert GENERATE_AGENT_SEC == 5 * 60 * 60
    assert ROUND_BUDGET_SEC == 2 * 60 * 60
    assert FIX_AGENT_SEC == GENERATE_AGENT_SEC
    assert OP_CONTINUE_SEC == GENERATE_AGENT_SEC
    assert REVIEW_AGENT_SEC == 20 * 60
    assert MODEL_INFO_SEC == 120
    assert _AGENT_HARD_CAP_SEC == GENERATE_AGENT_SEC
    assert default_lock_ttl_sec() == _LOCK_TTL_SECONDS
    assert OP_CONTINUE_TIMEOUT_KNOB == "HA_SPEC_OP_CONTINUE_SECONDS"


def test_lock_ttl_covers_the_longest_holder_not_generate_alone():
    assert lock_hold_budget_sec(3600, 7200, 1200, 3600) == 7200
    assert default_lock_ttl_sec(
        generate=3600, fix=7200, review=1200, op_continue=3600,
        validate=600, grace=900, round_budget=600, repair_rounds=2,
    ) == 7200 + 600 + 900
    assert default_lock_ttl_sec() == (
        max(GENERATE_AGENT_SEC, FIX_AGENT_SEC, REVIEW_AGENT_MAX_SEC, OP_CONTINUE_SEC,
            ROUND_BUDGET_SEC * (1 + max_repair_rounds()))
        + VALIDATE_SEC + LOCK_GRACE_SEC
    )


def test_lock_ttl_outlives_every_round_at_its_full_budget():
    """Rounds each get their own timer, so the lock covers 1 + repair rounds
    full rounds: with the defaults (60 min rounds, 2 repairs) that is 3 h,
    longer than the 2 h overall deadline."""
    assert rounds_hold_sec(3600, 2) == 3 * 3600
    assert rounds_hold_sec(3600, 0) == 3600
    assert default_lock_ttl_sec(
        generate=7200, fix=7200, review=1200, op_continue=7200,
        validate=600, grace=900, round_budget=3600, repair_rounds=2,
    ) == 3 * 3600 + 600 + 900
    assert default_lock_ttl_sec(repair_rounds=0) == (
        GENERATE_AGENT_SEC + VALIDATE_SEC + LOCK_GRACE_SEC)


def test_repair_rounds_knob_sizes_the_lock(monkeypatch):
    monkeypatch.setenv("HA_SPEC_REPAIR_ROUNDS", "4")
    assert max_repair_rounds() == 4
    assert default_lock_ttl_sec() == (
        5 * ROUND_BUDGET_SEC + VALIDATE_SEC + LOCK_GRACE_SEC)


def test_lock_ttl_env_cannot_go_below_the_required_budget():
    required = default_lock_ttl_sec()
    assert resolve_lock_ttl_sec(requested=60) == required
    assert resolve_lock_ttl_sec(requested=required + 120) == required + 120


def test_review_budget_is_the_floor_for_a_small_spec():
    # modeling_qwen2.py is ~21 KiB and a library spec brings no components.
    assert review_budget_sec(21_608, 0) == REVIEW_AGENT_SEC == 20 * 60


def test_review_budget_grows_with_source_and_components():
    # DeepSeek-V4.1-Flash: five staged files, 105,633 bytes; 13 draft
    # components. 300 + 6 * 103 + 90 * 13.
    assert review_budget_sec(105_633, 13) == 2088
    assert review_budget_sec(105_633, 14) == 2088 + 90
    assert review_budget_sec(105_633 + 1024, 13) == 2088 + 6


def test_review_budget_is_capped_and_the_lock_covers_the_cap():
    assert REVIEW_AGENT_MAX_SEC == 3 * REVIEW_AGENT_SEC
    assert review_budget_sec(10_000_000, 200) == REVIEW_AGENT_MAX_SEC
    assert review_budget_sec(10_000_000, 200, floor=600, ceiling=900) == 900
    assert default_lock_ttl_sec() >= REVIEW_AGENT_MAX_SEC + VALIDATE_SEC
