"""A group-limited router's group steps exist exactly when they mask something.

`router_softmax_topk`'s group score, group top-k and mask are
`when: "n_expert_groups - n_groups_active"` — the groups masked out per
token (DeepseekV2TopkRouter's `group_limited_greedy`). At 0 every group is
kept, the mask is the identity, and the three steps are not built. The
expression replaced a registry symbol, `n_groups_masked`, that said the same
subtraction under a name and was read nowhere else.

Over the corpus the 35 MoE snapshots whose routers read the group counts
bind them; the switch is on for 7 (DeepSeek-R1, V2, V3, V3.1, V3.2, V3.2-Exp
and Ling-2.6-1T) and off for 28 (GLM-5.x joined the corpus reading the
counts at 0); of the 9 that
run `router_softmax_topk`, only DeepSeek-V2 builds the steps.
"""
from __future__ import annotations

import pytest

from backend.arch.eval.dim import eval_dim_expr
from backend.arch.eval.program import Program
from tests.snapshots import SNAPSHOT_DIR, resolve_snapshot
from tests.backend.arch.test_ep_as_count import TOY_CONFIG, TOY_MODEL, toy_resolved

SWITCH = "n_expert_groups - n_groups_active"
GROUP_STEPS = {"group_max", "group_topk", "group_mask"}


def _built_group_steps(resolved) -> set:
    """The group steps a `router_softmax_topk` call builds."""
    return {node.name for node in Program.of(resolved).nodes()
            if node.name in GROUP_STEPS and node.parent is not None
            and node.parent.op == "router_softmax_topk"}


def _uses_the_router(resolved) -> bool:
    return any(node.op == "router_softmax_topk"
               for node in Program.of(resolved).nodes())


def test_a_router_with_a_group_limit_builds_its_group_steps():
    limited = TOY_MODEL.replace(
        '  n_expert_groups: "1"\n  n_groups_active: "1"\n',
        '  n_expert_groups: "4"\n  n_groups_active: "2"\n')
    assert limited != TOY_MODEL
    assert _built_group_steps(toy_resolved(limited, TOY_CONFIG)) == GROUP_STEPS


def test_a_router_without_one_builds_none():
    assert _built_group_steps(toy_resolved(TOY_MODEL, TOY_CONFIG)) == set()


@pytest.mark.parametrize("name, built", [
    ("deepseek-ai__DeepSeek-V2", GROUP_STEPS),
    ("Qwen__Qwen3-30B-A3B", set()),
])
def test_on_two_snapshots(name, built):
    assert _built_group_steps(resolve_snapshot(name)) == built


@pytest.mark.regression
def test_the_switch_over_the_corpus():
    on, off, builders, users = [], [], [], 0
    for path in sorted(SNAPSHOT_DIR.glob("*.json")):
        resolved = resolve_snapshot(path.stem)
        dims = resolved.dim_values
        if "n_expert_groups" not in dims:
            continue          # a router that reads no group count
        (on if eval_dim_expr(SWITCH, dims) > 0 else off).append(path.stem)
        if _uses_the_router(resolved):
            users += 1
            if _built_group_steps(resolved):
                builders.append(path.stem)
    assert sorted(on) == [
        "deepseek-ai__DeepSeek-R1", "deepseek-ai__DeepSeek-V2",
        "deepseek-ai__DeepSeek-V3", "deepseek-ai__DeepSeek-V3.1",
        "deepseek-ai__DeepSeek-V3.2", "deepseek-ai__DeepSeek-V3.2-Exp",
        "inclusionAI__Ling-2.6-1T"]
    assert len(off) == 28
    assert users == 9
    assert builders == ["deepseek-ai__DeepSeek-V2"]
