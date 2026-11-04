"""A block's short name (`block_label`): read off what is declared, never a
component id where a declared name exists."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.snapshots import shared_program, snapshot_config

#: An identifier as a spec writes one (`dsv4_moe_hash`, `mtp_block_with_head`).
_ID = re.compile(r"[a-z0-9]_[a-z0-9]")

# First: DeepSeek-V4-Pro, whose hash- and score-routed MoE once read as
# "CSA + dsv4_moe_hash"; then models each rule was written for.
_MODELS = ["deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V3",
           "moonshotai/Kimi-K3", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
           "THUDM/glm-4-9b", "zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4.1-Flash",
           "meituan-longcat/LongCat-Flash-Thinking-2601"]


@pytest.mark.parametrize("model_id", _MODELS)
def test_a_block_that_calls_a_named_component_is_named_by_it(model_id):
    """Every block that calls a named component reads as those names: no
    component or template id in its short name, and no two blocks of one
    stack share one. A block that calls nothing with a name (a vision tower
    of primitive ops) is outside the rule, by what it calls rather than by
    a list of models. Which blocks call a named component is read here off
    the spec directly, not through the labeller under test."""
    from backend.arch import roles
    snapshot_config(model_id)
    program = shared_program(model_id)
    spec = program.resolved.spec

    def resolve(name):
        return (spec.blocks or {}).get(name) or (spec.components or {}).get(name)

    def calls_a_named_component(name, seen=frozenset()):
        graph = resolve(name)
        if graph is None or name in seen:
            return False
        called = [node.op for node in (graph.graph or {}).values()]
        called += [str(v) for v in (graph.synth_params or {}).values()]
        for op in called:
            sub = resolve(op)
            if sub is None:
                continue
            role = roles.get(sub.role) if sub.role else None
            if sub.attn_display or (role is not None and role.label):
                return True
            if calls_a_named_component(op, seen | {name}):
                return True
        return False

    by_stack = {}
    for block in program.blocks:
        meta = block.meta
        by_stack.setdefault(meta.stack_name, {})[meta.template_name] = meta.short_name
        if calls_a_named_component(meta.template_name):
            assert not _ID.search(meta.short_name), (meta.template_name, meta.short_name)
    for stack, names in by_stack.items():
        assert len(set(names.values())) == len(names), (stack, names)
