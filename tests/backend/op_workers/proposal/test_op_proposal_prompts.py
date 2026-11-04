"""Contracts for the live op-proposal prompt."""

from backend.arch.ops import ShapeKind
from backend.op_workers.proposal._prompts.system import build_system_prompt


def test_shape_kind_prompt_tracks_the_live_enum():
    prompt = build_system_prompt()

    assert "one of the nine values" in prompt
    for kind in ShapeKind:
        assert f'{kind.name} = "{kind.value}"' in prompt
        assert f"- `{kind.value}`" in prompt
