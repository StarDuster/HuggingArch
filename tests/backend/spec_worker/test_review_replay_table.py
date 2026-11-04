"""The review report shows the analysis replay: the deployments validate ran
the analysis on, and the node each refusal names."""
from __future__ import annotations

from backend.spec_worker._prompts.review import _render_validate_summary


def test_the_replay_table_lists_every_deployment_and_refusal():
    text = _render_validate_summary({
        "valid": False,
        "errors": ["analysis engine refusal at node `lm_head`"],
        "analysis": "refused",
        "blocked_by_engine": True,
        "analysis_replay": {"deployments": [
            {"label": "single", "parallel": "one GPU", "refusals": []},
            {"label": "tp8", "parallel": "tp_attention=8",
             "refusals": [{"kind": "analysis_engine_refusal",
                           "node": "lm_head", "stage": "capacity"}]},
        ]},
        "analysis_refusals": [{"kind": "analysis_engine_refusal",
                               "node": "lm_head", "detail": "x"}],
    })
    assert "analysis replay: refused" in text
    assert "blocked_by_engine" in text
    assert "| single | one GPU | ok |" in text
    assert "| tp8 | tp_attention=8 | analysis_engine_refusal at `lm_head` (capacity) |" in text


def test_a_spec_refusal_carries_its_fix():
    text = _render_validate_summary({
        "valid": False, "errors": ["e"], "analysis": "not_run",
        "analysis_refusals": [{"kind": "analysis_refusal", "node": "mask",
                               "fix": "use slice"}],
    })
    assert "analysis replay: not_run" in text
    assert "analysis_refusal node `mask` — fix: use slice" in text


def test_a_result_without_analysis_renders_as_before():
    assert "analysis replay" not in _render_validate_summary({"valid": True})
