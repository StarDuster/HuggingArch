"""Every axis an op contracts away has one extent across its inputs — read
from the op's declared signature, for every op whose template is a contract."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.validate._shapes import _check_contract_pair


def _node(op, out):
    return SimpleNamespace(op=op, name="n", ir_node=None, axes=None, axis=None,
                           shape_out=out)


def test_a_dsa_key_of_another_head_width_is_refused():
    score = _node("dsa_attention_score", [1, 128, 8, 4096])
    assert _check_contract_pair(score, [[1, 128, 8, 192], [1, 1, 4096, 192]]) == []
    (finding,) = _check_contract_pair(score, [[1, 128, 8, 192], [1, 1, 4096, 199]])
    assert finding.kind == "contracted_axis_mismatch"
    assert "d_head" in finding.detail


def test_probabilities_and_values_agree_on_the_key_axis():
    apply = _node("attention_apply", [1, 128, 8, 128])
    (finding,) = _check_contract_pair(apply, [[1, 128, 8, 4096], [1, 128, 4000, 128]])
    assert "s_kv" in finding.detail


def test_a_window_attends_a_different_length_than_it_reads():
    """`swa_attention_apply` gives the window and the keys different labels:
    a window of 128 over 4096 cached keys is not a mismatch."""
    apply = _node("swa_attention_apply", [1, 64, 8, 128])
    assert _check_contract_pair(apply, [[1, 64, 8, 128], [1, 8, 4096, 128]]) == []
