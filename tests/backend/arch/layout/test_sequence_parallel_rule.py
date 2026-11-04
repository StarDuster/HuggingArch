"""The deployment rules the layout walk refuses by, whatever the model, are
stated once (`ParallelConfig.refusals(phase)`): the walk
(`LayoutPropagation.of`) raises them as `DeploymentRefusal`s and the gate
every request passes (`validate`) reports them for every phase the
deployment serves.

Each rule used to be stated only inside `LayoutPropagation.of`, so
`validate()` passed the deployment and the walk raised: first
`sequence_parallel=True` at `tp_attention=1`, then sequence parallelism
beside prefill context parallelism (both cut the residual stream's
sequence axis), which still passed the gate after the first was moved.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.refusal import DeploymentRefusal
from backend.arch.layout.parallel_config import CpScheme, ParallelConfig
from backend.arch.layout.propagation import LayoutPropagation

#: (config, phase, the knobs the walk of that phase refuses, or None). The
#: first case is the one the gate let through after the tp rule moved.
CASES = (
    (dict(tp_attention=2, tp_dense_ffn=2, cp=2, sequence_parallel=True),
     "prefill", ("sequence_parallel", "cp")),
    (dict(tp_attention=1, tp_dense_ffn=1, dp_attention=2, ep_moe=2,
          sequence_parallel=True), "prefill",
     ("sequence_parallel", "tp_attention")),
    (dict(sequence_parallel=True), "decode",
     ("sequence_parallel", "tp_attention")),
    # Prefill context parallelism runs no ranks in decode: that walk has
    # no second sequence cut, though the deployment is still refused.
    (dict(tp_attention=2, tp_dense_ffn=2, cp=2, sequence_parallel=True),
     "decode", None),
    (dict(tp_attention=2, tp_dense_ffn=2, sequence_parallel=True),
     "prefill", None),
    (dict(tp_attention=2, tp_dense_ffn=2, cp=2), "prefill", None),
)


@pytest.mark.parametrize("knobs,phase,refused", CASES,
                         ids=[f"case{i}" for i in range(len(CASES))])
def test_validate_reports_the_refusal_the_walk_raises(knobs, phase, refused):
    pc = ParallelConfig(**knobs)
    refusals = pc.refusals(phase)
    assert (refusals[0].knobs if refusals else None) == refused
    errs = pc.validate()
    for refusal in refusals:
        assert str(refusal) in errs
    if refused:
        with pytest.raises(DeploymentRefusal) as raised:
            LayoutPropagation.of(pc, phase)
        assert raised.value.knobs == refused
        assert str(raised.value) == str(refusals[0])
    else:
        LayoutPropagation.of(pc, phase)
        if not any(pc.refusals(p) for p in ("prefill", "decode")):
            assert errs == []


def _grid():
    """Small deployments over every knob the rule constructor reads.
    Ulysses is left out: its rule reads the program's cache region, so it
    is not built without a model."""
    degrees = (1, 2, 4)
    for (pp, tpa, tpd, tpe, ep, cp, dcp, dpa, sp, scheme) in itertools.product(
            (1, 2), degrees, degrees, degrees, degrees, degrees, (1, 2),
            degrees, (False, True), (CpScheme.PCP, CpScheme.LAYERSPLIT)):
        yield ParallelConfig(
            pp=pp, tp_attention=tpa, tp_dense_ffn=tpd, tp_moe_expert=tpe,
            ep_moe=ep, cp=cp, dcp=dcp, dp_attention=dpa,
            sequence_parallel=sp, cp_scheme=scheme)


@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_every_deployment_the_gate_passes_builds_a_layout_rule(phase):
    """Second route over the knob space: the gate and the walk agree on
    every small deployment — the walk builds whatever `validate` passes."""
    disagree = []
    for pc in _grid():
        if pc.validate():
            continue
        try:
            LayoutPropagation.of(pc, phase)
        except ValueError as exc:
            disagree.append((pc, str(exc)))
    assert disagree == []
