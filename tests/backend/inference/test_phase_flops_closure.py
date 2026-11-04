"""A step's served FLOPs are the model's FLOPs plus its activation casts'.

A phase's FLOPs are the sum of its per-op rows: the served graph, with the
collectives, the activation casts (`materialize_cast`) and the fused
launches grafted onto the walk. They used to be folded off the logical walk
instead, which has no cast node, so on every quantised-activation
deployment the headline fell short of its own rows by exactly the casts'
FLOPs and the SOL closure refused the estimate (MiniMax-M2.7 and every
other FP8 model of the SOL surface golden).

Now that the headline IS the rows' sum, `headline == Σ rows` holds by
construction and can catch nothing. What can is a conservation law between
two routes that share no arithmetic:

    Σ served rows' FLOPs == Σ logical walk's depth-0 FLOPs (`flops_fold`)
                            + Σ grafted casts' FLOPs (`CastGraft`, each
                              cast once, standalone or absorbed by a fusion,
                              counted by hand from its operand's elements)

Collectives compute nothing at one rank, and fusion moves FLOPs between
nodes without creating or dropping any. So this goes red when a rewrite
adds FLOPs (a cast billed both standalone and as a fused member) or drops
them (a fusion that absorbs a member and forgets its FLOPs).

Llama-3.2-1B under the FP8 weight preset: a small model whose matmuls read
an FP8 activation, so the served graph carries casts — the down projection's
absorbed into the activation epilogue, the rest standalone.
"""
from __future__ import annotations

import pytest

from tests.snapshots import resolve_snapshot

from backend.arch.eval.dim import value_of
from backend.arch.precision_profile import PrecisionProfile
from backend.inference.framework_bindings import get_binding
from backend.inference.materialize_cast import CAST_OP
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import flops_fold
from backend.inference.pipeline import analyze
from backend.inference.serving import rewrite_serving_graph
from backend.inference.step_eval import evaluate_step

_MODEL = "meta-llama/Llama-3.2-1B"

#: What a cast computes per element, stated here rather than read off the
#: op's `flops_fn`: one absmax pass and one multiply (vLLM `QuantFP8`,
#: sglang `per_token_group_quant_fp8`). The served rows' cast FLOPs come
#: from the op's declaration via `materialize_cast.cast_facts`; this side
#: must not, or the law would share its subject's arithmetic.
_CAST_FLOPS_PER_ELEMENT = 2


def _step(phase: str):
    return evaluate_step(
        resolve_snapshot(_MODEL), ParallelConfig(), phase=phase, batch=2,
        input_tokens=512, output_tokens=64,
        precision=PrecisionProfile.from_request({"preset": "w_fp8_kv_fp8"}))


def _cast_flops(step) -> int:
    """Every grafted cast's FLOPs on this rank, counted by hand: the
    elements of the operand it copies, as the walk shapes that operand on
    its consumer's input (one rank, so the walk's shapes are the rank's),
    times the per-element FLOPs stated above, times the times the walk runs
    the consumer. A cast several readers share is counted once. Only which
    edges carry a cast is read off the graft."""
    analysis = step.analysis
    casts = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm")).casts
    total, seen = 0, set()
    for site in casts.sites:
        if site.node.uid in seen:
            continue
        seen.add(site.node.uid)
        elements = 1
        for axis in analysis.shapes.values[site.consumer].ins[site.index]:
            elements *= int(value_of(axis))
        total += (_CAST_FLOPS_PER_ELEMENT * elements
                  * int(analysis.instances.values[site.consumer]))
    return total


@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_served_flops_are_the_models_plus_the_casts(phase):
    step = _step(phase)
    served = analyze(step.spec, parallel=ParallelConfig(), step_eval=step,
                     framework_binding=get_binding("vllm"),
                     breakdown=False).per_op_timing
    # Both kinds of cast are present, so both the standalone and the
    # absorbed term are exercised: the down projection's is absorbed.
    standalone = [r for r in served if r.get("op") == CAST_OP]
    casts = _cast_flops(step)
    assert standalone and casts > sum(int(r["flops_per_rank"]) for r in standalone)

    model = sum(int(r["flops_per_rank"]) for r in flops_fold(step).rows
                if r["depth"] == 0)
    assert sum(int(r.get("flops_per_rank") or 0) for r in served) \
        == model + casts
