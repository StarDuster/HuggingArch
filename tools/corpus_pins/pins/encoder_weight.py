"""P3: per-rank encoder weight bytes, for every snapshot with a vision or
audio encoder, at TP=1 and TP=8.

Per (snapshot x TP x encoder stack): the stack's resident bytes on one rank
(`resident_weights(...).by_stack`: parameters, quant metadata and buffers),
and each weight node's resident bytes on one rank (`physical_weight_leaves`,
summed over the node's layer instances). Both come from the
persistent-artifact join of the encoder's own encode walk.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

from tools.corpus_pins.corpus import CorpusModel, shared_loader
from tools.corpus_pins.pin import JsonMapPin, Row, Scope

TP_DEGREES = (1, 8)


def tp_config(degree: int) -> Any:
    from backend.inference.parallel_config import ParallelConfig
    return ParallelConfig(
        tp_attention=degree, tp_dense_ffn=degree, tp_moe_expert=degree)


def weight_leaves(resolved: Any, degree: int) -> Any:
    """`(by_stack, leaves)` of the whole checkpoint at one TP degree:
    resident bytes per stack and the per-node weight leaves, from the
    persistent-artifact join of the walk that holds every stack
    (`Deployment.checkpoint`)."""
    from backend.inference.step_eval import evaluate_step
    analysis = evaluate_step(
        resolved, tp_config(degree), phase="prefill", batch=1,
        input_tokens=128, output_tokens=0,
    ).as_deployment().checkpoint()
    return _leaves(analysis)


def encoder_weight_leaves(resolved: Any, degree: int, stack: str,
                          media: Mapping[str, Any] = MappingProxyType({})
                          ) -> Any:
    """`(by_stack, leaves)` of one encoder stack at one TP degree, from its
    own encode walk (on `media`; the weights are the same whatever the
    media)."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    analysis = Deployment(Program.of(resolved), config=tp_config(degree)).at(
        batch=1, media=dict(media)).during("encode", stack=stack).analyze()
    return _leaves(analysis)


def _leaves(analysis: Any) -> Any:
    from backend.arch.layout.persistent import (
        physical_artifacts, physical_weight_leaves, resident_weights)
    artifacts = physical_artifacts(analysis)
    leaves = physical_weight_leaves(analysis, artifacts=artifacts)
    named = {
        analysis.program.by_uid(uid).aggregation_id: leaf
        for uid, leaf in leaves.items()
    }
    return (resident_weights(artifacts).by_stack, named)


def encoder_stacks(resolved: Any) -> set:
    """The encoder stacks that hold weights of their own (a stack that runs
    another's weights, `Stack.reruns`, holds none)."""
    from backend.arch.eval.program import Program
    return {stack.name for stack in Program.of(resolved).stacks
            if stack.is_encoder and stack.reruns is None}


def measure_snapshot(model: CorpusModel, loader: Optional[Any] = None) -> Dict[str, Any]:
    """{tp: {stack: {"bytes_per_rank", "nodes"}}} for one snapshot's encoder
    stacks; empty when it has none."""
    resolved = model.resolved(loader or shared_loader())
    encoders = encoder_stacks(resolved)
    if not encoders:
        return {}
    out: Dict[str, Any] = {}
    for degree in TP_DEGREES:
        out[str(degree)] = {}
        for stack in sorted(encoders):
            by_stack, leaves = encoder_weight_leaves(resolved, degree, stack)
            out[str(degree)][stack] = {
                "bytes_per_rank": int(by_stack.get(stack, 0)),
                "nodes": {
                    node: int(leaf.resident_bytes_per_rank)
                    for node, leaf in sorted(leaves.items())
                    if leaf.stack == stack
                },
            }
    return out


@dataclass(frozen=True)
class EncoderWeightPin(JsonMapPin):
    def applies(self, model: CorpusModel) -> bool:
        return bool(encoder_stacks(model.resolved()))

    def measure(self, model: CorpusModel) -> Row:
        row = measure_snapshot(model)
        return {model.stem: row} if row else {}


ENCODER_WEIGHT = EncoderWeightPin(
    id="P3",
    title="encoder weight per rank",
    scope=Scope.CONDITIONAL,
    artifact="tests/golden/encoder_weight_per_rank.json",
    owner_tests=(
        "tests/backend/arch/layout/test_encoder_weight_per_rank_golden.py::"
        "TestEncoderWeightPerRankGolden::"
        "test_every_encoder_holds_its_pinned_bytes_per_rank",
    ),
    waits_on=("G4",),
)
