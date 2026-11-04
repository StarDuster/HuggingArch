"""C5: the checkpoint invariants besides the per-model reconciliation.

For one model: its weight column is exact (what the spec says the checkpoint
HOLDS is what it holds); every quant sibling a format implies lands on a
name the header contains; every `quant_context.overrides` row decides some
stored tensor's format; every dtype the header uses has a width."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from tools.corpus_pins.check import CheckKind, CorpusCheck
from tools.corpus_pins.corpus import CorpusModel, shared_loader
from tools.corpus_pins.pins.checkpoint_reconciliation import reconcile


def analysis_at_128(model: CorpusModel, resolved: Any = None) -> Any:
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    resolved = resolved if resolved is not None else model.resolved(shared_loader())
    return Deployment(Program.of(resolved)).at(batch=1, input_tokens=128).checkpoint()


def weight_delta(model: CorpusModel) -> int:
    return int(reconcile(model.stem).weight_delta)


def unknown_dtypes(model: CorpusModel) -> tuple:
    return reconcile(model.stem).unknown_dtypes


def orphan_siblings(model: CorpusModel) -> List[str]:
    """Quant siblings named after no tensor the checkpoint header contains.
    The byte columns cannot see a misnamed sibling (its bytes count whatever
    it is called), and neither can the coverage gate (a spec stamp with no
    counterpart is not a tensor). Read off the one name join's spec-to-header
    direction."""
    from backend.arch.validate._matching import CheckpointJoin
    join = CheckpointJoin.of(analysis_at_128(model), model.header)
    return [row.name for row in join.absent if row.is_metadata]


def dead_overrides(model: CorpusModel) -> List[str]:
    """`quant_context.overrides` rows that decide no stored tensor's format:
    they change no byte, so neither byte column can see them, and they still
    read as the dtype those tensors have."""
    from backend.arch.validate._packing_consistency import run_quant_override_check
    resolved = model.resolved(shared_loader())
    quant = resolved.spec.quant_context
    if quant is None or not quant.overrides:
        return []
    return [f.detail for f in run_quant_override_check(analysis_at_128(model, resolved))]


@dataclass(frozen=True)
class CheckpointInvariantsCheck(CorpusCheck):
    def violations(self, model: CorpusModel) -> List[str]:
        out = []
        if weight_delta(model):
            out.append(f"weight column off by {weight_delta(model)}")
        out += [f"orphan quant sibling {name}" for name in orphan_siblings(model)]
        out += [f"dead quant override: {detail}" for detail in dead_overrides(model)]
        out += [f"dtype without a width: {dtype}" for dtype in unknown_dtypes(model)]
        return out


_TEST = "tests/backend/arch/eval/test_eval_checkpoint.py::TestCheckpointInvariant::"
CHECKPOINT_INVARIANTS = CheckpointInvariantsCheck(
    id="C5",
    title="checkpoint invariants",
    kind=CheckKind.ADMIT,
    owner_tests=tuple(_TEST + name for name in (
        "test_the_weight_column_is_clean_almost_everywhere",
        "test_every_quant_sibling_is_named_after_a_tensor_that_exists",
        "test_every_quant_override_decides_a_stored_tensor",
        "test_every_dtype_the_headers_use_has_a_width",
    )),
)
