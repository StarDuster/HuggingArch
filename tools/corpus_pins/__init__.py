"""The corpus pins: every closed per-model table the tests compare, every
corpus-wide invariant, every decision a person owns, and every curated
subset, registered once.

`REGISTRY` is the one list. The tests that guard a pin import its measurement
and its file from here; the admitter (`tools/corpus_admit.py`) extends a pin
with a new model's keys through the same methods; the registry test
(`tests/meta/test_corpus_pin_registry.py`) holds every golden file and every
corpus-walking test module to having an owner here.

This package imports no pytest: the admitter runs where tests are not
installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

from tools.corpus_pins.check import CorpusCheck
from tools.corpus_pins.checks.arch_view_graph import ARCH_VIEW_GRAPH
from tools.corpus_pins.checks.chain_stacks import CHAIN_STACKS
from tools.corpus_pins.checks.checkpoint_invariants import CHECKPOINT_INVARIANTS
from tools.corpus_pins.checks.coverage_gate import COVERAGE_GATE
from tools.corpus_pins.checks.encoder_tp import ENCODER_TP
from tools.corpus_pins.checks.route_zone import ROUTE_ZONE
from tools.corpus_pins.checks.stack_attachment import STACK_ATTACHMENT
from tools.corpus_pins.curated import CI_ONLY_CHECKS, CURATED_OUT_OF_SCOPE, PARAMETERS
from tools.corpus_pins.decision import Decision
from tools.corpus_pins.decisions.checkpoint_reason import CHECKPOINT_REASON
from tools.corpus_pins.decisions.copy_contracts import COPY_CONTRACTS
from tools.corpus_pins.decisions.refusal_family import REFUSAL_FAMILY
from tools.corpus_pins.decisions.supported_models import SUPPORTED_MODELS
from tools.corpus_pins.decisions.tp_split import TP_SPLIT
from tools.corpus_pins.env import ADMIT_PYTHON
from tools.corpus_pins.pin import CuratedSet, PinBase
from tools.corpus_pins.pins.checkpoint_reconciliation import CHECKPOINT_RECONCILIATION
from tools.corpus_pins.pins.encode_surface import ENCODE_SURFACE
from tools.corpus_pins.pins.encoder_weight import ENCODER_WEIGHT
from tools.corpus_pins.pins.ep8_checkpoint import EP8_CHECKPOINT
from tools.corpus_pins.pins.gate_surfaces import STORAGE_SURFACE, TENSOR_NAMES
from tools.corpus_pins.pins.index_refusals import INDEX_REFUSALS
from tools.corpus_pins.pins.readme_counts import README_COUNTS
from tools.corpus_pins.pins.route_components import ROUTE_COMPONENTS
from tools.corpus_pins.pins.sol_surface import SOL_SURFACE
from tools.corpus_pins.pins.spec_match import SPEC_MATCH


PINS: Tuple[PinBase, ...] = (
    SPEC_MATCH, SOL_SURFACE, ENCODER_WEIGHT, CHECKPOINT_RECONCILIATION,
    EP8_CHECKPOINT, ROUTE_COMPONENTS, INDEX_REFUSALS, README_COUNTS,
    TENSOR_NAMES, STORAGE_SURFACE, ENCODE_SURFACE,
)
CHECKS: Tuple[CorpusCheck, ...] = (
    CHAIN_STACKS, STACK_ATTACHMENT, ROUTE_ZONE, COVERAGE_GATE,
    CHECKPOINT_INVARIANTS, ENCODER_TP, ARCH_VIEW_GRAPH, *CI_ONLY_CHECKS,
)
DECISIONS: Tuple[Decision, ...] = (
    TP_SPLIT, COPY_CONTRACTS, CHECKPOINT_REASON, REFUSAL_FAMILY, SUPPORTED_MODELS,
)

Entry = Union[PinBase, CorpusCheck, Decision, CuratedSet]


@dataclass(frozen=True)
class Registry:
    pins: Tuple[PinBase, ...]
    checks: Tuple[CorpusCheck, ...]
    decisions: Tuple[Decision, ...]
    curated: Tuple[CuratedSet, ...]

    def pin(self, pin_id: str) -> PinBase:
        (found,) = [p for p in self.pins if p.id == pin_id]
        return found


REGISTRY = Registry(PINS, CHECKS, DECISIONS, CURATED_OUT_OF_SCOPE)

__all__ = ["ADMIT_PYTHON", "PARAMETERS", "REGISTRY", "Registry"]
