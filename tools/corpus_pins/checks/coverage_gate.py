"""C4: the coverage gate. A spec is valid iff every checkpoint tensor maps
to a spec node and the byte diff is 0 — the regression guard that makes
`unmatched_hf_tensors` a ship gate."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from tools.corpus_pins.check import CheckKind, CorpusCheck
from tools.corpus_pins.corpus import CorpusModel


def validate_offline(model_id: str) -> Dict[str, Any]:
    """`validate(model_id)` with the hub forced offline: the snapshot carries
    everything validate needs (config, the safetensors header)."""
    from backend.arch.validate import validate
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return validate(model_id)
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


def gate_errors(report: Dict[str, Any]) -> List[str]:
    """Empty when the report is valid; its errors otherwise."""
    return [] if report["valid"] else [str(e) for e in report["errors"]]


@dataclass(frozen=True)
class CoverageGateCheck(CorpusCheck):
    def violations(self, model: CorpusModel) -> List[str]:
        return gate_errors(validate_offline(model.model_id))


COVERAGE_GATE = CoverageGateCheck(
    id="C4",
    title="coverage gate",
    kind=CheckKind.ADMIT,
    owner_tests=(
        "tests/backend/arch/validate/test_validator_golden.py::TestCoverageGate::"
        "test_every_snapshot_is_valid",
    ),
)
