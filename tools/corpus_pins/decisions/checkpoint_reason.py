"""D3: why a checkpoint residual exists. The reconciliation's `KNOWN`,
`MEASURED` and `NO_ANCHOR` entries carry a person's reason beside the
numbers; the admitter computes the numbers and checks a written entry
against them."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.decision import Decision, DecisionState, Verdict
from tools.corpus_pins.pins.checkpoint_reconciliation import (
    CHECKPOINT_RECONCILIATION, REASONED, split)


@dataclass(frozen=True)
class CheckpointReasonDecision(Decision):
    def judge(self, model: CorpusModel, measured: Dict[str, Any]) -> Verdict:
        row = measured.get("P4") or CHECKPOINT_RECONCILIATION.measure(model)
        needed = {k: v for k, v in row.items() if split(k)[0] in REASONED}
        if not needed:
            return Verdict(DecisionState.NOT_NEEDED)
        doc = CHECKPOINT_RECONCILIATION.load()
        pending, off = [], []
        for key, numbers in sorted(needed.items()):
            if key not in doc:
                pending.append(f"{key}: {numbers}")
                continue
            recorded = {k: v for k, v in doc[key].items() if k in numbers}
            if recorded != numbers:
                off.append(f"{key}: recorded {recorded}, measured {numbers}")
        state = (DecisionState.INCONSISTENT if off else
                 DecisionState.PENDING if pending else DecisionState.CONSISTENT)
        return Verdict(state, off + pending, {"entries": needed})


#: The reasons are written into P4's own file, beside the numbers.
CHECKPOINT_REASON = CheckpointReasonDecision(
    id="D3", title="checkpoint residual reason", artifacts=())
