"""D4: a new index-signature refusal family. A family no other model already
brings needs a person's note in the model's `index_refusals.json` entry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.decision import Decision, DecisionState, Verdict
from tools.corpus_pins.pins.index_refusals import INDEX_REFUSALS, unexplained


def unexplained_families(model: CorpusModel, families: Dict[str, int]) -> list:
    doc = INDEX_REFUSALS.load()
    known = (fam for stem, entry in doc.items() if stem != model.stem
             for fam in entry["families"])
    return unexplained(families, known)


@dataclass(frozen=True)
class RefusalFamilyDecision(Decision):
    def judge(self, model: CorpusModel, measured: Dict[str, Any]) -> Verdict:
        row = measured.get("P7") or INDEX_REFUSALS.measure(model)
        families = row[model.stem]["families"]
        new = unexplained_families(model, families)
        if not new:
            return Verdict(DecisionState.NOT_NEEDED, [], {"families": families})
        entry = INDEX_REFUSALS.load().get(model.stem)
        suggestion = {"families": families, "unexplained": new}
        if entry is None or not entry["note"]:
            return Verdict(DecisionState.PENDING, [f"unexplained {new}"], suggestion)
        return Verdict(DecisionState.CONSISTENT, [], suggestion)


#: The notes are written into P7's own file, beside the counts.
REFUSAL_FAMILY = RefusalFamilyDecision(
    id="D4", title="new refusal family", artifacts=())
