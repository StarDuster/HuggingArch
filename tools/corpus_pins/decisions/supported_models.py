"""D5: `supported_models.txt`. Listing a model there moves the README's
verified-id and org counts, which an admission cannot explain, so it lands in
a pull request of its own after the admission, with both READMEs' counts in
the same change. The admitter only says whether the model is listed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.decision import Decision, DecisionState, Verdict
from tools.corpus_pins.pins.readme_counts import verified_ids


@dataclass(frozen=True)
class SupportedModelsDecision(Decision):
    def judge(self, model: CorpusModel, measured: Dict[str, Any]) -> Verdict:
        listed = model.model_id in verified_ids()
        return Verdict(DecisionState.INFO,
                       [] if listed else ["not listed; list it in a later pull request "
                                          "that also moves both READMEs' counts"],
                       {"listed": listed})


SUPPORTED_MODELS = SupportedModelsDecision(
    id="D5", title="supported_models.txt", artifacts=("backend/arch/supported_models.txt",))
