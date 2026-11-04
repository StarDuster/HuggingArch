"""A corpus check: an invariant every model must hold, with no output of its
own. A model that violates one has a defective spec or has found a gap in
the rule; either way a person changes code. The admitter reports it and
writes nothing."""
from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tools.corpus_pins.corpus import Corpus, CorpusModel


class CheckKind(str, enum.Enum):
    #: run by the admitter on the one model before admission
    ADMIT = "admit"
    #: run only by CI over the whole corpus
    CI_ONLY = "ci_only"


@dataclass(frozen=True)
class CorpusCheck(abc.ABC):
    id: str
    title: str
    kind: CheckKind
    owner_tests: Tuple[str, ...]

    @abc.abstractmethod
    def violations(self, model: CorpusModel) -> List[str]:
        """What `model` breaks; empty when it holds the invariant."""

    def violations_all(self) -> Dict[str, List[str]]:
        return {model.stem: found for model in Corpus.models()
                if (found := self.violations(model))}


@dataclass(frozen=True)
class CiOnlyCheck(CorpusCheck):
    """A corpus-walking module the admitter does not run: its invariant is
    held by the test suite — the `regression` run, for a module marked
    `regression`."""

    def violations(self, model: CorpusModel) -> List[str]:
        return []
