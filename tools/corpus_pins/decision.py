"""A decision: a value or statement only a person can give — an external
framework's split semantics, a residual's cause, a curatorial choice. The
admitter computes what it can (numbers, candidate tuples, node tables) as a
suggestion, and once a person has written the decision it checks that the
decision agrees with the measurement."""
from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from tools.corpus_pins.corpus import CorpusModel


class DecisionState(str, enum.Enum):
    #: the model needs this decision and nobody has written it
    PENDING = "pending"
    #: written, and the measurement agrees
    CONSISTENT = "consistent"
    #: written, and the measurement disagrees
    INCONSISTENT = "inconsistent"
    #: informational; blocks nothing
    INFO = "info"
    #: the model needs no decision here
    NOT_NEEDED = "not_needed"


@dataclass(frozen=True)
class Verdict:
    state: DecisionState
    detail: List[str] = field(default_factory=list)
    suggestion: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision(abc.ABC):
    id: str
    title: str
    #: the data file (or files) a person writes the decision into
    artifacts: Tuple[str, ...]
    #: whether a pending or inconsistent state holds the G4 dependency
    gates_g4: bool = False

    @abc.abstractmethod
    def judge(self, model: CorpusModel, measured: Dict[str, Any]) -> Verdict:
        """The decision's state for `model`, given what the admitter measured
        (pin rows keyed by pin id), with the suggestion a person would
        need to write it."""
