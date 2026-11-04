"""P7: the nodes whose op templates do not describe them (index-signature
refusals), per model.

`shape_in` is an einsum signature, and `IndexSignature` labels every axis of
every input and the output. When no template of an op fits a node at the rank
it is applied, the signature is refused and the caller falls back to
positional dispatch. Each refusal belongs to a family `op@rank`. A family is
explained when another model already brings the same family; an unexplained
family needs a person's note (decision D4).

The file keeps `{stem: {count, families, note}}`. The test holds each model
at or under its count and the corpus total within 20 of the sum.
"""
from __future__ import annotations

import collections
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.pin import Doc, JsonMapPin, Row, Scope

#: how far under the pinned total the corpus may fall before the pins must
#: be lowered, so an improvement cannot drift back unnoticed
SLACK = 20


def family(op: str, rank: int) -> str:
    return f"{op}@{rank}"


class RefusalCounter:
    """Counts the refusals `validate(model)` meets, by family.

    The walk asks `dim_map` in forward mode (with `in_ranks` / `in_shapes`)
    as well as in checking mode; a refusal is counted once, in checking
    mode, where it means a template does not fit the node."""

    def __init__(self) -> None:
        self.families: collections.Counter = collections.Counter()

    def __enter__(self) -> "RefusalCounter":
        import backend.arch.axis_propagation as ax
        original = ax.dim_map
        self._original = original

        def counting(op, node, *args, **kwargs):
            if kwargs.get("in_ranks") is not None or kwargs.get("in_shapes") is not None:
                return original(op, node, *args, **kwargs)
            spec = ax._lookup(op)
            rank_out = ax._rank_out(node)
            if (spec is not None and rank_out is not None
                    and spec.name not in ax._EXPLICIT_DM
                    and ax.signature_for(spec, node, rank_out) is None):
                self.families[family(spec.name, rank_out)] += 1
            return original(op, node, *args, **kwargs)

        ax.dim_map = counting
        return self

    def __exit__(self, *exc: Any) -> None:
        import backend.arch.axis_propagation as ax
        ax.dim_map = self._original


def refusals(model: CorpusModel) -> Dict[str, int]:
    """`{family: count}` for one model. A model whose validate raises is a
    failure of the corpus, and it is raised."""
    from backend.arch.validate._orchestrator import validate
    # This pin counts template fits, not model-card prose. Keep the corpus
    # lane hermetic: an offline Hub read uses a cached card when one exists
    # and reports "no card" otherwise, instead of timing out behind every
    # model while a full regression run contends for network/proxy.
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        with RefusalCounter() as counter:
            validate(model.model_id)
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline
    return dict(sorted(counter.families.items()))


def unexplained(families: Dict[str, int], known: Iterator[str]) -> List[str]:
    """The families no other model already brings."""
    seen = set(known)
    return [fam for fam in families if fam not in seen]


@dataclass(frozen=True)
class IndexRefusalsPin(JsonMapPin):
    def measure(self, model: CorpusModel) -> Row:
        families = refusals(model)
        return {model.stem: {"count": sum(families.values()), "families": families,
                             "note": ""}}

    def value_matches(self, recorded: Any, measured: Any) -> bool:
        """As the test holds it: at or under the model's pinned count."""
        return measured["count"] <= recorded["count"]

    def totals(self, doc: Doc) -> Tuple[int, int]:
        """(pinned sum, floor): the corpus total may not exceed the sum nor
        fall more than `SLACK` under it."""
        pinned = sum(entry["count"] for entry in doc.values())
        return pinned, pinned - SLACK


INDEX_REFUSALS = IndexRefusalsPin(
    id="P7",
    title="index-signature refusals",
    scope=Scope.REQUIRED,
    artifact="tests/golden/pins/index_refusals.json",
    owner_tests=(
        "tests/backend/arch/test_index_signature_coverage.py::"
        "TestIndexSignatureCoverage::test_the_templates_still_describe_what_they_did",
    ),
    waits_on=("G4",),
)
