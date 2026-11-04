"""C2: where each stack attaches is declared, not synthesised.

A `group:` entry with an `aux_kind` is a stack of its own. A prediction head
(MTP, speculative) reads a hidden state the backbone produced and must name
it (`inputs:`); an encoder (vision, audio) reads an image or a waveform and
names nothing. A name a stack reads must be produced before it: an earlier
top-level layer, a stream an earlier stack declares (an appended stream, or
an inner one, whose final value is readable by name after its stack — the
schema's rule, `ArchSpec` model-input validation), or a stage inside an
earlier group (a group is a stack of its own and the evaluator resolves
`inputs:` against every stage of the stacks walked before).

The admitter's replay substitutes an older rule for this module through the
three functions below; their signatures are the module's interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

from tools.corpus_pins.check import CheckKind, CorpusCheck
from tools.corpus_pins.corpus import CorpusModel

#: Stacks that predict from the backbone's hidden state. The other `aux_kind`
#: values are encoders, and an encoder's input is not in this graph.
PREDICTION_KINDS = ("mtp", "speculative")


def aux_groups(spec: Any) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """(layer name, aux_kind, declared inputs) for every aux group."""
    out: List[Tuple[str, str, Tuple[str, ...]]] = []

    def walk(layers) -> None:
        for ml in layers:
            if ml.aux_kind:
                out.append((ml.name, ml.aux_kind, tuple(ml.inputs or ())))
            if ml.layers:
                walk(ml.layers)
    walk(spec.model)
    return out


def produced(ml: Any, appended: Dict[Tuple[int, ...], Tuple[str, ...]],
             path: Tuple[int, ...],
             reads: Dict[Tuple[int, ...], Any]) -> Iterator[str]:
    """The names a top-level layer makes readable to the layers after it."""
    yield ml.name
    yield from appended.get(path, ())
    yield from reads[path].inner
    for i, child in enumerate(ml.layers or ()):
        yield from produced(child, appended, (*path, i), reads)


def undeclared_heads(spec: Any) -> List[str]:
    return [layer for layer, kind, ins in aux_groups(spec)
            if kind in PREDICTION_KINDS and not ins]


def attached_encoders(spec: Any) -> List[str]:
    return [f"{layer} reads {list(ins)}" for layer, kind, ins in aux_groups(spec)
            if kind not in PREDICTION_KINDS and ins]


def unresolved_reads(spec: Any) -> List[str]:
    from backend.arch.spec_tree import spine_of_spec
    # The model's inputs are there before any layer runs.
    seen: set = set(spec.model_inputs)
    out = []
    appended, reads = spine_of_spec(spec)
    for i, ml in enumerate(spec.model):
        out.extend(f"{ml.name} -> {ref}" for ref in (ml.inputs or ())
                   if ref not in seen)
        seen.update(produced(ml, appended, (i,), reads))
    return out


@dataclass(frozen=True)
class StackAttachmentCheck(CorpusCheck):
    def violations(self, model: CorpusModel) -> List[str]:
        from tools.corpus_pins.checks.chain_stacks import served_specs
        spec = served_specs()[model.spec_file().stem]
        return ([f"head without inputs: {h}" for h in undeclared_heads(spec)]
                + [f"encoder with inputs: {e}" for e in attached_encoders(spec)]
                + [f"unresolved read: {r}" for r in unresolved_reads(spec)])


_TEST = "tests/backend/arch/eval/test_stack_attachment.py::TestStackAttachment::"
STACK_ATTACHMENT = StackAttachmentCheck(
    id="C2",
    title="stack attachment is declared",
    kind=CheckKind.ADMIT,
    owner_tests=tuple(_TEST + name for name in (
        "test_every_prediction_head_declares_what_it_reads",
        "test_encoders_declare_nothing",
        "test_declared_attachment_names_an_earlier_layer",
    )),
)
