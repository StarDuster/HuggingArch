"""C1: every chain stack carries a stream.

A stack threads its block's output into its next iteration only when it
declares so (`streams: { <name>: { carries: true } }`); there is no default.
`tools.chain_stacks.chain_stacks()` lists the stacks that repeat a block from
the authored YAML, independently of the loader, and each must carry a stream
in the spec the loader serves. The stream's name is the author's.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

from tools.corpus_pins.check import CheckKind, CorpusCheck
from tools.corpus_pins.corpus import CorpusModel


@functools.cache
def served_specs() -> Dict[str, Any]:
    """Every spec the loader serves, keyed by file stem, with machine-local
    custom specs ignored."""
    from backend.arch.loader import SpecLoader
    with mock.patch.object(SpecLoader, "_load_custom_spec_if_any",
                           lambda self: None):
        return SpecLoader().all_specs()


def layer_at(spec: Any, path: tuple) -> Any:
    layers = spec.model
    for name in path:
        layer = next(entry for entry in layers if entry.name == name)
        layers = layer.layers or []
    return layer


def _int_path(spec: Any, names: tuple) -> Tuple[int, ...]:
    layers = spec.model
    path: List[int] = []
    for name in names:
        index = next(i for i, entry in enumerate(layers) if entry.name == name)
        path.append(index)
        layers = layers[index].layers or []
    return tuple(path)


def stack_violations(stacks: List[Any]) -> List[str]:
    from backend.arch.spec_tree import spine_of_spec
    specs = served_specs()
    out = []
    for stack in stacks:
        spec = specs[Path(stack.file).stem]
        _appended, stack_reads = spine_of_spec(spec)
        reads = stack_reads[_int_path(spec, stack.path)]
        if reads.carried is None:
            out.append(stack.row())
    return out


@dataclass(frozen=True)
class ChainStacksCheck(CorpusCheck):
    def violations(self, model: CorpusModel) -> List[str]:
        from tools.chain_stacks import chain_stacks
        files = {path.name for path in model.spec_files()}
        return stack_violations([s for s in chain_stacks() if s.file in files])


CHAIN_STACKS = ChainStacksCheck(
    id="C1",
    title="every chain stack carries a stream",
    kind=CheckKind.ADMIT,
    owner_tests=(
        "tests/backend/arch/test_chain_stacks_carry_hidden.py::test_every_chain_stack_carries_a_stream",
    ),
)
