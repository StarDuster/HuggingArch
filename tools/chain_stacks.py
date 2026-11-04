"""List every chain stack in the shared model corpus.

A chain stack is a model layer that repeats a block and hands the block's
output to its next iteration: the layer that declares the stack's carried
stream (`streams: { hidden: { carries: true } }`). Read off the
authored YAML, independent of the loader, so it can check the corpus the
loader reads.

Rules:
  * a file is listed when it owns a `model:` section. `base:` merges shallowly
    and a child's `model:` replaces the parent's whole
    (`loader.merge_base_documents`), so a file with `base:` and no `model:`
    inherits its parent's stacks, while one that restates `model:`
    (`qwen3_next_mtp`) owns its own;
  * the whole model tree is walked;
  * a layer with `repeat:` and `template:` or `per_layer:` is a chain stack;
  * a repeated `group:` whose children do not repeat themselves is one
    (DeepSeek's `mtp`), declared on the group;
  * under a repeated group whose child repeats itself (Nemotron-3-Ultra's
    `mtp.layers`) the innermost repeated layer is the stack; the group's
    `repeat:` is reported as `outer`;
  * a repeated layer that reads an appended stream at its block boundary
    (`inputs:` names a stream some layer declares with `from:`), or appends
    its own block's output (`from: out`), runs its iterations side by side
    and hands nothing to the next one, so it is not a chain stack
    (Qwen3-VL's `visual.deepstack_merger_list`: merger j reads element j of
    `deepstack`). `spec_migrate._carry_hidden` skips the same stacks.

Run from a checkout root to print the rows as TSV (no header):

    .venv/bin/python -m tools.chain_stacks
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import yaml

MODELS = Path(__file__).resolve().parents[1] / "backend" / "arch" / "models"


@dataclass(frozen=True)
class ChainStack:
    file: str
    #: layer names from the top of `model:` down to the stack
    path: Tuple[str, ...]
    kind: str                     # template | per_layer | group-repeat
    aux_kind: Optional[str]
    outer: Optional[str]          # enclosing group's `repeat:`, if any

    def row(self) -> str:
        return "\t".join((self.file, self.path[-1], self.kind,
                          self.aux_kind or "", self.outer or ""))


def _layers(layers) -> Iterator[dict]:
    for layer in layers or ():
        yield layer
        yield from _layers(layer.get("layers"))


def _appended(layers) -> Set[str]:
    """The appended streams (`from:`) a model declares, anywhere in it."""
    return {name for layer in _layers(layers)
            for name, decl in (layer.get("streams") or {}).items()
            if isinstance(decl, dict) and "from" in decl}


def _side_by_side(layer: dict, appended: Set[str]) -> bool:
    return bool(appended & set(layer.get("inputs") or ())) or any(
        isinstance(decl, dict) and decl.get("from") == "out"
        for decl in (layer.get("streams") or {}).values())


def _walk(file: str, layers, prefix: Tuple[str, ...], aux: Optional[str],
          outer: Optional[str], appended: Set[str]) -> Iterator[ChainStack]:
    for layer in layers or ():
        path = (*prefix, layer["name"])
        children = layer.get("layers")
        if children:
            repeat = layer.get("repeat")
            kind = layer.get("aux_kind") or aux
            if repeat is not None and not any(
                    c.get("repeat") is not None for c in children) and not (
                    _side_by_side(layer, appended)):
                yield ChainStack(file, path, "group-repeat", kind, None)
            yield from _walk(file, children, path, kind,
                             str(repeat) if repeat is not None else outer,
                             appended)
            continue
        if layer.get("repeat") is not None and (
                layer.get("template") or layer.get("per_layer")) and not (
                _side_by_side(layer, appended)):
            kind = "per_layer" if layer.get("per_layer") else "template"
            yield ChainStack(file, path, kind, aux, outer)


def chain_stacks(models: Path = MODELS) -> List[ChainStack]:
    out: List[ChainStack] = []
    for path in sorted(models.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text()) or {}
        if "model" not in doc:
            continue
        out.extend(_walk(path.name, doc["model"], (), None, None,
                         _appended(doc["model"])))
    return out


if __name__ == "__main__":
    for stack in chain_stacks():
        sys.stdout.write(stack.row() + "\n")
