"""Carry a maintainer's decisions from an admission PR onto a new base.

update-pr recomputes an admission PR on the current main (§4.4). A
maintainer may have written the model's decisions on the PR branch: D1 in
`tp_split_decisions.json`, D2 in `copy_contracts.json`, D3 beside the
numbers in `checkpoint_reconciliation.json`, D4 as the note in
`index_refusals.json`. Those entries are the admission's inputs, so they
move to the new base before it runs. A change to any other entry of these
files names something the model does not own; the service then treats the
PR as taken over and stops updating it.

The caller reads every path the branch changed since the bot's last commit
into two directories that mirror the repository, as that commit held it and
as the head holds it; a path that is not a decision file is refused, and the
decision entries that differ are applied onto the files of the tree this
module lives in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from tools.corpus_pins.corpus import ROOT, CorpusModel, relative
from tools.corpus_pins.decisions import copy_contracts, tp_split
from tools.corpus_pins.pin import json_bytes
from tools.corpus_pins.pins.checkpoint_reconciliation import CHECKPOINT_RECONCILIATION, split
from tools.corpus_pins.pins.index_refusals import INDEX_REFUSALS

Entries = Dict[str, Any]


@dataclass(frozen=True)
class DecisionFile:
    """A data file a person writes decisions into, read as entries keyed by
    a string, with the rule that says which entries a model owns."""

    artifact: str
    parse: Callable[[bytes], Entries]
    render: Callable[[Entries], bytes]
    owner: Callable[[str], str]

    def owned(self, key: str, model: CorpusModel, files: Set[str]) -> bool:
        owner = self.owner(key)
        return owner == model.stem or owner in files


def _stem_map(data: bytes) -> Entries:
    return dict(json.loads(data.decode("utf-8")))


def _contracts(data: bytes) -> Entries:
    return {json.dumps(row): True for row in json.loads(data.decode("utf-8"))}


def _render_contracts(entries: Entries) -> bytes:
    return json_bytes(sorted((json.loads(k) for k in entries), key=json.dumps))


DECISION_FILES: Tuple[DecisionFile, ...] = (
    DecisionFile(tp_split.ARTIFACT, _stem_map, json_bytes, lambda k: k),
    DecisionFile(copy_contracts.ARTIFACT, _contracts, _render_contracts,
                 lambda k: json.loads(k)[0]),
    DecisionFile(CHECKPOINT_RECONCILIATION.artifact, CHECKPOINT_RECONCILIATION.parse,
                 CHECKPOINT_RECONCILIATION.render, lambda k: split(k)[1]),
    DecisionFile(INDEX_REFUSALS.artifact, INDEX_REFUSALS.parse, INDEX_REFUSALS.render,
                 lambda k: k),
)
ARTIFACTS = tuple(f.artifact for f in DECISION_FILES)


def _paths(root: Path) -> Set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _read(root: Path, artifact: str) -> Optional[bytes]:
    path = root / artifact
    return path.read_bytes() if path.exists() else None


def carry(model: CorpusModel, old_root: Path, new_root: Path,
          tree: Path = ROOT) -> Tuple[Dict[str, List[str]], List[str]]:
    """Apply onto `tree` the decision entries that changed from `old_root`
    to `new_root`. Returns `({artifact: changed keys}, problems)`; with any
    problem nothing is written."""
    files = {relative(p) for p in model.bundle() if p.suffix in (".yaml", ".yml")}
    problems: List[str] = [
        f"{path}: not a decision file" for path in sorted(
            _paths(old_root) | _paths(new_root)) if path not in ARTIFACTS]
    writes: Dict[str, bytes] = {}
    carried: Dict[str, List[str]] = {}
    for spec in DECISION_FILES:
        old_data, new_data = _read(old_root, spec.artifact), _read(new_root, spec.artifact)
        if old_data == new_data:
            continue
        if old_data is None or new_data is None:
            problems.append(f"{spec.artifact}: present on one side only")
            continue
        try:
            old, new = spec.parse(old_data), spec.parse(new_data)
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"{spec.artifact}: cannot be read: {exc}")
            continue
        changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        foreign = [k for k in changed if not spec.owned(k, model, files)]
        if foreign:
            problems.append(f"{spec.artifact}: entries {model.stem} does not own changed: {foreign}")
            continue
        base_data = (tree / spec.artifact).read_bytes()
        base = spec.parse(base_data)
        if spec.render(base) != base_data:
            problems.append(f"{spec.artifact}: the base does not read back byte for byte")
            continue
        merged = apply(base, new, changed)
        writes[spec.artifact] = spec.render(merged)
        carried[spec.artifact] = changed
    if problems:
        return {}, problems
    for artifact, data in writes.items():
        (tree / artifact).write_bytes(data)
    return carried, []


def apply(base: Mapping[str, Any], new: Mapping[str, Any], changed: List[str]) -> Entries:
    out = dict(base)
    for k in changed:
        if k in new:
            out[k] = new[k]
        else:
            out.pop(k, None)
    return out
