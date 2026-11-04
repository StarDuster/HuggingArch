"""D2: the expert-weight encodings, an explicit closed set.

Every `expert_weight` declaration in the component and model YAML — how the
checkpoint stores an expert bank's `n_expert_copies` axis, `unrolled` or
`leading_axis` — is an encoding a person accepted on purpose; the set lives
in `tests/golden/pins/copy_contracts.json` as 3-tuples
`(file, node, encoding)`. The admitter proposes the exact tuples a candidate
declares and never adds them itself: a table a program extends is no longer a
record of acceptance."""
from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import yaml

from tools.corpus_pins.corpus import ARCH_DIR, COMPONENTS_DIR, MODELS_DIR, ROOT, CorpusModel, relative
from tools.corpus_pins.decision import Decision, DecisionState, Verdict

ARTIFACT = "tests/golden/pins/copy_contracts.json"
Contract = Tuple[str, str, str]


def load() -> Set[Contract]:
    return {tuple(row) for row in json.loads((ROOT / ARTIFACT).read_text(encoding="utf-8"))}


def declarations(paths: Iterable[Path] = ()) -> List[Tuple[str, str, Dict[str, Any]]]:
    """`(file, node name, node)` for every node that declares
    `expert_weight`, over `paths` (every component and model yaml when
    empty)."""
    paths = list(paths) or (sorted(COMPONENTS_DIR.rglob("*.yaml"))
                            + sorted(MODELS_DIR.rglob("*.yaml")))
    out = []
    for path in paths:
        payload = yaml.safe_load(Path(path).read_text()) or {}

        def visit(value: Any, breadcrumb: str = "") -> None:
            if isinstance(value, dict):
                if value.get("expert_weight"):
                    out.append((str(Path(path).resolve().relative_to(ARCH_DIR)),
                                str(value.get("name") or breadcrumb), value))
                for key, child in value.items():
                    visit(child, str(key))
            elif isinstance(value, list):
                for child in value:
                    visit(child, breadcrumb)

        visit(payload)
    return out


def contract(path: str, name: str, node: Dict[str, Any]) -> Contract:
    return (path, name, node["expert_weight"])


def encoding_counts(contracts: Iterable[Contract]) -> collections.Counter:
    """How many declarations each encoding has."""
    return collections.Counter(c[2] for c in contracts)


@dataclass(frozen=True)
class CopyContractsDecision(Decision):
    def judge(self, model: CorpusModel, measured: Dict[str, Any]) -> Verdict:
        bundle = [p for p in model.bundle() if p.suffix in (".yaml", ".yml")]
        files = {relative(p) for p in bundle}
        declared = {contract(*d) for d in declarations(bundle)}
        recorded = {c for c in load() if c[0] in files}
        if not declared and not recorded:
            return Verdict(DecisionState.NOT_NEEDED)
        others = load() - recorded
        before, after = encoding_counts(others), encoding_counts(others | declared)
        suggestion = {
            "contracts": sorted([list(c) for c in declared], key=json.dumps),
            "counts": {k: f"{before[k]} -> {after[k]}"
                       for k in sorted(set(before) | set(after)) if before[k] != after[k]},
        }
        missing = sorted(declared - recorded, key=json.dumps)
        stale = sorted(recorded - declared, key=json.dumps)
        if stale:
            return Verdict(DecisionState.INCONSISTENT,
                           [f"recorded but not declared: {list(c)}" for c in stale], suggestion)
        if missing:
            return Verdict(DecisionState.PENDING,
                           [f"declared but not recorded: {list(c)}" for c in missing], suggestion)
        return Verdict(DecisionState.CONSISTENT, [], suggestion)


COPY_CONTRACTS = CopyContractsDecision(
    id="D2", title="expert-weight encodings", artifacts=(ARTIFACT,), gates_g4=True)
