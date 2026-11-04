"""`python -m tools.corpus_admit verify`: the corpus pins agree with the
corpus, without running the test suite.

Every CI lane runs it:
  * missing keys — every snapshot has its key in every required pin;
  * orphan keys — no pin keeps a key whose snapshot (or component file) is
    gone;
  * aggregate values — the README counts, recomputed, are stated once each
    in both READMEs.
None of these measures a model, so they cost seconds.

The spec-only lane adds `--changed --paths-file FILE`: the models the diff
touches (a changed, renamed or deleted snapshot or model yaml, the identity
files whose `base:` chain or `drafts:` reach a changed file) are measured
pin by pin and their keys compared with the files, and their recorded
TP-split decisions are checked against the measurement again. Together with
the full suite on every other lane, no change to the corpus reaches main
with its keys checked but not its values.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tools.corpus_pins import REGISTRY
from tools.corpus_pins.corpus import ARCH_DIR, ROOT, Corpus, CorpusModel
from tools.corpus_pins.decision import DecisionState
from tools.corpus_pins.decisions.tp_split import TP_SPLIT
from tools.corpus_pins.pin import Pin
from tools.corpus_pins.pins.readme_counts import README_COUNTS, counts


def key_problems() -> List[str]:
    """Missing and orphan keys over every keyed pin."""
    stems = set(Corpus.stems())
    components = {str(p.relative_to(ARCH_DIR)) for p in Corpus.component_files()}
    problems: List[str] = []
    for pin in REGISTRY.pins:
        if not isinstance(pin, Pin):
            continue
        doc = pin.load()
        for stem in sorted(stems):
            missing = pin.missing_keys(doc, CorpusModel(stem))
            if missing:
                problems.append(f"{pin.id} {pin.artifact}: {stem} has no key {missing}")
        problems += [f"{pin.id} {pin.artifact}: orphan key {key!r}" for key in sorted(doc)
                     if not pin.key_exists(key, stems, components)]
    return problems


def aggregate_problems() -> List[str]:
    return README_COUNTS.verify(README_COUNTS.texts(), counts())


def _gate_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ci_spec_gate", ROOT / "scripts" / "ci_spec_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def affected_models(paths: Sequence[str]) -> List[CorpusModel]:
    """The models with a snapshot whose spec a changed path defines (the
    spec gate's own selection: snapshots, model yamls, drafts, base chains)."""
    ids = _gate_module().models_to_validate_for_paths(list(paths), root=ROOT)
    return [CorpusModel(model_id.replace("/", "__", 1)) for model_id in ids]


def value_problems(model: CorpusModel) -> List[str]:
    """Each pin's keys for one model, measured, against its file."""
    problems: List[str] = []
    for pin in REGISTRY.pins:
        if not isinstance(pin, Pin) or not pin.applies(model):
            continue
        doc = pin.load()
        for key, value in sorted(pin.measure(model).items()):
            if key not in doc:
                problems.append(f"{pin.id}: {model.stem} has no key {key!r}")
            elif not pin.value_matches(doc[key], value):
                problems.append(f"{pin.id}: {model.stem} {key!r} differs from {pin.artifact}")
    verdicts = TP_SPLIT.judge_model(model)
    problems += [f"D1 {model.stem} {stack or '(backbone)'}: {line}"
                 for stack, verdict in verdicts.items()
                 if verdict.state is DecisionState.INCONSISTENT for line in verdict.detail]
    return problems


def verify(changed: Sequence[str] = ()) -> Dict[str, List[str]]:
    out = {"keys": key_problems(), "aggregate": aggregate_problems()}
    if changed:
        out["values"] = [line for model in affected_models(changed)
                         for line in value_problems(model)]
    return out


def read_paths(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
