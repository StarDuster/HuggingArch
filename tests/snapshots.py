"""The one way a test gets a corpus snapshot.

Two facts, two outcomes, and every test used to spell them itself (90
hand-written `skipTest("... snapshot not present")` sites): a snapshot that is
not checked out is a SKIP — the test has nothing to run on — and a snapshot
that is present but does not resolve is a FAILURE — the corpus is broken and
a skip would hide it. The hand-written form conflated the two (`if resolved
is None: skipTest(...)` skipped both). This module keeps them apart, and
`tests/meta` refuses a new hand-written snapshot skip.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Optional

# The snapshot directory, the process-wide loader and the shared resolution /
# program / walk are the corpus's (`tools/corpus_pins/corpus.py`), so a test
# and the admitter read the corpus the same way.
from tools.corpus_pins.corpus import (
    CorpusModel, SNAPSHOT_DIR, shared_loader, shared_walk)

__all__ = ["SNAPSHOT_DIR", "any_snapshot", "backbone_call", "bench_call",
           "own_walk", "resolve_snapshot", "shared_loader", "shared_program",
           "shared_walk", "snapshot_config", "snapshot_header",
           "snapshot_path", "snapshot_present"]


def snapshot_path(name: str) -> Path:
    """`name` may be a model id (`org/model`), a stem (`org__model`) or a
    filename (`org__model.json`)."""
    stem = name[:-5] if name.endswith(".json") else name.replace("/", "__")
    return SNAPSHOT_DIR / f"{stem}.json"

def snapshot_present(name: str) -> bool:
    return snapshot_path(name).exists()

def snapshot_config(name: str, overlay_dir: Any = None) -> dict:
    """The snapshot's HF config. Skips the calling test when the snapshot is
    not checked out; a present snapshot that does not parse raises.
    `overlay_dir` is accepted for signature parity with `fetch_config` and
    unused: a snapshot is already the resolved record."""
    path = snapshot_path(name)
    if not path.exists():
        raise unittest.SkipTest(f"{name}: snapshot not present")
    return json.loads(path.read_text(encoding="utf-8"))["config"]

def snapshot_header(name: str) -> Any:
    """The snapshot's safetensors header (`hf.header.SafetensorsHeader`).
    Skips when the snapshot is not checked out."""
    from backend.hf.header import SafetensorsHeader
    path = snapshot_path(name)
    if not path.exists():
        raise unittest.SkipTest(f"{name}: snapshot not present")
    return SafetensorsHeader.from_json(
        json.loads(path.read_text(encoding="utf-8"))["metadata"]["header"])

def resolve_snapshot(name: str, loader: Optional[Any] = None) -> Any:
    """The snapshot resolved against its spec. Skips when the snapshot is not
    checked out; a present snapshot that does not resolve is a failure, never
    a skip."""
    config = snapshot_config(name)
    # The model id lets `match` read the snapshot's header (aux-stack and
    # storage tie-breaks between same-type specs).
    model_id = snapshot_path(name).stem.replace("__", "/", 1)
    resolved = (loader or shared_loader()).match(config, model_id=model_id)
    if resolved is None:
        raise AssertionError(f"{name}: snapshot present but does not resolve to a spec")
    return resolved

def shared_program(name: str) -> Any:
    """`Program.of` this snapshot's resolution, shared for the whole process
    (`CorpusModel.program`).

    The resolution and the graph expansion behind it (~140 ms on a large
    model) are then paid once a process instead of once a test; measured
    over the 26 converted files, resolutions fell 590 -> 220 and expansions
    751 -> 350. The same `Program` object for every caller, which is also
    what lets `shared_walk` recognise two deployments of it as the same
    walk.

    Skips when the snapshot is not checked out; a snapshot that is present
    but does not resolve raises, as `resolve_snapshot` does. Build your own
    resolution (`resolve_snapshot` with your own loader) if you are going to
    WRITE to what you are handed."""
    return _model(name).program()

def own_walk(deployment: Any, *, why: str) -> Any:
    """A walk this test keeps to ITSELF, and the reason it has to.

    `shared_walk` hands the same `Analysis` to every test in the module, and
    an `Analysis` is not frozen. A consumer that writes to the one it is
    given has to have its own, and the two known reasons are:

      * it calls `Analysis.assumed()`, which writes the assumption ledger
        back to the analysis when the scope closes
        (`analysis.py`: `self.assumptions = tuple(ledger.held)`) -- directly,
        or through `inference.serving` / `inference.step_eval` /
        `inference.activation_peak`, which each open that scope;
      * it assigns a layer on the analysis to build a deliberately broken
        graph (`test_layout_materializer` overwrites `input_references`).

    `why` is not read at runtime. It exists so the reason sits at the call
    site, where the next person to touch the test will see it, instead of
    in a list of filenames somewhere else; `tests/meta/
    test_shared_corpus_layer.py` is what makes a bare `analyze()` in a
    sharing file come here."""
    return deployment.analyze()

def _model(name: str) -> CorpusModel:
    """The `CorpusModel` for a model id / stem / filename, skipping the
    calling test when the snapshot is not checked out."""
    path = snapshot_path(name)
    if not path.exists():
        raise unittest.SkipTest(f"{name}: snapshot not present")
    return CorpusModel(path.stem)

def any_snapshot() -> None:
    """Skip when no snapshot at all is checked out (a test that walks the
    whole corpus has nothing to run on); otherwise return."""
    if not any(SNAPSHOT_DIR.glob("*.json")):
        raise unittest.SkipTest("no snapshots checked out")

def backbone_call(resolved: Any, component: str) -> Any:
    """The call a test evaluates `component` at alone: its first call on the
    model's backbone (`CallSite.calls`). A model may call a component in
    several places, and isolation is made at a named one."""
    from backend.arch.eval.transforms import CallSite
    return next(site for site in CallSite.calls(resolved)
                if site.component == component and site.stack == "")


def bench_call(resolved: Any, component: str, loader: Optional[Any] = None) -> Any:
    """The call a test benches `component` at in `resolved`, as a Playground
    entry names one: among those the entry could name
    (`module_bench.component_calls`) — its own, or, for a component the
    model does not call, those of a component in its role, whose place it
    takes — the first on the backbone, or the first when it makes none
    there (an encoder's, made in the encode phase)."""
    from backend.arch.eval.transforms import CallSite
    from backend.playground import module_bench as mb
    calls = mb.component_calls(resolved, component,
                               loader or mb.get_default_loader())
    first = next((c for c in calls if c["label"].startswith("backbone")),
                 calls[0])
    return CallSite.of(resolved, component, call=first["call"])
