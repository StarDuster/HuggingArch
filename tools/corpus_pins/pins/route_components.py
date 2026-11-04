"""P6: the component files whose graph dispatches tokens (`permute`).

Pinned as relative paths: a bare count lets an addition and a removal
cancel, and same-named files in different directories must not collapse.
A model owns the routed components its admission brings (its drafts, or a
new library file in its bundle).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tools.corpus_pins.corpus import ARCH_DIR, CorpusModel, relative
from tools.corpus_pins.pin import Doc, Pin, Row, Scope, json_bytes


def routed(paths: Iterable[Path]) -> Row:
    from backend.arch.layout.zone import graph_boundary
    import yaml
    out: Row = {}
    for path in paths:
        doc = yaml.safe_load(Path(path).read_text())
        if not (isinstance(doc, dict) and isinstance(doc.get("graph"), dict)):
            continue
        graph_boundary(doc)
        if any(spec.get("op") == "permute" for spec in doc["graph"].values()):
            out[relative(path)] = True
    return out


@dataclass(frozen=True)
class RouteComponentsPin(Pin):
    def parse(self, data: bytes) -> Doc:
        paths = json.loads(data.decode("utf-8"))
        if paths != sorted(set(paths)):
            raise ValueError(f"{self.artifact}: not a sorted set")
        return {path: True for path in paths}

    def render(self, doc: Doc) -> bytes:
        return json_bytes(sorted(doc))

    def applies(self, model: CorpusModel) -> bool:
        return bool(self.measure(model))

    def key_exists(self, key: str, stems, components) -> bool:
        return key in components

    def measure(self, model: CorpusModel) -> Row:
        return routed(p for p in model.bundle()
                      if p.suffix in (".yaml", ".yml")
                      and relative(p).startswith("components/"))

    def measure_all(self) -> Doc:
        from backend.arch.layout.zone import iter_component_graphs
        return {relative(path): True
                for path, graph, _boundary in iter_component_graphs(ARCH_DIR)
                if any(spec.get("op") == "permute" for spec in graph.values())}


ROUTE_COMPONENTS = RouteComponentsPin(
    id="P6",
    title="routed component files",
    scope=Scope.CONDITIONAL,
    artifact="tests/golden/pins/route_components.json",
    owner_tests=(
        "tests/backend/arch/layout/test_layout_axis_use_zone.py::"
        "TestRoutedZoneCorpus::test_every_dispatch_graph_in_corpus_is_zone_clean",
    ),
)
