"""C3: every dispatch graph satisfies the routed-zone rules.

`validate_dispatch_region` checks each component graph (and each graph a
model yaml declares) against the three rules that replaced the route state
machine. A model is checked on the graphs its admission brings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import yaml

from tools.corpus_pins.check import CheckKind, CorpusCheck
from tools.corpus_pins.corpus import CorpusModel


def graph_violations(graphs: Iterable[Tuple[Path, Any, Any]]) -> List[Tuple[str, Any]]:
    from backend.arch.layout.zone import validate_dispatch_region
    out = []
    for path, graph, boundary in graphs:
        found = validate_dispatch_region(graph, boundary)
        if found:
            out.append((path.name, found))
    return out


def bundle_graphs(model: CorpusModel) -> List[Tuple[Path, Any, Any]]:
    from backend.arch.layout.zone import graph_boundary
    out = []
    for path in model.bundle():
        if path.suffix not in (".yaml", ".yml"):
            continue
        doc = yaml.safe_load(path.read_text())
        if isinstance(doc, dict) and isinstance(doc.get("graph"), dict):
            out.append((path, doc["graph"], graph_boundary(doc)))
    return out


@dataclass(frozen=True)
class RouteZoneCheck(CorpusCheck):
    def violations(self, model: CorpusModel) -> List[str]:
        return [f"{name}: {found}" for name, found in graph_violations(bundle_graphs(model))]


ROUTE_ZONE = RouteZoneCheck(
    id="C3",
    title="routed zone has no violations",
    kind=CheckKind.ADMIT,
    owner_tests=(
        "tests/backend/arch/layout/test_layout_axis_use_zone.py::"
        "TestRoutedZoneCorpus::test_every_dispatch_graph_in_corpus_is_zone_clean",
    ),
)
