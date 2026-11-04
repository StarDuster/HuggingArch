"""A conservative structural identity for component YAML documents.

The signature deliberately includes every field that can change evaluation or
reference resolution and excludes only prose (`description`; YAML comments are
not represented after parsing). It is used to decide whether two same-named
components are interchangeable, so a formatting-only change must not look like
a conflict and a changed dim binding must.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from backend.arch.schema.custom_ops import CustomOpSpec
from backend.arch.spec_parsing import _parse_block_graph


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def component_structure_signature(text: str, *, path: str = "<component>") -> str:
    """Return a stable SHA-256 identity for one component document.

    Parsing through ``_parse_block_graph`` rejects unknown fields instead of
    silently dropping them. ``custom_ops`` travels beside the graph and is
    validated with the same Pydantic model the loader uses. Invalid candidates
    therefore raise; callers treat them as repairable data errors, never as
    “structurally different but usable” documents.
    """
    try:
        raw = load_yaml(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: component YAML is not parseable: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: component document must be a YAML mapping")

    # Keep graph-shape failures in the same ValueError channel as the parser's
    # other data errors. Letting `.items()` fail with AttributeError would force
    # every caller to know this parser's internal representation.
    if "graph" in raw and not isinstance(raw["graph"], dict):
        raise ValueError(f"{path}: component graph must be a mapping")

    graph = _parse_block_graph(raw, identity=Path(path).stem)
    custom_ops_raw = raw.get("custom_ops") or {}
    if not isinstance(custom_ops_raw, dict):
        raise ValueError(f"{path}: custom_ops must be a mapping")
    custom_ops = {
        str(name): CustomOpSpec(**spec).model_dump(mode="json")
        for name, spec in custom_ops_raw.items()
    }
    payload: Dict[str, Any] = {
        "component": graph.model_dump(mode="json", exclude={"description"}),
        "custom_ops": custom_ops,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()
