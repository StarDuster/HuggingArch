"""Gating errors tell the writer to author fields the schema still accepts.

A finding that names a retired key (`kv_label`, `kv_cached`, …) sends the
spec-gen repair round into a loop: the writer copies the field, the loader
refuses it. This test extracts the field names the findings *suggest* and
checks them against the live authoring schema — the contract, not the
wording.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import get_args, get_origin, Union, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.document import SpecDocument
from backend.arch.loader import ResolvedSpec
from backend.arch.ops import OP_REGISTRY
from backend.arch.schema import (
    ArchSpec,
    Axiom,
    AxiomSet,
    BlockGraph,
    CacheEntry,
    CacheSpec,
    GraphNode,
    ModelLayer,
)
from backend.arch.schema.spec import ValidationExemptions
from backend.arch.spec_parsing import (
    RETIRED_KV_REPLACEMENTS,
    _BLOCK_GRAPH_SOURCE_KEYS,
    _GRAPH_NODE_SOURCE_KEYS,
    _RETIRED_CARRY_KEYS,
    _RETIRED_IDENTITY_KEYS,
    _RETIRED_KV_KEYS,
)
from backend.arch.validate._type_contracts import check_field_contracts


_SUGGESTED_FIELD = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)(?::[^`]*)?`")


def _literals(annotation) -> set[str]:
    origin = get_origin(annotation)
    if origin is Literal:
        return {a for a in get_args(annotation) if isinstance(a, str)}
    if origin is Union:
        names: set[str] = set()
        for arg in get_args(annotation):
            names |= _literals(arg)
        return names
    return set()


def _authorable_leaves() -> set[str]:
    """Top-level field names and Literal values a writer may put in YAML."""
    names = set(_BLOCK_GRAPH_SOURCE_KEYS) | set(_GRAPH_NODE_SOURCE_KEYS)
    names.add("in")  # YAML alias for GraphNode.inputs
    names.update(OP_REGISTRY)  # a node's `op:` value
    models = (
        CacheSpec, CacheEntry, Axiom, AxiomSet, ArchSpec, SpecDocument,
        ValidationExemptions, BlockGraph, GraphNode,
    )
    for model in models:
        names.update(model.model_fields)
        for field in model.model_fields.values():
            names.update(_literals(field.annotation))
    return names


def _schema_accepts(name: str) -> bool:
    """True when `name` is a live authorable field, nested path, or Literal.

    Nested paths walk the real models: `cache.*` against CacheSpec,
    `cache.entries.*` against CacheEntry, `validation_exemptions.*`
    against ValidationExemptions. A top-level key being authorable does
    not accept an arbitrary child (`cache.typo` fails).
    """
    parts = name.split(".")
    if len(parts) == 1:
        return name in _authorable_leaves()
    if parts[0] == "cache":
        if len(parts) == 2:
            return parts[1] in CacheSpec.model_fields
        if parts[1] == "entries" and len(parts) == 3:
            return parts[2] in CacheEntry.model_fields
        return False
    if parts[0] == "validation_exemptions":
        return len(parts) == 2 and parts[1] in ValidationExemptions.model_fields
    return False


def _retired_keys() -> set[str]:
    return set(_RETIRED_KV_KEYS) | set(_RETIRED_CARRY_KEYS) | set(_RETIRED_IDENTITY_KEYS)


def _attn(role="attention", family=None, entries=None, **kw):
    cache = (CacheSpec(family=family, entries=entries or [])
             if (family is not None or entries is not None) else None)
    return BlockGraph(
        inputs=["x"],
        graph={"wq": GraphNode(op="linear", inputs=["x"],
                               dims={"in": "d_model", "out": "d_model"})},
        outputs=["wq"],
        role=role,
        cache=cache,
        **kw,
    )


def _make_resolved(components, config=None):
    graph = {}
    prev = "x"
    for i, name in enumerate(components):
        nid = f"c{i}"
        graph[nid] = GraphNode(op=name, inputs=[prev])
        prev = nid
    block = BlockGraph(inputs=["x"], graph=graph, outputs=[prev])
    spec = ArchSpec(
        model_inputs={"input_ids": {"kind": "tokens"}},
        name="demo",
        dims={"d_model": "config.hidden_size"},
        model=[
            ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                       boundary_role="input_embedding",
                       dims={"n_vocab": "d_model", "d_model": "d_model"}),
            ModelLayer(name="layers", template="block", repeat="n_layers",
                       inputs=["embed"]),
        ],
        blocks={"block": block},
        components=dict(components),
    )
    return ResolvedSpec(spec, {"d_model": 16, "n_layers": 1},
                        {"hidden_size": 16, "torch_dtype": "bfloat16", **(config or {})}, {})


def _assert_suggested_fields_are_authorable(text, *, kind):
    suggested = set(_SUGGESTED_FIELD.findall(text))
    leaked = suggested & _retired_keys()
    assert not leaked, (
        f"{kind} suggests retired field(s) {sorted(leaked)}: {text}"
    )
    unknown = {name for name in suggested if not _schema_accepts(name)}
    assert not unknown, (
        f"{kind} suggests {sorted(unknown)}, which the schema "
        f"does not accept: {text}"
    )


def test_retired_kv_replacements_are_authorable():
    assert set(RETIRED_KV_REPLACEMENTS) == set(_RETIRED_KV_KEYS)
    for retired, path in RETIRED_KV_REPLACEMENTS.items():
        assert _schema_accepts(path), (
            f"replacement for {retired!r} is {path!r}, which the schema "
            f"does not accept"
        )


def test_gating_findings_suggest_authorable_fields():
    from backend.arch.eval.cache import MissingCachedNode
    from backend.arch.validate._axioms import _calls, _check_config_windowing

    findings = []
    findings += check_field_contracts(_make_resolved({"a": _attn()}))
    findings += check_field_contracts(
        _make_resolved({"a": _attn(entries=[{"node": "wq", "kind": "kv"}])}))
    findings += check_field_contracts(
        _make_resolved({"a": _attn(family="GQA")}))
    swa = _make_resolved({"swa": _attn(family="SWA", entries=[{"node": "wq", "kind": "kv"}])})
    findings += [f for call in _calls(swa) for f in call.judge()]
    windowed = _make_resolved({"a": _attn(entries=[{"node": "wq", "kind": "kv"}])},
                              config={"sliding_window": 4096})
    findings += _check_config_windowing(windowed, _calls(windowed))

    assert findings, "expected missing-cache / missing-window findings"
    for finding in findings:
        _assert_suggested_fields_are_authorable(
            finding.detail, kind=finding.kind)

    missing = MissingCachedNode(
        "'attn' declares `cache.entries`: ['k'] and has no "
        "such node — the declaration names the graph, so one of the "
        "two is a typo. Present: []"
    )
    _assert_suggested_fields_are_authorable(str(missing), kind="MissingCachedNode")
    assert not _schema_accepts("cache.typo")
