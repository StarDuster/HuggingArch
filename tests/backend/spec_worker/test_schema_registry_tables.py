"""The schema doc quotes every entry of the registries the loader checks.

`templates/schema.md` used to carry hand-copied tables of roles, cache
families, cache slot kinds, quant schemes and custom-op vocabularies. They
drifted from the registries (6 of 20 roles, no `nvfp4`, a cache-family
"not a closed enum" the loader in fact enforces), so the writer was told a
smaller language than the one it is judged against. The tables are now
rendered from the registries; these tests hold the rendered prompt to
every entry.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from backend.arch import cache_families, roles
from backend.arch.ops_base import INPUT_ROLES
from backend.arch.quant_schemes import QUANT_SCHEMES, SCHEME_NONE
from backend.arch.schema.custom_ops import CustomOpSpec
from backend.spec_worker._prompts import system as _sys
from backend.spec_worker._prompts.components import (
    schema_registry_tables,
)


def _doc() -> str:
    _sys._load_template.cache_clear()
    return _sys.schema_doc()


def test_every_placeholder_is_filled():
    doc = _doc()
    for placeholder in schema_registry_tables():
        assert placeholder not in doc


def test_every_role_is_listed_with_its_kind():
    doc = _doc()
    for role in roles.all_roles().values():
        assert f"| `{role.name}` | {role.kind} |" in doc, role.name


def test_every_cache_family_and_slot_kind_is_listed():
    doc = _doc()
    for fam in cache_families.all_families().values():
        assert f"| `{fam.name}` | {fam.kind} |" in doc, fam.name
    for kind in cache_families.slot_kinds():
        assert f"| `{kind}` |" in doc, kind
    assert "not a closed" not in doc


def test_every_quant_scheme_is_listed_with_its_fields():
    doc = _doc()
    rows = {line.split("|")[1].strip(): line for line in doc.splitlines()
            if line.startswith("| `")}
    assert f"`{SCHEME_NONE.name}`" in rows
    for name, scheme in QUANT_SCHEMES.items():
        row = rows[f"`{name}`"]
        for field in (*scheme.required_fields, *scheme.optional_fields,
                      *(k for k, _ in scheme.default_fields)):
            assert f"`{field}" in row, (name, field)
        for sibling in scheme.metadata_siblings:
            assert f"`{sibling.suffix}`" in row, (name, sibling.suffix)


def test_custom_op_vocabularies_are_listed():
    doc = _doc()
    shape_kinds = CustomOpSpec.__private_attributes__["_YAML_SHAPE_KINDS"].default
    line = next(ln for ln in doc.splitlines()
                if "`shape_kind` is how the output shape derives" in ln)
    for kind in shape_kinds:
        assert f"`{kind}`" in line, kind
    vocab = next(ln for ln in doc.splitlines() if "closed vocabulary" in ln
                 and "`q`" in ln)
    for role in INPUT_ROLES:
        assert f"`{role.key}`" in vocab, role.key


def test_a_lost_placeholder_is_an_error(monkeypatch):
    _sys._load_template.cache_clear()
    monkeypatch.setattr(_sys, "_load_template", lambda name: "no tables here")
    try:
        _sys.schema_doc()
    except ValueError as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("schema_doc accepted a template with no placeholders")
