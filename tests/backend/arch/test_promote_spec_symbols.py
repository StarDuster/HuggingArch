"""Open PR moves a spec's own `dim_symbols` into dims.yaml.

The comparison is against the dims.yaml of the tree the PR is built on: a
new name is appended with the model as `used_by`; a name main already has
with the same row is bound to that row and the model joins its `used_by`; a
name main has with a different row refuses with the fix.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from backend.arch.promote_op_proposal import promote_spec_symbols

REGISTRY = (Path(__file__).resolve().parents[3] / "backend" / "arch"
            / "dims.yaml").read_text()

SPEC = """\
match: {model_id: [org/Model-X]}
# The table is the model's n-gram memory.
dim_symbols:
  n_engram_rows:
    kind: vocab
    shardable_by: [attention]
    group: per_arch_one_offs
    meaning: Rows of one n-gram hash table.
dims:
  d_model: config.hidden_size   # the residual width
"""


def test_a_new_symbol_is_appended_and_the_block_leaves_the_spec():
    out = promote_spec_symbols("org/Model-X", SPEC, REGISTRY)
    rows = yaml.safe_load(out.registry)["symbols"]
    assert rows["n_engram_rows"]["used_by"] == "Model-X"
    assert rows["n_engram_rows"]["shardable_by"] == ["attention"]
    spec = yaml.safe_load(out.spec_yaml)
    assert "dim_symbols" not in spec
    assert "# the residual width" in out.spec_yaml
    # dims.yaml keeps its comments and every row it had.
    assert out.registry.startswith(REGISTRY.split("\n", 1)[0])
    assert set(yaml.safe_load(REGISTRY)["symbols"]) < set(rows)


def test_an_identical_row_joins_the_existing_ones_used_by():
    first = promote_spec_symbols("org/Model-X", SPEC, REGISTRY)
    again = promote_spec_symbols(
        "org/Model-Y", SPEC.replace("Model-X", "Model-Y").replace(
            "[attention]", "[ attention ]"), first.registry)
    row = yaml.safe_load(again.registry)["symbols"]["n_engram_rows"]
    assert row["used_by"] == "Model-X, Model-Y"


def test_a_different_row_refuses_with_the_fix():
    first = promote_spec_symbols("org/Model-X", SPEC, REGISTRY)
    whole = SPEC.replace("    kind: vocab\n    shardable_by: [attention]\n",
                         "    kind: whole\n")
    with pytest.raises(ValueError, match="bind the registered symbol"):
        promote_spec_symbols("org/Model-Y", whole, first.registry)


def test_a_spec_without_symbols_is_untouched():
    plain = "match: {model_id: [org/M]}\ndims: {d_model: config.hidden_size}\n"
    out = promote_spec_symbols("org/M", plain, REGISTRY)
    assert out.spec_yaml == plain and out.registry is None


def test_the_registry_must_have_been_read():
    with pytest.raises(ValueError, match="was not read"):
        promote_spec_symbols("org/Model-X", SPEC, "")
