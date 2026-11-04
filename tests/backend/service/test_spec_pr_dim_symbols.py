"""Open PR promotes the record's `dim_symbols` into main's dims.yaml."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import yaml

from backend.service import github
from backend.spec_worker.bundle import FormalSource, SpecBundleError, is_repairable

REGISTRY = (Path(__file__).resolve().parents[3] / "backend" / "arch"
            / "dims.yaml").read_text()
SPEC = """\
match: {model_id: [org/Model-X]}
dim_symbols:
  n_engram_rows:
    kind: vocab
    shardable_by: [attention]
    group: per_arch_one_offs
    meaning: Rows of one n-gram hash table.
dims: {d_model: config.hidden_size}
"""


def test_the_pr_carries_the_promoted_rows_and_a_spec_without_them():
    files = github._bundle_files("org/Model-X", SPEC, [], None, REGISTRY)
    spec = yaml.safe_load(files["backend/arch/models/org__Model-X.yaml"])
    assert "dim_symbols" not in spec
    rows = yaml.safe_load(files["backend/arch/dims.yaml"])["symbols"]
    assert rows["n_engram_rows"]["used_by"] == "Model-X"


def test_a_conflict_with_main_is_a_repairable_bundle_refusal():
    taken = REGISTRY.replace("  n_vocab:\n", (
        "  n_engram_rows:\n    kind: whole\n    group: per_arch_one_offs\n"
        "    meaning: Something else.\n  n_vocab:\n"), 1)
    with pytest.raises(SpecBundleError) as caught:
        github._bundle_files("org/Model-X", SPEC, [], None, taken)
    assert is_repairable(caught.value)
    assert "rename this spec's symbol" in str(caught.value)


def test_main_dims_yaml_is_read_with_the_formal_namespace(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        body = b"symbols: {}\n"
        info = tarfile.TarInfo("repo-sha/backend/arch/dims.yaml")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    source = github._formal_source_from_archive(buffer.getvalue())
    assert source.registry == "symbols: {}\n"
    assert FormalSource.from_deployment().registry == REGISTRY
