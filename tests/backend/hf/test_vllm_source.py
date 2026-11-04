"""vLLM's model files as reference source for the spec agents: which files a
checkpoint's architectures name (its module, a package's CUDA files, the MTP
draft vLLM's speculative config assigns it), staged beside the modeling
files and citable by their path. Runs on an in-memory vLLM tree; no network."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.hf.vllm_source import (
    REGISTRY_PATH, SPECULATIVE_PATH, VllmModelSource, VllmTree, mtp_architectures,
    registry_modules,
)
from backend.spec_worker._prompts.system import source_files_section
from backend.spec_worker.overlay import stage_vllm_source
from backend.spec_worker.review_evidence import SnapshotFacts

REGISTRY = '''
_TEXT_GENERATION_MODELS = {
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "HYV3ForCausalLM": ("hy_v3", "HYV3ForCausalLM"),
    "HYV4ForCausalLM": ("vllm.models.hy_v4", "HYV4ForCausalLM"),
    "MiMoV2ForCausalLM": ("mimo_v2", "MiMoV2ForCausalLM"),
}
_SPECULATIVE_DECODING_MODELS = {
    "DeepSeekMTPModel": ("deepseek_mtp", "DeepSeekMTP"),
    "HYV3MTPModel": ("hy_v3_mtp", "HYV3MTP"),
    "HYV4MTPModel": ("vllm.models.hy_v4", "HYV4MTP"),
    "MiMoV2MTPModel": ("mimo_v2_mtp", "MiMoV2MTP"),
}
'''

SPECULATIVE = '''
class SpeculativeConfig:
    @staticmethod
    def hf_config_override(hf_config):
        if hf_config.model_type in ("deepseek_v3", "deepseek_v32"):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            hf_config.update({"architectures": ["DeepSeekMTPModel"]})
        if hf_config.model_type == "hy_v3":
            hf_config.model_type = "hy_v3_mtp"
            hf_config.update({"architectures": ["HYV3MTPModel"]})
        if hf_config.model_type == "hy_v4":
            hf_config.update({"architectures": ["HYV4MTPModel"]})
        return hf_config
'''

FILES = {
    REGISTRY_PATH: REGISTRY,
    SPECULATIVE_PATH: SPECULATIVE,
    "vllm/model_executor/models/deepseek_v2.py": "class DeepseekV3ForCausalLM: ...\n",
    "vllm/model_executor/models/deepseek_mtp.py": "class DeepSeekMTP: ...\n",
    "vllm/model_executor/models/hy_v3.py": "class HYV3ForCausalLM: ...\n",
    "vllm/model_executor/models/hy_v3_mtp.py": "self.enorm = RMSNorm(config.hidden_size)\n",
    "vllm/model_executor/models/mimo_v2.py": "class MiMoV2ForCausalLM: ...\n",
    "vllm/model_executor/models/mimo_v2_mtp.py": "class MiMoV2MTP: ...\n",
    "vllm/models/hy_v4/__init__.py": "from .nvidia.model import HYV4ForCausalLM\n",
    "vllm/models/hy_v4/nvidia/model.py": (
        "from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP\n"
        "from vllm.model_executor.models.utils import maybe_prefix\n"
        "class HYV4ForCausalLM: ...\n"),
    "vllm/model_executor/models/utils.py": "def maybe_prefix(): ...\n",
    "vllm/models/hy_v4/nvidia/mtp.py": "self.eh_proj = nn.Linear(2 * h, h)\n",
    "vllm/models/hy_v4/amd/model.py": "class HYV4ForCausalLM: ...\n",
}


class _Tree(VllmTree):
    def read(self, path: str) -> str:
        return FILES[path]


TREE = _Tree(commit="c0ffee", paths=frozenset(FILES))


def _paths(config):
    return [f["path"] for f in VllmModelSource.for_config(config, TREE).files]


def test_registry_maps_each_architecture_to_its_module():
    reg = registry_modules(REGISTRY)
    assert reg["HYV3ForCausalLM"] == "hy_v3"
    assert reg["HYV4MTPModel"] == "vllm.models.hy_v4"


def test_the_mtp_draft_follows_the_model_type_rewrites():
    archs = set(registry_modules(REGISTRY))
    assert mtp_architectures(SPECULATIVE, "deepseek_v3", archs) == ["DeepSeekMTPModel"]
    assert mtp_architectures(SPECULATIVE, "hy_v3", archs) == ["HYV3MTPModel"]
    assert mtp_architectures(SPECULATIVE, "qwen3", archs) == []


def test_a_model_brings_its_module_and_its_mtp_draft():
    assert _paths({"architectures": ["HYV3ForCausalLM"], "model_type": "hy_v3"}) == [
        "vllm/model_executor/models/hy_v3.py", "vllm/model_executor/models/hy_v3_mtp.py"]
    assert _paths({"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"}) == [
        "vllm/model_executor/models/deepseek_v2.py", "vllm/model_executor/models/deepseek_mtp.py"]


def test_an_mtp_module_beside_the_models_own_counts_without_an_override():
    assert _paths({"architectures": ["MiMoV2ForCausalLM"], "model_type": "mimo_v2"}) == [
        "vllm/model_executor/models/mimo_v2.py", "vllm/model_executor/models/mimo_v2_mtp.py"]


def test_a_package_brings_its_cuda_files_not_its_rocm_ones():
    source = VllmModelSource.for_config({"architectures": ["HYV4ForCausalLM"],
                                         "model_type": "hy_v4"}, TREE)
    assert [f["path"] for f in source.files] == [
        "vllm/models/hy_v4/__init__.py", "vllm/models/hy_v4/nvidia/model.py",
        "vllm/models/hy_v4/nvidia/mtp.py", "vllm/model_executor/models/deepseek_v2.py"]
    # a registered model module it imports comes along, marked imported; a
    # helper module (utils) does not
    assert source.imported == ("vllm/model_executor/models/deepseek_v2.py",)


def test_an_architecture_vllm_lacks_brings_nothing():
    assert VllmModelSource.for_config({"architectures": ["Unknown"]}, TREE) is None
    assert VllmModelSource.for_config({}, TREE) is None


def test_staged_files_are_cited_by_their_vllm_path(tmp_path):
    source = VllmModelSource.for_config(
        {"architectures": ["HYV3ForCausalLM"], "model_type": "hy_v3"}, TREE)
    staged = stage_vllm_source(tmp_path, source.files)
    assert staged[1] == str((tmp_path / "source/vllm/model_executor/models/hy_v3_mtp.py").resolve())
    facts = SnapshotFacts.from_run({}, staged)
    check = facts.check_source("vllm/model_executor/models/hy_v3_mtp.py", 1, 1,
                               "self.enorm = RMSNorm(config.hidden_size)", "raw")
    assert check.status == "verified", check
    section = source_files_section(staged)
    assert "vLLM's implementation (reference)" in section
    assert "## Full source files" in section


def test_a_path_that_leaves_the_source_dir_is_refused(tmp_path):
    with pytest.raises(ValueError):
        stage_vllm_source(tmp_path, [{"path": "../escape.py", "text": "x"}])
