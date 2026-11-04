"""Where the forward-source strategies find a model's modeling code.

Native architectures resolve through transformers' own model_type → module
mapping (text sub-configs like `gemma3_text` live in `models/gemma3`).
Remote code named `org/repo--module.Class` in auto_map lives in that other
repo. A repo is listed only when something is still left to find. A snapshot
refresh whose source fetch fails reports one FAIL line instead of writing a
sourceless snapshot. No network: the Hub client is monkeypatched.
"""
from __future__ import annotations

import httpx2  # the hub's HTTP client: what its calls raise
import pytest
from huggingface_hub import errors as hf_errors

from backend.hf import forward_source as fsrc


class _Cfg:
    def __init__(self, config):
        self.config = config


def _not_found():
    resp = httpx2.Response(404, request=httpx2.Request("GET", "https://hf.co"))
    return hf_errors.RemoteEntryNotFoundError("404", response=resp)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Stand-in Hub: `repos` maps repo id → {filename: text}; a repo mapped
    to an exception raises it on listing. Records every call."""
    state = {"repos": {}, "calls": []}

    def listing(*, repo_id, **_):
        state["calls"].append(("list", repo_id))
        files = state["repos"].get(repo_id, {})
        if isinstance(files, BaseException):
            raise files
        return list(files)

    def download(*, repo_id, filename, **_):
        state["calls"].append(("get", repo_id, filename))
        files = state["repos"].get(repo_id, {})
        if filename not in files:
            raise _not_found()
        path = tmp_path / repo_id.replace("/", "__") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[filename])
        return str(path)

    monkeypatch.setattr("huggingface_hub.list_repo_files", listing)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return state


def _names(files):
    return [f["path"].rsplit("/", 1)[-1] for f in files]


@pytest.mark.parametrize("model_type, module", [
    ("gemma3_text", "modeling_gemma3.py"),   # google/gemma-3-1b-it
    ("llama4_text", "modeling_llama4.py"),
    ("openai-gpt", "modeling_openai.py"),
])
def test_text_subconfigs_resolve_to_the_installed_module(model_type, module):
    files = fsrc._try_local_transformers("org/m", _Cfg({"model_type": model_type}))
    assert _names(files) == [module]


def test_a_cross_repo_auto_map_is_fetched_from_the_referenced_repo(hub):
    hub["repos"] = {"quant/Model-AWQ": {"config.json": "{}"},
                    "upstream/Base": {"modeling_basey.py": "class BaseyForCausalLM: ..."}}
    cfg = {"auto_map": {"AutoModelForCausalLM":
                        "upstream/Base--modeling_basey.BaseyForCausalLM"}}
    files = fsrc._try_hf_modeling_files("quant/Model-AWQ", _Cfg(cfg))
    assert _names(files) == ["modeling_basey.py"]
    assert ("get", "upstream/Base", "modeling_basey.py") in hub["calls"]


def test_missing_auto_map_files_fall_back_to_the_repo_scan(hub):
    hub["repos"] = {"org/m": {"modeling_m.py": "class MForCausalLM: ..."}}
    cfg = {"auto_map": {"AutoModel": "modeling_gone.MModel"}}
    assert _names(fsrc._try_hf_modeling_files("org/m", _Cfg(cfg))) == ["modeling_m.py"]


def test_a_complete_auto_map_fetch_does_not_list_the_repo(hub):
    hub["repos"] = {"org/m": {"modeling_m.py": "class MForCausalLM: ..."}}
    cfg = {"auto_map": {"AutoModel": "modeling_m.MModel"}}
    assert _names(fsrc._try_hf_modeling_files("org/m", _Cfg(cfg))) == ["modeling_m.py"]
    assert not [c for c in hub["calls"] if c[0] == "list"]


def test_a_sibling_decoder_is_followed_in_the_repo_it_was_imported_from(hub):
    hub["repos"] = {
        "org/m": {},
        "upstream/Base": {
            "modeling_wrap.py": "from .modeling_dec import DecForCausalLM\n",
            "modeling_dec.py": "class DecForCausalLM: ...",
        },
    }
    cfg = {"auto_map": {"AutoModel": "upstream/Base--modeling_wrap.Wrap"}}
    files = fsrc._try_hf_modeling_files("org/m", _Cfg(cfg))
    assert _names(files) == ["modeling_wrap.py", "modeling_dec.py"]


def test_a_reference_impl_follows_modules_imported_from_its_own_directory(hub):
    """DeepSeek-V4.1's `inference/model.py` runs as a script and imports its
    Engram and vision blocks absolutely from `inference/engram.py` and
    `inference/vision.py`; both are staged, resolved against the importer's
    directory. Installed / stdlib modules and files elsewhere are not."""
    hub["repos"] = {"org/m": {
        "config.json": "{}",
        "inference/model.py": (
            "import math\n"
            "import torch\n"
            "from engram import EngramLayout, NgramHashState\n"
            "from kernel import act_quant\n"
            "from vision import Aligner, ViT\n"
            "import convert\n"),
        "inference/kernel.py": "def act_quant(): ...\n",
        "inference/engram.py": "class NgramHashState: ...\n",
        "inference/vision.py": "from functools import lru_cache\nclass ViT: ...\n",
        "inference/convert.py": "def main(): ...\n",
        "vision.py": "WRONG = 'root-level namesake'\n",
    }}
    files = fsrc._try_hf_modeling_files("org/m", _Cfg({}))
    got = sorted(f["path"].split("org__m/", 1)[1] for f in files)
    assert got == ["inference/convert.py", "inference/engram.py",
                   "inference/kernel.py", "inference/model.py",
                   "inference/vision.py"]
    asked = {c[2] for c in hub["calls"] if c[0] == "get"}
    assert not any(name.startswith(("inference/math", "inference/torch",
                                    "inference/functools")) for name in asked)


def test_a_relative_decoder_import_resolves_in_the_importers_directory(hub):
    hub["repos"] = {"org/m": {
        "inference/model.py": "from .modeling_dec import DecModel\n",
        "inference/modeling_dec.py": "class DecModel: ...\n",
    }}
    files = fsrc._try_hf_modeling_files("org/m", _Cfg({}))
    assert _names(files) == ["model.py", "modeling_dec.py"]


def test_a_failed_source_fetch_is_one_fail_line_in_the_snapshot_refresh(monkeypatch):
    from backend.hf import snapshot
    monkeypatch.setattr(snapshot, "fetch_config_raw", lambda m: {"model_type": "x"})
    monkeypatch.setattr(snapshot, "fetch_metadata_raw", lambda m, c: {"t": 1})

    def offline(*_):
        raise httpx2.ConnectError("offline")
    monkeypatch.setattr(snapshot, "_fetch_forward_source_raw", offline)
    line = snapshot.snapshot_one("org/never-written")
    assert line.startswith("FAIL forward_source: org/never-written")


def test_a_model_with_no_modeling_code_anywhere_is_not_snapshotted(monkeypatch):
    """No strategy found code (the repo ships none and the installed
    transformers has no module for the model_type): a FAIL line, and no
    snapshot written — not a sourceless one every source check would
    then pass by having nothing to read."""
    from backend.hf import snapshot
    monkeypatch.setattr(snapshot, "fetch_config_raw",
                        lambda m: {"model_type": "brand_new"})
    monkeypatch.setattr(snapshot, "fetch_metadata_raw", lambda m, c: {"t": 1})
    monkeypatch.setattr(snapshot, "_fetch_forward_source_raw", lambda m, c: {})
    written = []
    monkeypatch.setattr("backend.arch.snapshots.write_snapshot",
                        lambda *a, **k: written.append(a))
    line = snapshot.snapshot_one("org/never-written")
    assert line.startswith("FAIL forward_source: org/never-written")
    assert "'brand_new'" in line
    assert written == []
