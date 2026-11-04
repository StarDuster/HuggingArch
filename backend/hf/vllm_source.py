"""vLLM's implementation of a model, as reference source for the spec agents.

The transformers modeling code is the primary source a spec is written and
reviewed against, but it does not cover everything a served model runs: an
MTP head is often absent from it entirely (transformers drops the layer on
load), and so are weights only a serving framework reads. vLLM implements
those parts, so the spec agents also read vLLM's model files for the
checkpoint's `architectures`.

Which files: vLLM's model registry (`vllm/model_executor/models/registry.py`)
maps an architecture to a module, either one file under
`vllm/model_executor/models/` or a package under `vllm/models/` (every file
of the package is taken). The MTP draft model is the architecture vLLM's
speculative config rewrites the checkpoint's `model_type` to
(`SpeculativeConfig.hf_config_override`), read here from that method's
source, plus the registry's `<module>_mtp` module of the model's own module
where vLLM keeps one (an override that imports its draft class by name
names no architecture); its module is taken the same way. A package's
`amd/` subpackage is left out: the package dispatches to it only on ROCm,
and its CUDA files are the implementation the spec agents read. A model
module that builds on another registered model module (a subclass of its
model, a layer it reuses) brings that module too, one import deep.

The files are read at vLLM main's tip over GitHub (the newest models need the
newest vLLM) and cached per commit; the commit travels with the files so a
run can say which vLLM it read. A fetch that fails yields no files: the vLLM
source is a reference, never a precondition of a run.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx

logger = logging.getLogger(__name__)

REPO = "vllm-project/vllm"
_API = f"https://api.github.com/repos/{REPO}"
_RAW = f"https://raw.githubusercontent.com/{REPO}"
REGISTRY_PATH = "vllm/model_executor/models/registry.py"
SPECULATIVE_PATH = "vllm/config/speculative.py"
_MODELS_DIR = "vllm/model_executor/models"
_CACHE_DIR = Path.home() / ".cache" / "huggingarch" / "vllm_source"
#: how long main's tip is reused before it is asked for again
_TIP_TTL_SEC = 3600


@dataclass(frozen=True)
class VllmTree:
    """vLLM at one commit: its file list, and file text on demand (cached)."""

    commit: str
    paths: frozenset

    @classmethod
    def at_main(cls) -> "VllmTree":
        commit = _main_tip()
        listing = _cache_path(commit, "_tree.json")
        if listing.is_file():
            paths = json.loads(listing.read_text(encoding="utf-8"))
        else:
            r = _get(f"{_API}/git/trees/{commit}", params={"recursive": "1"})
            paths = [e["path"] for e in r.json()["tree"] if e.get("type") == "blob"]
            _write(listing, json.dumps(paths))
        return cls(commit=commit, paths=frozenset(paths))

    def read(self, path: str) -> str:
        cached = _cache_path(self.commit, path)
        if cached.is_file():
            return cached.read_text(encoding="utf-8")
        text = _get(f"{_RAW}/{self.commit}/{path}").text
        _write(cached, text)
        return text

    def module_files(self, module: str) -> List[str]:
        """The repo paths of a registry module: `hy_v3` is
        `vllm/model_executor/models/hy_v3.py`; `vllm.models.hy_v4` is every
        `.py` file of the package `vllm/models/hy_v4/`."""
        if "." not in module:
            path = f"{_MODELS_DIR}/{module}.py"
            return [path] if path in self.paths else []
        package = module.replace(".", "/")
        single = f"{package}.py"
        if single in self.paths:
            return [single]
        return sorted(p for p in self.paths
                      if p.startswith(package + "/") and p.endswith(".py")
                      and "amd" not in p[len(package) + 1:].split("/")[:-1])


@dataclass(frozen=True)
class VllmModelSource:
    """The vLLM files of one model: `files` is `[{path, text}]`, each `path`
    relative to the vLLM repo root (`vllm/models/hy_v4/nvidia/mtp.py`);
    `imported` names the ones only brought in by an import of another model
    module, the rest are the model's own."""

    commit: str
    architectures: tuple
    files: tuple
    imported: tuple = ()

    @classmethod
    def for_config(cls, config: dict, tree: Optional[VllmTree] = None) -> Optional["VllmModelSource"]:
        """vLLM's files for the config's architectures and their MTP draft
        model, or None when vLLM has none of them or cannot be read."""
        archs = [a for a in (config.get("architectures") or []) if isinstance(a, str)]
        if not archs:
            return None
        try:
            tree = tree or VllmTree.at_main()
            registry = registry_modules(tree.read(REGISTRY_PATH))
            drafts = mtp_architectures(tree.read(SPECULATIVE_PATH),
                                       str(config.get("model_type") or ""), set(registry))
            own = {registry[a] for a in archs if a in registry}
            siblings = [a for a, m in registry.items() if m in {f"{o}_mtp" for o in own}]
            wanted = list(dict.fromkeys(a for a in [*archs, *drafts, *siblings] if a in registry))
            paths: List[str] = []
            for arch in wanted:
                paths += [p for p in tree.module_files(registry[arch]) if p not in paths]
            if not paths:
                return None
            own = list(paths)
            model_files = {f for m in set(registry.values()) for f in tree.module_files(m)}
            for path in own:
                for module in _imported_modules(tree.read(path)):
                    paths += [f for f in tree.module_files(module)
                              if f in model_files and f not in paths]
            files = tuple({"path": p, "text": tree.read(p)} for p in paths)
        except (httpx.HTTPError, OSError, ValueError, KeyError, SyntaxError) as exc:
            logger.warning("vLLM source for %s unavailable: %s", archs, exc)
            return None
        return cls(commit=tree.commit, architectures=tuple(wanted), files=files,
                   imported=tuple(p for p in paths if p not in own))


def _imported_modules(source: str) -> List[str]:
    """The vLLM model modules a file imports from (`vllm.model_executor.models.x`,
    `vllm.models.x...`), as registry-style module names."""
    out = []
    prefix = "vllm.model_executor.models."
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(prefix):
                out.append(node.module[len(prefix):])
            elif node.module.startswith("vllm.models."):
                out.append(node.module)
    return out


def registry_modules(registry_source: str) -> Dict[str, str]:
    """`{architecture: module}` from vLLM's registry module: every dict
    literal whose entries are `"Arch": ("module", "Class")`."""
    out: Dict[str, str] = {}
    for node in ast.walk(ast.parse(registry_source)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and isinstance(value, ast.Tuple) and value.elts
                    and isinstance(value.elts[0], ast.Constant)
                    and isinstance(value.elts[0].value, str)):
                out.setdefault(key.value, value.elts[0].value)
    return out


def mtp_architectures(speculative_source: str, model_type: str,
                      architectures: Set[str]) -> List[str]:
    """The draft architectures vLLM's `hf_config_override` assigns a
    checkpoint of `model_type`. The method is a sequence of
    `if hf_config.model_type == / in …:` blocks, each of which may rewrite
    `model_type` for the blocks after it (`deepseek_v3` → `deepseek_mtp` →
    `DeepSeekMTPModel`); a block that applies contributes every
    architecture name it writes."""
    method = next((n for n in ast.walk(ast.parse(speculative_source))
                   if isinstance(n, ast.FunctionDef) and n.name == "hf_config_override"),
                  None)
    if method is None or not model_type:
        return []
    types = {model_type}
    found: List[str] = []
    for stmt in method.body:
        if not isinstance(stmt, ast.If) or not (_tested_types(stmt.test) & types):
            continue
        for node in ast.walk(ast.Module(body=stmt.body, type_ignores=[])):
            if isinstance(node, ast.Assign) and _is_model_type(node.targets[0]) \
                    and isinstance(node.value, ast.Constant):
                types.add(str(node.value.value))
            if (isinstance(node, ast.Constant) and node.value in architectures
                    and node.value not in found):
                found.append(node.value)
    return found


def _is_model_type(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "model_type"


def _tested_types(test: ast.AST) -> Set[str]:
    """The model types an `if` compares `….model_type` against."""
    out: Set[str] = set()
    for node in ast.walk(test):
        if isinstance(node, ast.Compare) and _is_model_type(node.left):
            for comp in node.comparators:
                for c in ast.walk(comp):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        out.add(c.value)
    return out


def _main_tip() -> str:
    marker = _CACHE_DIR / "_main.json"
    if marker.is_file():
        cached = json.loads(marker.read_text(encoding="utf-8"))
        if time.time() - cached.get("at", 0) < _TIP_TTL_SEC:
            return cached["commit"]
    commit = _get(f"{_API}/commits/main").json()["sha"]
    _write(marker, json.dumps({"commit": commit, "at": time.time()}))
    return commit


def _get(url: str, params: Optional[dict] = None) -> httpx.Response:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith(_API):
        headers["Authorization"] = f"Bearer {token}"
    r = httpx.get(url, params=params, headers=headers, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    return r


def _cache_path(commit: str, path: str) -> Path:
    return _CACHE_DIR / commit / path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

