"""HF model metadata fetchers used to build prompt ground-truth sections.

Three input sources to the spec-gen prompt come from here:
  • `model_info` subprocess — runs the local venv's transformers to
    `print(model)` and capture forward source. Fails when checkpoint
    requires a transformers version we don't have.
  • `config.json` — always fetchable; primary evidence for dim bindings.
  • `safetensors` metadata — `fetch_metadata_raw` reads the checkpoint
    header under the scheme the config names (the snapshot writer's entry),
    then `format_header` lists it one group per line, each numeric segment's
    index set written in braces.

When `model_info` fails, we fall back to raw modeling source from the
HF repo (`describe_forward_source`) so the agent still has class
`forward()` bodies to read structure off.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import replace
from typing import Optional

from backend.arch.quant_schemes import is_quantization_metadata, scheme_named
from backend.hf.metadata import HUB_UNREACHABLE, format_header
from backend.hf.snapshot import fetch_metadata_raw

from .limits import MODEL_INFO_SEC
from .runtime import _PROJECT_ROOT


logger = logging.getLogger(__name__)


def get_python_transformers_version(python: str) -> Optional[str]:
    """Get the transformers version installed in the given Python env."""
    try:
        result = subprocess.run(
            [python, "-c", "import transformers; print(transformers.__version__)"],
            cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("transformers version probe failed: %s", exc)
        return None
    return None


def summarize_model_info_attempts(attempts: list[dict]) -> str:
    """Summarize the most useful model_info failure message."""
    for attempt in attempts:
        stderr = (attempt.get("stderr") or "").strip()
        if stderr:
            lines = [line.strip() for line in stderr.splitlines() if line.strip()]
            return lines[-1]
    return "model_info subprocess failed without stderr"


def get_model_info(model_id: str) -> dict:
    """Run model_info subprocess. Returns parsed JSON plus status metadata."""
    # Reuse the interpreter running the backend — its venv already has the
    # project's transformers. Hardcoding `.venv/bin/python` breaks any layout
    # where the venv lives elsewhere (e.g. /opt/venv in the container image).
    python = sys.executable
    current_tf = get_python_transformers_version(python)
    attempts = []
    # model_info loads the model with trust_remote_code=True, which EXECUTES
    # the (user-chosen, possibly malicious) HF repo's modeling code. Run it in
    # a scrubbed env (no server secrets to steal) and, in the container, as the
    # unprivileged HA_AGENT_USER — the same hardening as the BYOK agent. In dev
    # (HA_AGENT_USER unset) `user` is omitted, so behavior is unchanged.
    from backend.agents.base import build_scrubbed_env, untrusted_run_user
    run_kwargs: dict = {"env": build_scrubbed_env()}
    _u = untrusted_run_user()
    if _u:
        run_kwargs["user"] = _u
    try:
        result = subprocess.run(
            [python, "-m", "backend.hf.model_info", model_id],
            cwd=_PROJECT_ROOT, capture_output=True, text=True,
            timeout=MODEL_INFO_SEC, check=False,
            **run_kwargs,
        )
        attempts.append({
            "args": [],
            "returncode": result.returncode,
            "stderr": (result.stderr or "").strip(),
        })
        if result.returncode == 0:
            data = json.loads(result.stdout)
            data["_model_info_status"] = {
                "success": True,
                "transformers_version": current_tf,
                "attempts": attempts,
            }
            return data
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        attempts.append({
            "args": [],
            "returncode": None,
            "stderr": str(e),
        })
    err = summarize_model_info_attempts(attempts)
    logger.warning(f"model_info failed for {model_id}: {err}")
    return {
        "_model_info_status": {
            "success": False,
            "transformers_version": current_tf,
            "attempts": attempts,
            "error": err,
        }
    }


def get_tensor_section(model_id: str, config: dict,
                       meta_out: Optional[dict] = None) -> str:
    """Fetch safetensors metadata and format as prompt section.
    `meta_out`, when provided, receives the fetched metadata so callers
    (snapshot path) can persist it without re-fetching.

    Reads the hub under the scheme `config` names, as the snapshot writer
    does: the serving `fetch_metadata` reads an unsnapshotted model under
    `SCHEME_NONE`, and a record captured that way would count a new
    quantized model's scale sidecars as weights."""
    hf_meta = fetch_metadata_raw(model_id, config)
    if meta_out is not None:
        meta_out["metadata"] = hf_meta
    header = hf_meta.get("header")
    if not header:
        return ""
    # The sidecars are the resolved scheme's (its `also_stores` layouts
    # included) — the split `summarize` already made for the totals below.
    scheme = scheme_named(hf_meta["quant_scheme"])
    sidecars, weights = [], []
    for group in header:
        (sidecars if is_quantization_metadata(group.template, scheme)
         else weights).append(group)
    tensor_list = format_header(replace(header, groups=tuple(weights)))
    section = f"""
Your spec MUST account for all tensors below.
Each line is one group of tensors: the index set at each numeric segment is
written in braces (`model.layers.{{0-60}}.mlp.gate.weight` is 61 tensors). Your
spec names a tensor with `X` at each position its declarations index and the
literal segment everywhere else; that name stands for every tensor that agrees
with it at all other segments.

```
{tensor_list}
```
"""
    if sidecars:
        suffixes = ", ".join(f"`*{s.suffix}`" for s in scheme.metadata_siblings)
        section += f"""
Quantization metadata of the checkpoint's scheme `{scheme.name}` (its sidecar
suffixes: {suffixes}). Do NOT declare these: they are not params, and
`quant_context` implies them from the weights they sit beside. Their shapes
show the scheme's block / group size.

```
{format_header(replace(header, groups=tuple(sidecars)))}
```
"""
    unknown = hf_meta.get("quant_method_unknown")
    if unknown:
        section += f"""
The config's `quant_method: {unknown}` names no registered quant scheme, so
no tensor above is recognised as quantization metadata: tell its sidecars
apart in `quant_context` by the scheme their layout follows.
"""
    return section + f"""
Total params from safetensors: {hf_meta.get('total_params', 0):,}
"""


def extract_model_card_highlights(model_id: str) -> str:
    """Extract concise architecture-relevant highlights from the HF model card."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(path, encoding="utf-8") as _f:
            text = _f.read()
    except (OSError, *HUB_UNREACHABLE) as exc:
        logger.debug("model card download failed for %s: %s", model_id, exc)
        return ""

    lines = []
    keywords = (
        "architecture", "attention", "sparse", "index", "indexer",
        "dsa", "mla", "moe", "mtp", "inference", "repo", "vllm", "sglang",
    )
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if any(k in low for k in keywords):
            lines.append(line)
        if len(lines) >= 20:
            break
    return "\n".join(lines)


def fetch_raw_modeling_sources(model_id: str, config: dict) -> str:
    """Pull raw forward source from the HF repo when transformers can't
    build the model locally. Returns a markdown ground-truth section (or "").

    Single source of truth: defers to
    `backend.hf.forward_source.describe_forward_source`, which already knows
    about all the layouts (root `modeling_*.py`, `auto_map` references,
    `inference/model.py` for DSv3.2 / Kimi-K2.5, snapshot,
    model_info_cache, …). One discovery fix → both the runtime Forward
    Source tab AND this prompt section pick it up.
    """
    from backend.hf.forward_source import describe_forward_source

    class _Shim:
        def __init__(self, cfg): self.config = cfg
    info = describe_forward_source(model_id, _Shim(config or {}))
    files = info.get("files") or []
    if not files:
        return ""

    # Cap total text injected — prompts have real limits.
    MAX_BYTES_PER_FILE = 60_000
    MAX_FILES = 3
    parts = []
    for f in files[:MAX_FILES]:
        path = f.get("path") or "(unknown)"
        body = f.get("text") or ""
        # Show repo-relative tail rather than the full HF cache path.
        display = path
        for marker in ("/snapshots/", "/blobs/"):
            if marker in display:
                display = display.split(marker, 1)[1].split("/", 1)[1] if marker == "/snapshots/" else display
                break
        body = _fit_file_to_budget(body, MAX_BYTES_PER_FILE)
        parts.append(f"### `{display}`\n```python\n{body}\n```")

    if not parts:
        return ""

    src_label = info.get("source") or "?"
    return (
        f"## Ground Truth 2: Raw Modeling Source (via `{src_label}` strategy)\n\n"
        "**THIS IS THE GOLD STANDARD FOR BLOCK WIRING.** Your block "
        "graph's node `in:` connections, norm placements, and residual "
        "adds MUST be derived from the `forward()` methods in these "
        "files — not from tensor-name pattern matching alone. If a "
        "class's `forward()` shows N parallel branches, M shared norms, "
        "or an unusual residual shape, the spec must reflect it. "
        "Parameter equality is necessary but NOT sufficient.\n\n"
        + "\n\n".join(parts)
    )


def _fit_file_to_budget(body: str, max_bytes: int) -> str:
    """Smart truncation for the raw-modeling-source section: when a file
    exceeds the prompt budget, keep every `forward()` body whole and cut
    `__init__` / helper boilerplate instead of slicing mid-method.

    The prompt consumes forward WIRING (norm placement, residual adds,
    branch structure) — parameter declarations are already in the
    checkpoint metadata above this section, so a byte-cut through a
    forward() loses the one thing that cannot be recovered while keeping
    the one thing that can. Falls back to the byte cut when the file has
    no forward to prioritize or does not parse.
    """
    if len(body) <= max_bytes:
        return body
    try:
        import ast
        tree = ast.parse(body)
        lines = body.splitlines(keepends=True)
        forwards = [n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == "forward"]
        if forwards:
            head = "".join(lines[:forwards[0].lineno - 1])[: max_bytes // 3]
            parts, budget = [head], max_bytes - len(head) - 200
            kept = 0
            for n in sorted(forwards, key=lambda n: n.lineno):
                seg = "".join(lines[n.lineno - 1:n.end_lineno])
                if len(seg) > budget:
                    break
                parts.append(seg)
                budget -= len(seg)
                kept += 1
            if kept:
                return ("\n\n".join(parts)
                        + f"\n\n# ...[smart-truncated: kept {kept}/{len(forwards)} "
                          f"forward() bodies whole; __init__ and helpers cut — "
                          f"fetch the file directly if needed]...")
    except (SyntaxError, ValueError) as exc:
        logger.debug("smart-truncate of modeling source failed: %s", exc)
    return (body[:max_bytes]
            + f"\n\n# ...[truncated {len(body) - max_bytes} bytes — "
              f"fetch file directly if needed]...")


def build_model_info_failure_section(model_id: str, config: dict, model_info: dict) -> str:
    """Build a visible ground-truth section when transformers structure
    extraction fails."""
    status = model_info.get("_model_info_status", {})
    current_tf = status.get("transformers_version") or "unknown"
    target_tf = config.get("transformers_version") or "unknown"
    model_type = config.get("model_type") or "unknown"
    architectures = config.get("architectures") or []
    error = status.get("error") or "unknown error"

    parts = [
        "## Ground Truth: Structure Extraction Status",
        "",
        "WARNING: The local transformers environment could not construct this model skeleton.",
        f"- Current local transformers: `{current_tf}`",
        f"- Checkpoint requested transformers: `{target_tf}`",
        f"- `config.model_type`: `{model_type}`",
        f"- `config.architectures`: `{architectures}`",
        f"- Failure: `{error}`",
        "",
        "Because `print(model)` and `forward` methods are unavailable in this run, treat new config fields, tensor namespaces, and official implementation repos as primary evidence for unknown structures.",
    ]

    highlights = extract_model_card_highlights(model_id)
    if highlights:
        parts.extend([
            "",
            "## Ground Truth: Hugging Face Model Card Highlights",
            "",
            "```text",
            highlights,
            "```",
        ])

    return "\n".join(parts)


def build_config_section(config: dict, section_num: int) -> str:
    """Build a ground-truth section for config.json."""
    if not config:
        return ""
    cfg_text = json.dumps(config, indent=2, ensure_ascii=False)
    return f"""## Ground Truth {section_num}: Config JSON (from config.json)

Use config fields as first-class evidence for dimensions, layer policies, and newly introduced architectural signals.

```json
{cfg_text}
```"""
