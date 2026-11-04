"""Ground-truth assembly: every authoritative source the agent should
read before authoring a spec, formatted as numbered prompt sections.

Pipeline:
  1. transformers `print(model)` (when local env can build the skeleton).
  2. `forward()` method bodies of every named submodule.
  3. Raw `modeling_*.py` from HF (fallback when (1) fails).
  4. `config.json` (always available).
  5. Tensor patterns from safetensors (always available — also
     populates the snapshot capture so the post-success writer doesn't
     re-probe HF).

The output of `get_ground_truth(model_id)` is `(ground_truth_str, meta)`
where `meta._snapshot_data` carries the inputs needed to write a
post-success spec snapshot without re-fetching anything.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Tuple

from backend.hf.metadata import HUB_UNREACHABLE, metadata_record
from backend.hf.snapshot import fetch_config_raw

from ._metadata import (
    get_model_info,
    get_tensor_section,
    fetch_raw_modeling_sources,
    build_model_info_failure_section,
    build_config_section,
)

logger = logging.getLogger(__name__)


def capture_forward_source_for_snapshot(model_id: str, config: dict) -> dict:
    """Best-effort describe_forward_source bypassing the snapshot itself —
    so refreshes pick up the real upstream source, not a stale snapshot."""
    from backend.hf.forward_source import (
        _try_custom_model, _try_local_transformers,
        _try_model_info_cache, _try_hf_modeling_files,
    )

    class _ResolvedShim:
        def __init__(self, cfg): self.config = cfg
    shim = _ResolvedShim(config or {})

    for label, fn in (
        ("custom_model", lambda: _try_custom_model(model_id)),
        ("local_transformers", lambda: _try_local_transformers(model_id, shim)),
        ("model_info_cache", lambda: _try_model_info_cache(model_id)),
        ("hf_modeling_files", lambda: _try_hf_modeling_files(model_id, shim)),
    ):
        # Best-effort capture for the snapshot: a failed strategy (the HF one
        # raises on transport failure) falls through to the next.
        # `full_module_sources` re-fetches and reports the typed failure.
        try:
            files = fn()
        except Exception as exc:
            logger.info("forward source strategy %s failed for %s: %s",
                        label, model_id, exc)
            continue
        if files and any(f.get("text") for f in files):
            return {
                "source": label,
                "paths": [f["path"] for f in files if f.get("path")],
                "files": files,
            }
    return {}


class ForwardSourceUnavailable(RuntimeError):
    """Prep could not obtain the model's real modeling source, so no spec-gen
    agent (generate / refine / review / fix) may start: a spec written or
    judged from forward excerpts alone may compute a different function.

    `retryable` separates a fetch that failed (network, Hub outage) from a
    model whose source cannot be read anywhere; the message says which and
    what to do. The job reports it as its terminal `done` event."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable

    @property
    def error_kind(self) -> str:
        return ("forward_source_unreachable" if self.retryable
                else "forward_source_missing")

    def done_event(self) -> dict:
        return {"type": "done", "success": False, "message": str(self),
                "error_kind": self.error_kind, "retryable": self.retryable}


def full_module_sources(model_id: str, config: dict, forward_source: dict | None) -> list:
    """Complete modeling files (`[{path, text}]`) the agent reads and cites.

    A source counts only as whole module files: the HF repo's own modeling
    code (`auto_map` files, root `modeling_*.py`, or reference code such as
    `inference/model.py`), the installed transformers modeling module for its
    `model_type` when the repo ships none, or a custom model's own
    `forward.py`. The prep-time capture is reused when it holds such `.py`
    files; the model_info cache yields extracted `forward` excerpts in one
    `.json`, which does not count — then the HF repo's own code, then the
    installed transformers module.
    Files fetched from HF must also contain the decoder they import (see
    `_complete`). Never returns empty: raises `ForwardSourceUnavailable` when
    no channel yields a complete module source."""
    captured = forward_source or {}
    files = [f for f in captured.get("files") or []
             if str(f.get("path") or "").endswith(".py") and f.get("text")]
    if files:
        if captured.get("source") == "hf_modeling_files":
            _complete(model_id, files)
        return files
    from huggingface_hub import errors as hf_errors

    from backend.hf.forward_source import _try_hf_modeling_files, _try_local_transformers

    class _ResolvedShim:
        def __init__(self, cfg): self.config = cfg
    shim = _ResolvedShim(config or {})
    try:
        fetched = _try_hf_modeling_files(model_id, shim)
    except hf_errors.OfflineModeIsEnabled as exc:
        raise ForwardSourceUnavailable(
            f"Could not fetch the modeling source of '{model_id}': HF_HUB_OFFLINE "
            f"is set on the server and the files are not in its cache. Unset it "
            f"(or pre-fetch the repo); retrying as-is will not help.",
            retryable=False) from exc
    except (hf_errors.RepositoryNotFoundError,  # includes GatedRepoError
            hf_errors.RevisionNotFoundError, hf_errors.DisabledRepoError,
            hf_errors.HFValidationError) as exc:
        raise ForwardSourceUnavailable(
            f"Could not read the Hugging Face repo '{model_id}' to fetch its "
            f"modeling source ({type(exc).__name__}): the id is invalid, or the "
            f"repo (or the repo its auto_map points at) is missing, private, "
            f"gated or disabled. Check the model id and the HF token's access; "
            f"retrying as-is will not help.", retryable=False) from exc
    except Exception as exc:  # transport failure, Hub outage, offline cache miss
        raise ForwardSourceUnavailable(
            f"Could not fetch the modeling source of '{model_id}' from Hugging "
            f"Face ({type(exc).__name__}: {exc}). Check the server's network / "
            f"Hugging Face status and retry.", retryable=True) from exc
    fetched = [f for f in fetched if f.get("text")]
    if not fetched:
        native = _try_local_transformers(model_id, shim)
        if native:
            return native
        model_type = (config or {}).get("model_type") or "?"
        raise ForwardSourceUnavailable(
            f"'{model_id}' has no modeling source to read: its model_type "
            f"'{model_type}' is not an architecture in the installed "
            f"transformers, and its Hugging Face repo ships no modeling code "
            f"(no auto_map, modeling_*.py or reference model.py). A spec needs "
            f"the real forward code; retrying will not help.", retryable=False)
    _complete(model_id, fetched)
    return fetched


def _complete(model_id: str, files: list) -> None:
    """Refuse HF-fetched files that import their decoder / backbone from a
    module the fetch did not pull (a subdirectory, past the hop / file caps):
    that source is a wrapper plus config, not the model's forward. A network
    failure while following siblings already raised as retryable, so what
    remains here is permanent. Installed transformers modules and custom
    `forward.py` are self-contained and not checked (same gate as
    `describe_forward_source`)."""
    from backend.hf.forward_source import _unresolved_decoder_imports
    missing = _unresolved_decoder_imports(files)
    if missing:
        refs = ", ".join(f"{m['module']}.py ({', '.join(m['names'])})" for m in missing)
        raise ForwardSourceUnavailable(
            f"The modeling source of '{model_id}' is incomplete: the fetched "
            f"files import the decoder from {refs}, which could not be found "
            f"next to them on Hugging Face. A spec written without that code "
            f"may be wrong; retrying will not help.", retryable=False)


def get_ground_truth(model_id: str) -> Tuple[str, dict]:
    """Build ground truth sections: model structure + forward logic +
    tensor patterns."""
    sections = []
    # The same hub reads the snapshot writer makes (`hf.snapshot`): what the
    # prompt shows is what the snapshot records.
    try:
        config = fetch_config_raw(model_id)
    except (OSError, json.JSONDecodeError, *HUB_UNREACHABLE) as e:
        logger.warning(f"Failed to fetch config.json for {model_id}: {e}")
        config = {}

    model_info = get_model_info(model_id)
    status = model_info.get("_model_info_status", {})
    model_str = model_info.get("model_str", "")
    source = model_info.get("source", {})

    if status.get("success") and model_str:
        sections.append(f"""## Ground Truth 1: Model Architecture (from transformers)

This is the AUTHORITATIVE module hierarchy. Map each module to a component from the library.

```
{model_str}
```""")

    # Forward methods — show how modules connect (block pattern, residuals,
    # heterogeneous logic).
    forwards = source.get("forward_methods", {})
    if status.get("success") and forwards:
        forward_parts = []
        for label, code in forwards.items():
            # Skip the top-level CausalLM forward (boilerplate); the rest is
            # one forward per distinct module class the layers hold, each
            # labelled by its class name (`backend.hf.model_info`).
            if label == "Model":
                continue
            forward_parts.append(f"### {label}.forward\n```python\n{code.strip()}\n```")
        if forward_parts:
            modeling_file = source.get("modeling_file", "")
            header = f"## Ground Truth 2: Forward Logic (from `{os.path.basename(modeling_file)}`)"
            sections.append(
                f"{header}\n\n"
                "One `forward()` per distinct module class inside the decoder "
                "layers (every layer kind of a hybrid), by class name. "
                "**THIS IS THE GOLD STANDARD FOR BLOCK WIRING.** Your block graph's "
                "node `in:` connections, norm placements, and residual adds MUST "
                "match the compute order in these `forward()` methods. Parameter "
                "count equality alone is insufficient — many topologies yield the "
                "same param total. Cite line ranges in comments when the wiring "
                "is non-obvious.\n\n" + "\n\n".join(forward_parts)
            )
    elif not status.get("success"):
        # Fallback: fetch raw modeling_*.py directly from the HF repo so the
        # agent always has authoritative forward logic to ground its wiring.
        raw_section = fetch_raw_modeling_sources(model_id, config)
        if raw_section:
            sections.append(raw_section)
        sections.append(build_model_info_failure_section(model_id, config, model_info))

    config_section = build_config_section(config, len(sections) + 1)
    if config_section:
        sections.append(config_section)

    # Capture the fetched metadata so the post-success snapshot writer
    # doesn't have to re-probe HF.
    tensor_capture: dict = {}
    tensor_section = get_tensor_section(model_id, config, meta_out=tensor_capture)
    if tensor_section:
        n = len(sections) + 1
        sections.append(
            f"## Ground Truth {n}: Weight Tensors (from safetensors)\n\n"
            f"Use this to verify param counts and identify dtypes.\n{tensor_section}"
        )

    # Pre-compute the forward_source payload in snapshot shape. Bypasses
    # the snapshot-first strategy so we capture the real upstream source.
    forward_source = capture_forward_source_for_snapshot(model_id, config)

    return "\n\n".join(sections), {
        "config": config,
        "has_transformers_structure": bool(status.get("success") and model_str),
        "model_info_status": status,
        "_snapshot_data": {
            "config": config,
            "metadata": (metadata_record(tensor_capture["metadata"])
                         if tensor_capture.get("metadata") else {}),
            "forward_source": forward_source,
            "model_str": model_str,
        },
    }


def assemble_task_prompt(system: str, ground_truth: str, task_body: str) -> str:
    """Assemble the final prompt from shared context and task-specific body."""
    return f"""{system}

---

{ground_truth}

---

{task_body}"""
