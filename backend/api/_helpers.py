"""Cross-route helpers used by /analyze + /supported-models.

Two responsibilities, one file because both are small:
  • `fetch_forward_source_payload(model_id, resolved_spec)` — wraps
    `describe_forward_source` for the /analyze response, capping per-file
    text size and degrading gracefully on failure.
  • `load_supported_model_list()` — reads the curated model-id list at
    `backend/arch/supported_models.txt`, drives the /supported-models
    autocomplete.
"""
from __future__ import annotations

import os
from typing import List

from backend.hf.forward_source import describe_forward_source


_SUPPORTED_LIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "arch", "supported_models.txt",
)


def fetch_forward_source_payload(model_id: str, resolved_spec) -> dict:
    """Best-effort wrapper: any exception degrades to an empty payload so
    /analyze doesn't fail over an unavailable forward source."""
    try:
        info = describe_forward_source(model_id, resolved_spec)
    except (OSError, ValueError):
        return {"source": "", "files": [], "paths": [], "text": ""}
    # Per-file cap — transformers modeling files can top 500KB; the
    # frontend shows each in its own scrollable frame but we still cap to
    # keep the JSON payload reasonable.
    max_per_file = 300_000
    truncated_any = False
    capped_files = []
    for f in info.get("files") or []:
        t = f.get("text") or ""
        if len(t) > max_per_file:
            truncated_any = True
            t = t[:max_per_file] + (
                f"\n\n# … truncated ({len(f['text']) - max_per_file} more chars) …\n"
            )
        capped_files.append({"path": f.get("path", ""), "text": t})
    out = {
        "source": info.get("source", ""),
        "files": capped_files,
        "paths": info.get("paths") or [],
        # Joined `text` kept for legacy copy-all use; length not capped
        # additionally since each constituent already is.
        "text": "\n\n".join(
            f"# ==== {x['path']} ====\n{x['text']}" for x in capped_files
        ),
    }
    if truncated_any:
        out["truncated"] = True
    return out


def load_supported_model_list() -> List[str]:
    """Read curated model IDs from backend/arch/supported_models.txt.
    Returns [] if the file is missing. Ignores blank lines and # comments."""
    try:
        with open(_SUPPORTED_LIST_PATH, "r", encoding="utf-8") as f:
            ids = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ids.append(line)
            return ids
    except OSError:
        return []
