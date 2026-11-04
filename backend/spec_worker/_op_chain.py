"""Detect a spec-gen gap.json and chain the op writer into the same job."""
from __future__ import annotations

import json
import os
from typing import Optional

from backend.op_workers.proposal.runtime import GAP_DIR


def find_fresh_gap(model_id: str, started_at: float) -> Optional[dict]:
    """Newest gap.json for this model written during the current run."""
    if not os.path.isdir(GAP_DIR):
        return None
    matches: list[tuple[float, dict]] = []
    for name in os.listdir(GAP_DIR):
        if not name.endswith(".gap.json"):
            continue
        path = os.path.join(GAP_DIR, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime < started_at - 1:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                body = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if body.get("context_model_id") != model_id:
            continue
        if not body.get("op_name"):
            continue
        matches.append((mtime, body))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]
