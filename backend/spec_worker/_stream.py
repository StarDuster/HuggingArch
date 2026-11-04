"""Spec-only prefetch-log helper.

`sse` and `format_usage_log` moved to `backend._agent_runner.runner`
when the three workers were consolidated. This file now only houses
`build_prefetch_logs`, which interprets HF model_info / transformers
status and is spec-specific.
"""
from __future__ import annotations


def build_prefetch_logs(meta: dict) -> list[str]:
    """Build logs that should be shown before launching the agent."""
    status = meta.get("model_info_status", {}) or {}
    if status.get("success"):
        return []

    config = meta.get("config", {}) or {}
    current_tf = status.get("transformers_version") or "unknown"
    target_tf = config.get("transformers_version") or "unknown"
    model_type = config.get("model_type") or "unknown"
    error = status.get("error") or "unknown error"

    return [
        f"Local transformers {current_tf} could not build this model skeleton for model_type `{model_type}`.",
        f"Checkpoint metadata requests transformers `{target_tf}`.",
        f"Structure extraction failure: {error}",
        "Falling back to unknown-structure discovery using config.json, tensor names, and auxiliary sources.",
    ]
