"""Emit the Gallery control-bar axes for the React app.

React must not hand-write a second copy of `sweep_grid` presets.
`web/src/gallery/axes.json` is generated from this module; tests pin
the file to `GALLERY_*` / `AUDIT_PARALLEL`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from backend.inference.sweep_grid import (
    AUDIT_PARALLEL,
    FRONTIER_ACCEPTANCE_DEFAULT,
    FRONTIER_ACCEPTANCE_RATES,
    FRONTIER_DRAFT_TOKENS,
    GALLERY_GPUS,
    GALLERY_LANDING_GPU,
    GALLERY_PARALLEL_PRESETS,
)

_WEB_AXES = Path(__file__).resolve().parents[2] / "web" / "src" / "gallery" / "axes.json"


def _preset_label(p: Dict[str, int]) -> str:
    if not p:
        return "Single GPU · TP1"
    if "tp_attention" in p:
        n = int(p["tp_attention"])
        return f"{n}×GPU · TP{n} (attn+dense+expert)"
    ep = int(p.get("ep_moe", 1))
    dp = int(p.get("dp_attention", 1))
    n = max(ep, dp)
    return f"{n}×GPU · TP1/EP{ep}/CP1/DP{dp} (default)"


def axes_payload() -> Dict[str, Any]:
    presets: List[Dict[str, Any]] = [
        {"label": _preset_label(dict(p)), "value": dict(p)}
        for p in GALLERY_PARALLEL_PRESETS
    ]
    return {
        "landing_gpu": GALLERY_LANDING_GPU,
        "gpus": list(GALLERY_GPUS),
        "draft_tokens": list(FRONTIER_DRAFT_TOKENS),
        "acceptance_rates": list(FRONTIER_ACCEPTANCE_RATES),
        "acceptance_default": FRONTIER_ACCEPTANCE_DEFAULT,
        "parallel_presets": presets,
        "audit_parallel": dict(AUDIT_PARALLEL),
    }


def write_web_axes(path: Path | None = None) -> Path:
    dest = path or _WEB_AXES
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(axes_payload(), indent=2, ensure_ascii=False) + "\n")
    return dest
