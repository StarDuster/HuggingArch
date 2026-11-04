"""Fusion-scheme YAML registry.

Loads `backend/inference/fusion_plans/*.yaml` into a `Dict[str, FusionOpSpec]`
registry. Each YAML carries exactly two top-level keys:

  - `fused_op`      — the registry name (also matches the framework-binding
                      `op_implementations` key; required).
  - `naive_pattern` — the subgraph the matcher should look for plus the
                      `io_mapping` from fused-op operand names to
                      matched-node ports.
  - `allocates_output` — optional: where a framework's source shows this
                      launch writes its output into a buffer of its own
                      rather than rewriting its input in place, for a plan
                      that absorbs a `slice` / `concat` re-partition
                      (`arch.eval.fusion._passed_through`). A launch that
                      writes its output quantised names the
                      `quantise_activation` cast as a pattern member
                      instead: that is topology, not a declaration.
  - `output_dtype`  — optional: `{<io_mapping output port>: <dtype>}`, the
                      dtype the launch writes an output at where the
                      frameworks' source shows it is not the activation
                      dtype (MiniMax-M3's fp32 block scores, `output`;
                      sglang's fp32 GDN gates, `g_out` / `beta_out`). Only
                      on a fused op with no primitive registration: a
                      registered op states its output width on the op
                      (`Fusion.named` refuses the restatement).
  - `output_scale_tile` — optional: `{rows: <int>, groups: <int>}`, the
                      tile the launch pads the quantised output record's
                      scale sidecar to (FlashInfer's F8_128x4 on DeepSeek-
                      V4.1's q-quantising a-norm). Only meaningful with
                      a `quantise_activation` member; the bytes stay derived in
                      `traffic`, this states the record's layout law.

That is the entire surface. `FusionOpSpec` is `extra="forbid"`, so any
other field is rejected at load time. Cost arithmetic (flops / params),
dtype hints, taxonomy tags, and shape templates were all removed once
it became clear they were either redundant with matched analysis layers
or never consumed at all — see commit history if you're
about to "add a useful-looking field".

Per the CLAUDE.md §2.6 invariant, this directory lives under
`backend/inference/`, NOT `backend/arch/`, because fusion is an
analysis-layer concern; the spec language stays analysis-agnostic.
Framework bindings sit alongside under `backend/inference/framework_bindings/`.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import os
from typing import Dict

import yaml as _yaml

from backend.arch.schema import FusionOpSpec


def _load_fusion_plans_dir(dir_path: str) -> Dict[str, FusionOpSpec]:
    """Load every *.yaml under `dir_path` into a fused_op -> FusionOpSpec map."""
    out: Dict[str, FusionOpSpec] = {}
    if not os.path.isdir(dir_path):
        return out
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".yaml"):
            continue
        path = os.path.join(dir_path, fname)
        try:
            with open(path) as f:
                data = load_yaml(f) or {}
        except _yaml.YAMLError as e:
            raise ValueError(f"fusion_plans/{fname}: YAML parse error: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(
                f"fusion_plans/{fname}: top-level must be a mapping, "
                f"got {type(data).__name__}."
            )
        unknown_top_level = set(data.keys()) - {
            "fused_op", "naive_pattern", "allocates_output", "output_dtype",
            "output_scale_tile"}
        if unknown_top_level:
            raise ValueError(
                f"fusion_plans/{fname}: unknown top-level field(s) "
                f"{sorted(unknown_top_level)!r}. The fusion YAML only "
                f"accepts `fused_op`, `naive_pattern`, `allocates_output`, "
                f"`output_dtype` and `output_scale_tile`; everything else "
                f"(flops / params / dtype hints / shape templates / "
                f"taxonomy tags) was removed because it was redundant or "
                f"unconsumed. Put author docstrings in `#` comments."
            )
        fused_op = data.get("fused_op") or fname.removesuffix(".yaml")
        naive_pattern_data = data.get("naive_pattern")
        if naive_pattern_data is None:
            raise ValueError(
                f"fusion_plans/{fname}: missing required field `naive_pattern`."
            )
        output_dtype = data.get("output_dtype") or {}
        if not isinstance(output_dtype, dict):
            raise ValueError(
                f"fusion_plans/{fname}: `output_dtype` is a mapping from an "
                f"io_mapping output port to the dtype the launch writes it at "
                f"(`{{output: float32}}` for the primary output); got "
                f"{output_dtype!r}.")
        try:
            scheme = FusionOpSpec(
                naive_pattern=naive_pattern_data,
                allocates_output=str(data.get("allocates_output") or ""),
                output_dtype=output_dtype,
                output_scale_tile=data.get("output_scale_tile"))
        except Exception as e:
            raise ValueError(f"fusion_plans/{fname}: {e}") from e
        out[fused_op] = scheme
    return out


_FUSION_PLANS_DIR = os.path.dirname(__file__)
FUSION_SCHEMES: Dict[str, FusionOpSpec] = _load_fusion_plans_dir(_FUSION_PLANS_DIR)


def reload_fusion_schemes(dir_path: str | None = None) -> Dict[str, FusionOpSpec]:
    """Reload the registry from `dir_path` (default: this module's directory).

    Test helper — production code reads the module-level `FUSION_SCHEMES`
    populated at import time.
    """
    global FUSION_SCHEMES
    FUSION_SCHEMES = _load_fusion_plans_dir(dir_path or _FUSION_PLANS_DIR)
    return FUSION_SCHEMES
