"""Module-execution validation — actually RUN the spec's blocks at basic configs.

The static checks (anchor realness, field contracts) can't tell whether a
component's DimMap / custom op / new primitive actually computes. This
runs the engine on every per-layer BLOCK TEMPLATE the model uses, at basic
configs (B=1, B=2, S=128), and asserts each:
  - doesn't crash (a broken module becomes a reported warning, not a
    validate() exception),
  - produces finite, non-negative flops/params,
  - is batch-linear (params batch-independent; forward FLOPs exactly ×2
    under B→2B — a precise invariant that needs no correct constant),
plus a model-level KV-cache sanity at a basic inference config.

Granularity is the block template, not the individual component: only
fully-bound top-level templates are computable in isolation — a
sub-component (swiglu_ffn needs its `d_ffn=d_ffn_expert` binding, head_split its input
rank) gets its dims/params from the call site, so running it standalone
gives false positives. Templates exercise their whole sub-module stack
transitively, so a broken new sub-component surfaces as its parent template
crashing / going non-finite. Synthesized template names embed their
components (`..._attn-X_ffn-Y`), which localizes the failure; templates
touching a newly-authored module are flagged via `new`.

Layer-2 discipline: returns `{kind, template, detail, new}` warnings and
NEVER raises.
"""
from __future__ import annotations

import math
from typing import List, Optional, Set

from backend.arch.eval.projection import project_blocks
from backend.arch.spec_tree import iter_blocks

from ._finding import Finding

_S0 = 128  # basic sequence length for the smoke configs


def _finite_nonneg(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v) and v >= 0


def _top_templates(resolved) -> List[str]:
    """Distinct block templates the model instantiates, first-use order:
    every block invocation `iter_blocks` enumerates — heterogeneous
    per-group assignments and nested groups included, an assignment that
    selects no layer excluded."""
    return list(dict.fromkeys(
        meta.template_name for meta in iter_blocks(resolved)))


def run_module_exec_checks(resolved, new_components: Optional[Set[str]] = None) -> List[dict]:
    """Run each block template the model uses through eval at basic
    configs and assert it computes sanely. Best-effort; never raises."""
    out: List[dict] = []

    # Newly-authored / highest-risk modules: caller-flagged + spec custom ops.
    new: Set[str] = set(new_components or set())
    new |= set(resolved.spec.custom_ops or {})

    def _is_new(tmpl: str) -> bool:
        # Synthesized template names embed component names (attn-X_ffn-Y);
        # flag a template touching a new module.
        return tmpl in new or any(n in tmpl for n in new)

    analyses = {}
    analysis_error = None

    def _project(batch: int) -> None:
        nonlocal analysis_error
        try:
            analyses[batch] = project_blocks(
                resolved, batch=batch, input_tokens=_S0,
            )
        except Exception as exc:
            analysis_error = analysis_error or exc

    for batch in (1, 2):
        _project(batch)

    for tmpl in _top_templates(resolved):
        is_new = _is_new(tmpl)
        if analysis_error is not None:
            out.append(Finding(
                kind="module_exec_error",
                template=tmpl,
                detail=f"eval analysis raised for model-referenced block "
                       f"{tmpl!r}: {type(analysis_error).__name__}: "
                       f"{analysis_error}",
                extra={"new": is_new},
            ))
            continue

        r1 = analyses[1].get(tmpl)
        r2 = analyses[2].get(tmpl)
        if r1 is None or r2 is None:
            out.append(Finding(
                kind="module_exec_error",
                template=tmpl,
                detail=f"eval analysis omitted model-referenced block "
                       f"{tmpl!r} at B={{1,2}},S={_S0}",
                extra={"new": is_new},
            ))
            continue

        if not _finite_nonneg(r1.total_flops) or not _finite_nonneg(r1.total_params):
            out.append(Finding(
                kind="module_nonfinite",
                template=tmpl,
                detail=f"block {tmpl!r} produced non-finite/negative "
                       f"flops={r1.total_flops!r} params={r1.total_params!r}.",
                extra={"new": is_new},
            ))
            continue

        if r1.total_params != r2.total_params:
            out.append(Finding(
                kind="module_batch_noninvariant",
                template=tmpl,
                detail=f"block {tmpl!r}: total_params changed with batch "
                       f"(B=1:{r1.total_params:,} vs B=2:{r2.total_params:,}) — "
                       "params must be batch-independent.",
                extra={"new": is_new},
            ))
        if r1.total_flops > 0 and r2.total_flops != 2 * r1.total_flops:
            out.append(Finding(
                kind="module_batch_nonlinear",
                template=tmpl,
                detail=f"block {tmpl!r}: forward FLOPs not ×2 under B→2B "
                       f"(B=1:{r1.total_flops:,} vs B=2:{r2.total_flops:,}) — "
                       "forward FLOPs must be linear in batch size.",
                extra={"new": is_new},
            ))

    out.extend(_kv_sanity(resolved))
    return out


def _kv_sanity(resolved) -> List[dict]:
    """Run the per-token KV cache at a basic inference config (bf16) and
    assert the result is finite & non-negative. Catches KV topologies that
    blow up or go nonsensical — the 128TB family at the inference layer."""
    try:
        from backend.arch.eval.cache_view import cache_view_from_analysis
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.layout.parallel_config import ParallelConfig
        analysis = Deployment(
            Program.of(resolved), config=ParallelConfig(),
        ).at(batch=1, input_tokens=128, logit_positions=1).analyze()
        reader = cache_view_from_analysis(analysis)
    except Exception as exc:
        return [Finding(
            kind="module_kv_error",
            template="<model>",
            detail=f"per-token KV cache computation raised: "
                   f"{type(exc).__name__}: {exc}",
            extra={"new": False},
        )]
    # The sanity signal: growth slope + constant floor, plus one storage
    # evaluation at a realistic length — the laws themselves have to
    # produce a finite, non-negative number, which is the 128TB family's
    # tripwire at the inference layer.
    try:
        growth, floor = reader.growth_and_floor()
        kv_total = growth + floor + reader.storage_bytes_at(4096)
    except Exception as exc:
        return [Finding(
            kind="module_kv_error",
            template="<model>",
            detail=f"per-token KV cache computation raised: "
                   f"{type(exc).__name__}: {exc}",
            extra={"new": False},
        )]
    if kv_total is not None and not _finite_nonneg(kv_total):
        return [Finding(
            kind="module_kv_insane",
            template="<model>",
            detail=f"KV bytes non-finite/negative: {kv_total!r}.",
            extra={"new": False},
        )]
    return []
