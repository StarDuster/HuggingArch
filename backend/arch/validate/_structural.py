"""Structural-check warnings for spec block templates.

Best-effort comparison of every spec block template's compute graph
against its forward() source. Warns about likely topology errors (e.g.
spec missing a residual that forward() has). When the decoder layer's
forward() cannot be located — no source, or a construction the walk cannot
follow — that is one `forward_block_unlocated` warning saying where it
stopped.

Pure presentation: failures here don't fail validate. They surface FYI
warnings on PASSED runs because numbers can match while topology drifts.
"""
from __future__ import annotations

from dataclasses import replace

from ._finding import Finding
from backend.arch.eval.projection import project_blocks
from backend.arch.spec_tree import iter_blocks
from backend.arch.structural import check, locate_layer_forward
from backend.hf.forward_source import auto_model_class, describe_forward_source


def _collect_templates(resolved) -> list:
    """Distinct block templates the backbone instantiates, in first-use
    order: the block invocations `iter_blocks` enumerates outside every
    auxiliary group — the layers the decoder layer's forward() is compared
    against."""
    return list(dict.fromkeys(
        meta.template_name for meta in iter_blocks(resolved)
        if meta.aux_kind is None))


def _structural_diff_warnings(check, layer_forward, resolved, templates
                              ) -> list:
    """Each block template's compute graph vs the layer's forward()."""
    out: list = []
    for tmpl in templates:
        try:
            block = resolved.spec.get_block(tmpl)
        except KeyError:
            out.append(Finding(
                kind="unknown_template", template=tmpl,
                detail=f"model refers to block template {tmpl!r}, which the "
                       f"spec does not declare"))
            continue
        out.extend(replace(w, template=tmpl) for w in check(layer_forward, block))
    return out


def _role_coverage_warnings(resolved, templates) -> list:
    """Components that own params but declare no `role:`."""
    out: list = []
    # Role coverage check: a component referenced by the spec should
    # declare `role:` so the engine can attribute its params into
    # attention_params / ffn_params buckets. A roleless component whose
    # params land in no bucket silently understates per-layer attn/FFN
    # totals (the bug that hit MiniMax-M2 before role-tagging).
    #
    # But rolelessness is CORRECT for sub-components and shape utilities
    # (components/README.md): their leaves inherit the nearest role-bearing
    # ancestor's role (a `linear` inside `dsv4_indexer` inside
    # `dsv4_hca_attention` is `role=attention` at the leaf), and pure
    # shape wrappers (`hc_pre` / `head_split` / …) own no params at all.
    # So flag a roleless component ONLY when it DIRECTLY owns a
    # weight-bearing leaf whose inherited role is still None — the one
    # case where params genuinely escape every bucket. Keyed by the leaf's
    # direct parent component (`component_path[-1]`); top-level leaves
    # (embed / lm_head / final norm) carry no component path and are
    # legitimately roleless. Fall back to flagging all roleless components
    # if the leaf walk is unavailable.
    try:
        from backend.arch.spec_tree import iter_leaves
        unattributed_param_ops: set | None = set()
        for lt in iter_leaves(resolved):
            if lt.role is not None:
                continue
            cp = lt.component_path or ()
            if not cp:
                continue
            n = 1
            for d in lt.shape:
                n *= int(d)
            if n * int(lt.multiplicity) > 0:
                unattributed_param_ops.add(cp[-1][1])
    except Exception:
        unattributed_param_ops = None  # leaf walk failed → flag all roleless

    seen_comp: set = set()
    for tmpl in templates:
        try:
            block = resolved.spec.get_block(tmpl)
        except KeyError:
            out.append(Finding(
                kind="unknown_template", template=tmpl,
                detail=f"model refers to block template {tmpl!r}, which the "
                       f"spec does not declare"))
            continue
        stack = [block]
        while stack:
            bg = stack.pop()
            for gn in bg.graph.values():
                op = gn.op
                if not op or op.startswith("<"):
                    continue
                comp = (resolved.spec.components or {}).get(op) or \
                       (resolved.spec.blocks or {}).get(op)
                if comp is None or op in seen_comp:
                    continue
                seen_comp.add(op)
                if not comp.role and (
                    unattributed_param_ops is None
                    or op in unattributed_param_ops
                ):
                    out.append(Finding(
                        kind="component_missing_role",
                        template=tmpl,
                        detail=f"component {op!r} is missing top-level `role:` "
                               "(attention / ffn_dense / ffn_moe / router / "
                               "memory / hyper_connection / block / aux_block) "
                               "— params/flops won't be attributed to "
                               "attention, FFN, memory or residual params.",
                    ))
                if not comp.description:
                    out.append(Finding(
                        kind="component_missing_description",
                        template=tmpl,
                        detail=f"component {op!r} is missing top-level "
                               "`description:` — without it the component "
                               "won't be discoverable in the spec-gen prompt "
                               "manifest, future agents will skip past it.",
                    ))
                stack.append(comp)
    return out


def _low_role_coverage_warnings(resolved, templates) -> list:
    """Blocks where attention+FFN+memory+residual cover noticeably less
    than the total."""
    out: list = []
    # Per-block coverage: warn when attention+FFN sums to noticeably less
    # than the block's total params. Indicates either a missing role on a
    # sub-component, an inline FFN/attention without a call-site role
    # tag, or a substantial parallel branch outside the standard
    # attention/FFN axes that the spec author should explicitly account
    # for. 80% threshold leaves headroom for norms / residuals / quant
    # metadata which legitimately fall outside both buckets.
    blocks = project_blocks(resolved)
    for tmpl in templates:
        br = blocks.get(tmpl)
        if br is None:
            continue
        covered = ((br.attention_params or 0) + (br.ffn_params or 0)
                   + (br.memory_params or 0) + (br.residual_params or 0))
        total = br.total_params or 0
        if total > 0 and covered / total < 0.80:
            out.append(Finding(
                kind="low_role_coverage",
                template=tmpl,
                detail=f"attention+FFN+memory+residual params "
                       f"({covered/1e9:.2f}B) are "
                       f"{covered/total*100:.0f}% of block total "
                       f"({total/1e9:.2f}B). Likely a sub-component is "
                       "missing `role:` or an inline group needs a "
                       "call-site `role:` override.",
            ))
    return out


def _dead_end_warnings(resolved, templates) -> list:
    """Nodes computed but never consumed — the smoking gun for a missing
    stream binding."""
    out: list = []
    # Stream coverage: a graph node referencing a name that's not a
    # block input, sibling node, or declared `<param>` placeholder is
    # almost always an unwired stream — the spec-gen agent saw a
    # cross-iteration value in forward source and computed it as a node
    # but never declared it via `stream_out:`. Surface these as a warning.
    for tmpl in templates:
        try:
            block = resolved.spec.get_block(tmpl)
        except KeyError:
            out.append(Finding(
                kind="unknown_template", template=tmpl,
                detail=f"model refers to block template {tmpl!r}, which the "
                       f"spec does not declare"))
            continue
        # Outputs without a consumer that aren't in `outputs:` —
        # i.e. computed-and-thrown-away nodes. These are the smoking
        # gun for a missing stream binding.
        consumed: set = set()
        for n in block.graph.values():
            for src in n.source_names:
                consumed.add(src)
        from backend.arch.spec_tree import iter_blocks
        declared_outs = next(
            (set(meta.outputs) for meta in iter_blocks(resolved)
             if meta.template_name == tmpl),
            None)
        # A template named in `assignments:` but never instantiated has
        # no BlockMeta. Keep the recipe outputs so dead-end does not
        # fire on a call that is not in the tree.
        if declared_outs is None:
            declared_outs = set(block.outputs or ())
        stream_targets = set((block.stream_out or {}).values())
        for nname in block.graph:
            if nname in consumed:
                continue
            if nname in declared_outs:
                continue
            if nname in stream_targets:
                continue
            out.append(Finding(
                kind="dead_end_node",
                template=tmpl,
                detail=f"node {nname!r} is computed but has no consumer "
                       "(not in `outputs:`, not bound by `stream_out:`, "
                       "not used as input by any sibling). If forward "
                       "source threads this value across layers, declare "
                       "the stream under the stack's `streams:` and bind "
                       "this node via `stream_out:`. If it's genuinely "
                       "unused at inference time, drop the node from the "
                       "spec.",
            ))
    return out


def _dedupe_branch_warnings(warnings_out: list, templates) -> list:
    """One `forward_has_branches` warning is enough when the spec uses
    multiple templates — the heterogeneity is already expressed."""
    # Dedupe `forward_has_branches` across templates — spec intentionally
    # using multiple templates (via `assignments:`) already expresses the
    # heterogeneity, so one warning is enough to flag the underlying
    # conditional in forward.
    if any(w.kind == "forward_has_branches" for w in warnings_out) and len(templates) > 1:
        seen_branch = False
        deduped: list = []
        for w in warnings_out:
            if w.kind == "forward_has_branches":
                if seen_branch:
                    continue
                seen_branch = True
                # Multi-template spec → heterogeneity almost certainly
                # already expressed. Downgrade detail to a reminder.
                w = replace(w, template="<all>",
                            detail=w.detail + " (spec uses multiple templates — likely covered)")
            deduped.append(w)
        warnings_out = deduped
    return warnings_out


def _run_structural_checks(model_id: str, resolved) -> list:
    """Compare every spec block template against its forward() source,
    returning structural-check warnings. Best-effort — a forward() that
    can't be located is one `forward_block_unlocated` warning.

    Both the source fetcher and the structural checker are accessed
    only through their package facades; validate_spec knows nothing
    about their internals.
    """

    forward_source = describe_forward_source(model_id, resolved)
    if not forward_source["files"]:
        return [Finding(
            kind="forward_block_unlocated", template="<model>",
            detail=("no forward() source was found for this model (snapshot, "
                    "custom model, installed transformers, model_info cache "
                    "and the HF repo's modeling files were tried). The "
                    "comparison of block templates against forward() was "
                    "skipped."),
            extra={"paths": [], "trail": [], "stop": "no source files"})]

    templates = _collect_templates(resolved)
    layer_forward, trail, stop = locate_layer_forward(
        forward_source["files"], resolved.config or {}, auto_model_class)
    if layer_forward is None:
        warnings_out = [Finding(
            kind="forward_block_unlocated", template="<model>",
            detail=(f"in {forward_source['paths']}, following the model's "
                    f"construction ({' → '.join(trail) or 'no start'}) found "
                    f"no text decoder layer class: {stop}. The comparison of "
                    f"block templates against forward() was skipped."),
            extra={"paths": list(forward_source["paths"]),
                   "trail": list(trail), "stop": stop})]
    else:
        warnings_out = _structural_diff_warnings(
            check, layer_forward, resolved, templates)
    warnings_out += _role_coverage_warnings(resolved, templates)
    warnings_out += _low_role_coverage_warnings(resolved, templates)
    warnings_out += _dead_end_warnings(resolved, templates)
    return _dedupe_branch_warnings(warnings_out, templates)

