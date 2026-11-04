"""The dashboard's FLOPs are the Inference breakdown's, summed by role —
plus what serving grafts onto the graph.

The dashboard (`projection.formulas.display_formulas`) states a model's
attention, FFN and total forward FLOPs; an estimate's breakdown states the
same step module by module (`phases.*.flops_by_module`). Both are the
backbone's — no aux stack, no MTP draft — so summed by the one role rule
(the outermost node of an attention or FFN role holds its subtree; a
wrapper's children are looked through), the breakdown is the dashboard,
for every model — up to the rows the SERVED graph adds: the grafted
activation casts and cache transforms (`materialize_cast`, each marked
`provenance="cast"` on the pre-fusion program), whether a graft stands as
its own row or a fused launch absorbed it and bills it (`Rewrite.origins`,
the closure law of `test_phase_flops_closure`). The two sides share no
arithmetic: the dashboard walks the logical program, the graft FLOPs are
read off the served walk's own layers, and the law is that their
difference is the breakdown. A model whose served graph grafts nothing
(Qwen3-30B-A3B, DeepSeek-V2-Lite) is held to plain equality without
paying for the second walk. The dashboard states the prefill step; the
decode step's breakdown is held to the same role sums over the decode
walk.
"""
from __future__ import annotations

import pytest

from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.projection.formulas import display_formulas
from backend.arch.eval.projection.result import (_ATTN_ROLES, _FFN_ROLES,
                                                 _WRAPPER_ROLES)
from backend.arch.eval.transforms import DECODE, Deployment
from backend.arch.snapshots import iter_snapshot_ids
from backend.inference.capacity import estimate_pending
from backend.inference.framework_bindings import (DEFAULT_FRAMEWORK,
                                                  ServingDeployment,
                                                  get_binding)
from backend.inference.module_rows import physical_module_rows
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.inference.step_eval import evaluate_step
from tests.snapshots import resolve_snapshot, snapshot_config

BATCH, PROMPT = 2, 1024


def _table_by_role(rows, roles, values=None) -> int:
    """The module table's FLOPs under `roles`: each row of one of them with
    its subtree (parents are Σ-children), a row of another role skipped
    with its subtree, a wrapper (or the block template's row) looked
    through. `values` overrides the column summed (the grafted-FLOPs
    subtree sums, when the serving side is reconciled)."""
    if values is None:
        values = [int(row["flops_per_rank"] or 0) for row in rows]
    total, below = 0, None
    for i, row in enumerate(rows):
        depth = int(row.get("depth") or 0)
        if below is not None and depth > below:
            continue
        below = None
        role = row.get("role") or None
        if role in roles:
            total += values[i]
            below = depth
        elif role not in (None, "block_template") and role not in _WRAPPER_ROLES:
            below = depth
    return total


def _walk_by_role(analysis, roles) -> int:
    """The walk's backbone FLOPs under `roles`, by the same rule over its
    accounting roots (`Program.contributions`)."""
    backbone = analysis.program.included_stack_names()

    def visit(node) -> int:
        if node.role in roles:
            cost = analysis.cost.values.get(node.uid)
            return int(value_of(cost.flops)) if cost is not None else 0
        if node.role is not None and node.role not in _WRAPPER_ROLES:
            return 0
        return sum(visit(child) for child in node.children or ())
    return sum(int(value_of(p.count)) * visit(p.node)
               for p in analysis.program.contributions if p.stack in backbone)


def _serving_graft_flops(resolved, phase: str, rows) -> dict:
    """The FLOPs the phase's served table bills beyond the model's graph,
    summed by the role rule: the grafted activation casts and cache
    transforms (`materialize_cast`, `provenance="cast"` on the pre-fusion
    program). A graft that stands alone is its own row; one a fused launch
    absorbed is billed by the launch's row (`Rewrite.origins` maps the
    launch to the graft), so the grafted share is read off the rewrite and
    attributed to the carrier row.

    Read off a re-run of the phase's serving walk (`pipeline.analyze`)
    under the estimate's own binding (the default framework's, deployed on
    the request's card and activation dtype) — guarded to reproduce the
    estimate's table row for row, so the uids the graft shares are keyed
    by are the table's.
    """
    step = evaluate_step(resolved, ParallelConfig(), phase=phase, batch=BATCH,
                         input_tokens=PROMPT, output_tokens=0)
    binding = get_binding(DEFAULT_FRAMEWORK).deployed(ServingDeployment.at(
        gpu_sku="B200",
        activation_dtype=step.analysis.resolved_precision.model_dtype))
    result = analyze(step.spec, parallel=ParallelConfig(), step_eval=step,
                     framework_binding=binding, breakdown=True)
    mine = physical_module_rows(result, result.per_op_timing)

    def identity(row) -> tuple:
        return (row.get("aggregation_id"), row.get("name"),
                row.get("role") or "", int(row.get("depth") or 0),
                int(row["flops_per_rank"] or 0))
    if [identity(row) for row in mine] != [identity(row) for row in rows]:
        raise AssertionError(
            f"the re-run {phase} walk's module table is not the estimate's")

    grafted = {node.uid for node in result.pre_program.nodes()
               if getattr(node, "provenance", None) == "cast"}
    graft_of = {}
    for node in result.program.nodes():
        sources = result.rewrite.sources(node.uid) or (node.uid,)
        flops = sum(
            int(value_of(result.pre_cost.values[origin].flops))
            * int(value_of(result.pre_instances.values[origin]))
            for origin in sources if origin in grafted)
        if flops:
            graft_of[node.uid] = flops

    # A held row's figure is its subtree's sum, so its grafted share is its
    # subtree's grafted sum (grafts are leaves: no row holds both a graft
    # and a child with one).
    own = [graft_of.get(row.get("uid"), 0) if row.get("uid") is not None else 0
           for row in mine]
    subtree = [0] * len(mine)
    for i in range(len(mine) - 1, -1, -1):
        depth = int(mine[i].get("depth") or 0)
        total = own[i]
        j = i + 1
        while j < len(mine) and int(mine[j].get("depth") or 0) > depth:
            total += own[j]
            j += 1
        subtree[i] = total
    return {
        "attention": _table_by_role(mine, _ATTN_ROLES, subtree),
        "ffn": _table_by_role(mine, _FFN_ROLES, subtree),
        "total": sum(subtree[i] for i, row in enumerate(mine)
                     if int(row.get("depth") or 0) == 0),
    }


@pytest.mark.regression  # the whole corpus, an estimate each
@pytest.mark.parametrize("model_id", sorted(iter_snapshot_ids()))
def test_the_dashboard_is_the_breakdown_summed_by_role(model_id):
    resolved = resolve_snapshot(model_id)
    estimate = estimate_pending(
        config=snapshot_config(model_id), model_id=model_id, resolved=resolved,
        gpu_sku="B200", parallel=ParallelConfig(), num_gpus=1,
        gpu_memory_fraction=0.9, avg_input_tokens=PROMPT, avg_output_tokens=0,
        prefill_batch_size=BATCH, decode_batch_size=BATCH,
        selection_mode="theoretical_only", measurement_scope=None).finish()

    dashboard = display_formulas(resolved, batch_size=BATCH, seq_len=PROMPT)
    prefill = estimate["phases"]["prefill"]
    want = {key: int(dashboard[key]["result"].replace(",", ""))
            for key in ("attention", "ffn", "total")}
    got = {
        "attention": _table_by_role(prefill["flops_by_module"], _ATTN_ROLES),
        "ffn": _table_by_role(prefill["flops_by_module"], _FFN_ROLES),
        "total": prefill["op_table_total"]["flops_per_rank"]}
    if got != want:
        # The one allowed difference: the grafted casts' and cache
        # transforms' FLOPs, which the served graph bills and the logical
        # walk never saw.
        graft = _serving_graft_flops(resolved, "prefill",
                                     prefill["flops_by_module"])
        got = {key: got[key] - graft[key] for key in got}
    assert got == want

    decode = estimate["phases"]["decode"]
    walk = Deployment(Program.of(resolved)).at(
        batch=BATCH, input_tokens=PROMPT, output_tokens=0,
        logit_positions=1).during(DECODE).analyze()
    got = {
        "attention": _table_by_role(decode["flops_by_module"], _ATTN_ROLES),
        "ffn": _table_by_role(decode["flops_by_module"], _FFN_ROLES)}
    want = {"attention": _walk_by_role(walk, _ATTN_ROLES),
            "ffn": _walk_by_role(walk, _FFN_ROLES)}
    if got != want:
        graft = _serving_graft_flops(resolved, "decode",
                                     decode["flops_by_module"])
        got = {key: got[key] - graft[key] for key in got}
    assert got == want
