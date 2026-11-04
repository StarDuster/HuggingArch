"""A rendered block graph must add up to the module's own totals.

Rendering is a view over the eval program: whatever it folds or regroups for
display, its nodes (plus the parameter siblings folded into them as
`absorbed_params`) must sum to the program's totals. The fold used to key on
`is_bare_parameter()` — a checkpoint-NAMING property — so `linear_param`
(bare-named, but it computes `x @ W`) was folded as if compute-free and its
FLOPs vanished from the rendered graph: 22.5 GFLOPs on DSv4's MoE router at
4096 tokens. These three components are the known instances.
"""
from __future__ import annotations

import pytest

from backend.arch.eval.transforms import analyze_isolated
from backend.arch.eval.precision import AS_SHIPPED
from tests.snapshots import bench_call


def _host(model_id: str, component: str):
    from tests.snapshots import shared_loader
    from backend.playground import module_bench as mb
    from tests.snapshots import resolve_snapshot
    loader = shared_loader()
    base = resolve_snapshot(model_id, loader)
    return mb, mb.inject_component(base, component, loader)


def _sum(nodes, key):
    return sum(int(n.get(key) or 0)
               + sum(int(a.get(key) or 0) for a in (n.get("absorbed_params") or []))
               for n in nodes)


@pytest.mark.parametrize("component,model_id", [
    ("dsv4_moe_hash", "deepseek-ai/DeepSeek-V4-Pro"),
    ("dsv4_moe_sigmoid", "deepseek-ai/DeepSeek-V4-Pro"),
    ("moe_ffn_fused_scaled_shared", "Qwen/Qwen3-Coder-Next"),
])
def test_rendered_nodes_sum_to_module_totals(component, model_id):
    mb, host = _host(model_id, component)
    # The drawing and the report are of one call: the host's own call on
    # its backbone, as the bench isolates it.
    kw = dict(batch=1, input_tokens=512, dim_overrides={},
              site=bench_call(host, component))
    report = mb.project_block(host, component, **kw)
    nodes = mb._block_graph(host, component, precision=AS_SHIPPED, **kw)["nodes"]
    assert _sum(nodes, "flops") == int(report.total_flops)
    assert _sum(nodes, "params") == int(report.total_params)
    # Weights add up to the stored copy: each node carries the scales of the
    # weights it holds. DeepSeek-V4's FP8 expert bank draws one expert, and
    # its scales repeat with the experts the bank counts.
    stored, _parts = mb._weight_storage(
        analyze_isolated(host, component, {}, kw["site"]), kw["site"],
        bound=False)
    assert _sum(nodes, "weight_bytes") == stored["bytes"]
    # The FLOP-bearing weight stays a node of its own, inside the expert
    # template it belongs to; only pure weight sources (no inputs) fold into
    # their consumer.
    def ops(ns):
        for n in ns:
            yield n.get("op")
            yield from ops(n.get("children") or [])
    assert "linear_param" in set(ops(nodes))


@pytest.mark.regression  # walks every shipped snapshot
def test_no_rendered_input_points_at_a_missing_node():
    """Every edge the renderer draws names a node it drew. A `repeat: 0`
    sibling is not built and leaves no edge: moe_out's `shared_experts` on a
    model with no shared expert is read through to `combine`."""
    from tests.snapshots import shared_loader
    from backend.rendering.builders import build_block_graph_data
    from tests.snapshots import SNAPSHOT_DIR, any_snapshot, resolve_snapshot

    def walk(nodes, acc):
        for n in nodes:
            acc.append(n)
            walk(n.get("children") or [], acc)

    loader = shared_loader()
    dangling = []
    any_snapshot()
    for sp in sorted(SNAPSHOT_DIR.glob("*.json")):
        name = sp.stem
        # present-but-unresolvable is a failure, not a skip
        resolved = resolve_snapshot(name, loader)
        blocks, _ = build_block_graph_data(resolved, batch_size=2, seq_len=128)
        for tmpl, entry in blocks.items():
            nodes = []
            walk(entry.get("nodes", []), nodes)
            ids = {n["id"] for n in nodes}
            for n in nodes:
                for ref in n.get("inputs") or []:
                    if ref in ids or any(ref.startswith(i + ".") for i in ids):
                        continue
                    dangling.append(f"{name}:{tmpl}:{n['id']} <- {ref}")
    assert not dangling, "\n".join(dangling[:20])


def _val(n, key):
    return int(n.get(key) or 0) + sum(int(a.get(key) or 0) for a in (n.get("absorbed_params") or []))


@pytest.mark.regression  # walks every shipped snapshot
def test_expanding_a_node_shows_all_of_its_compute():
    """Every composite node equals what it expands into: its children (plus
    their folded parameters), times `repeat` for a bank drawn as one
    representative. A router used to hide its own gate matmul — a no-module
    leaf `linear` dropped by module absorption — so the expanded router showed
    less than its total on every MoE model.

    Params must match exactly. FLOPs match exactly except for repeated banks,
    where the parent prices all tokens at once and a representative expert
    prices its ceil'd share, so the two can differ by rounding."""
    from tests.snapshots import shared_loader
    from backend.rendering.builders import build_block_graph_data
    from tests.snapshots import SNAPSHOT_DIR, any_snapshot, resolve_snapshot

    any_snapshot()
    loader = shared_loader()
    bad = []

    def walk(nodes, where):
        for n in nodes:
            ch = n.get("children") or []
            if ch:
                rep = int(n.get("repeat") or 1)
                df = _val(n, "flops") - rep * sum(_val(c, "flops") for c in ch)
                dp = _val(n, "params") - rep * sum(_val(c, "params") for c in ch)
                flops_ok = df == 0 if rep == 1 else abs(df) <= 1e-6 * max(1, _val(n, "flops"))
                if dp or not flops_ok:
                    bad.append(f"{where}:{n['id']} flops{df:+d} params{dp:+d} (repeat {rep})")
                walk(ch, where)

    for sp in sorted(SNAPSHOT_DIR.glob("*.json")):
        resolved = resolve_snapshot(sp.stem, loader)
        blocks, _ = build_block_graph_data(resolved, batch_size=2, seq_len=128)
        for tmpl, entry in blocks.items():
            walk(entry.get("nodes", []), f"{sp.stem}:{tmpl}")
    assert not bad, "\n".join(bad[:20])


def _layer_carved(model_id: str, tensors_re: str):
    """`model_id` with the matching weight of its last three layers saved at
    BF16 instead of the width the rest of the stack uses.

    The shape under test, BUILT rather than looked for in the corpus: a node
    whose layers store ONE tensor at TWO saved formats, which a layer-scoped
    `quant_context.overrides` row is how a spec says. `Analysis.tensors()` is
    then one row per layer segment and each row carries that SEGMENT's share
    of the node's copies, so a reader that takes a node-level fact off one row
    takes it off three of the fifty-eight layers that run the bank.
    """
    from backend.arch.schema.quant import OverrideEntry, OverrideSelector
    from tests.snapshots import resolve_snapshot

    base = resolve_snapshot(model_id)
    context = base.spec.quant_context
    tail = OverrideEntry(
        select=OverrideSelector(
            layers={"range": ["n_layers - 3", "n_layers - 1"]},
            tensors_re=tensors_re),
        dtype="BF16")
    # `model_copy`, not a deep copy: the spec holds sentinels whose identity
    # matters to the tree walk.
    return base.with_spec(base.spec.model_copy(update={
        "quant_context": context.model_copy(update={
            "overrides": [tail, *context.overrides]})}))


def test_a_node_stored_at_two_formats_is_priced_over_all_its_copies():
    """A node whose layers save one tensor at two widths is drawn once, so
    the one width it is drawn at is the mean over its copies.

    The renderer used to read the node's copy count off a tensor row and
    refuse when the rows disagreed (`uid 233 stores tensors at 63 and 3
    copies` on Step-5-Preview, whose routed expert bank is FP8 in 63 of its
    66 layers and BF16 in 3), which aborted the whole page. The copy count is
    not a row's to answer: a row is one layer SEGMENT and the walk publishes
    the node's own count as `Analysis.instances`.
    """
    from backend.arch.eval import capacity as capacity_mod
    from backend.arch.eval.dim import value_of
    from backend.arch.eval.formats import known_dtype_bytes
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.rendering._memory import stored_weights

    resolved = _layer_carved("deepseek-ai__DeepSeek-V3",
                             r"mlp\.experts\.X\.gate_proj\.weight$")
    # The walk the page's render runs on — the layout rule's per-rank face,
    # which is the one `stored_weights` reads (`local_tensors`).
    analysis = Deployment(Program.of(resolved)).at(
        input_tokens=16).checkpoint()
    rows_by_tensor: dict = {}
    for row in analysis.local_tensors():
        if row.is_metadata or row.is_buffer:
            continue
        rows_by_tensor.setdefault((row.uid, row.tensor_index), []).append(row)
    carved = [(key, rows) for key, rows in rows_by_tensor.items()
              if len(rows) > 1]
    assert len(carved) == 1, (
        "the fixture stopped building the shape under test: expected exactly "
        f"one tensor saved at several widths across its layers, got {len(carved)}")
    (uid, _), rows = carved[0]

    node = analysis.program.by_uid(uid)
    copies = int(value_of(analysis.instances[node]))
    assert copies == sum(int(value_of(row.count)) for row in rows), (
        "a tensor's layer segments cover the node's copies between them")
    # Every segment is the same tensor at the same shape, so its elements are
    # the node's per-copy elements — counted once, not once per segment.
    assert len({int(value_of(row.elements)) for row in rows}) == 1
    elements = int(value_of(rows[0].elements))

    held = stored_weights(analysis)[uid]
    ledger = capacity_mod.weights(analysis).by_uid[uid]
    assert held.elements == float(elements)
    assert held.width == pytest.approx(
        float(ledger.parameters) / copies / elements)
    # And it reads as the mix it is: strictly between the two widths. A width
    # taken off either segment's share of the copies lands on one of them.
    widths = sorted(known_dtype_bytes(row.storage_dtype) for row in rows)
    assert widths[0] < held.width < widths[-1]
