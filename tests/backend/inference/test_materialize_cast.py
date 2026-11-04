"""The activation-cast graft (`backend.inference.materialize_cast`).

A quantised matmul reads a narrow copy of its operand, and a kernel makes
one — vLLM's `QuantFP8`, sglang's `per_token_group_quant_fp8` — so the copy
is a node of the serving graph, grafted between `materialize_comm` and the
framework's fusion rewrite (`serving.rewrite_serving_graph`). These tests
hold the graft's contract on an FP8 model (DeepSeek-V4-Flash: FP8 weights
with an FP8 activation format on its matmuls):

  * one `quantise_activation` node per conversion no producer absorbed,
    wired between the edge's producer and the consumer that needed it;
  * the conversion's bytes moved off the consumer onto that node — the
    consumer no longer carries `convert_bytes`, and reads the edge at the
    narrow width the cast declares;
  * the node's own facts (shape, cost, count) are its consumer's.
"""
from __future__ import annotations

from fractions import Fraction
from math import ceil, prod
from typing import Any, Dict

import pytest

from tests.snapshots import resolve_snapshot

from backend.arch.eval import cache as cache_mod
from backend.arch.eval import traffic as traffic_layer
from backend.arch.eval.dim import same_law, value_of
from backend.arch.eval.values import product
from backend.arch.layout.physical import physical_kernel_shapes
from backend.inference.framework_bindings import get_binding
from backend.inference.materialize_cast import cast_facts
from backend.inference.parallel_config import ParallelConfig
from backend.inference.serving import _on_program, rewrite_serving_graph
from backend.inference.step_eval import evaluate_step

_FP8_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
_BF16_MODEL = "meta-llama/Llama-3.2-1B"


def _analysis(model_id: str):
    resolved = resolve_snapshot(model_id)
    return evaluate_step(
        resolved, ParallelConfig(), phase="decode",
        batch=1, input_tokens=128, output_tokens=1,
    ).analysis


def _serving(model_id: str = _FP8_MODEL):
    return rewrite_serving_graph(
        _analysis(model_id), framework_binding=get_binding("vllm"))


def _flows_over_graft(analysis, serving, facts):
    """`Traffic` re-derived over the grafted pre-fusion program, with the
    casts' shapes in the geometry — the derivation the access rows are
    priced from."""
    pre = serving.pre_program
    full_shapes = _on_program(
        analysis.shapes, pre, serving.materialized.shapes)
    full_shapes = _on_program(full_shapes, pre, facts.full_shapes)
    geometry = traffic_layer.TrafficGeometry(
        shapes=full_shapes, whole_shapes=full_shapes,
        storage=analysis.storage, formats=analysis.formats,
        cache=cache_mod.derive(analysis))
    return traffic_layer.derive(analysis, geometry=geometry, program=pre)


def _record_bytes_by_hand(dims, fmt) -> Fraction:
    """Bytes a quantised activation record of shape `dims` occupies, counted
    the way the quantise kernel lays it out — not asked of the traffic
    derivation: every element at its dtype's bytes, and per row one scale
    for each `group` channels (a partial group still takes a scale), each
    at the scale dtype's bytes — or one per row for a per-token record (the
    DSA indexer's FP8 query, one scale a head). The byte sizes are spelled
    out here."""
    element_bytes = {"F8_E4M3": Fraction(1), "FP4_E2M1": Fraction(1, 2)}[
        fmt.dtype]
    scale_bytes = {"F8_E8M0": 1, "F32": 4}[fmt.scale_dtype]
    values = [Fraction(str(value_of(d))) for d in dims]
    inner, rows = values[-1], prod(values[:-1])
    if fmt.scale_granularity == "per-token":
        return rows * inner * element_bytes + rows * scale_bytes
    assert fmt.scale_granularity == "per-group", fmt.name
    return (rows * inner * element_bytes
            + rows * ceil(inner / fmt.group) * scale_bytes)


def _narrow(fmt, inner) -> float:
    """Bytes per element of a quantised record whose rows are `inner` wide,
    counted by hand (`_record_bytes_by_hand`): the element and its share of
    the row's scales."""
    inner = value_of(inner)
    return float(_record_bytes_by_hand((1, inner), fmt) / inner)


def _facts(analysis, serving):
    return cast_facts(
        serving.casts,
        local_shapes=physical_kernel_shapes(analysis),
        full_shapes=analysis.shapes,
        instances=analysis.instances,
    )


def test_no_conversions_no_graft():
    """A BF16 deployment's matmuls eat their edges as they arrive: nothing
    to graft, and the program is the comm materialisation's own."""
    serving = _serving(_BF16_MODEL)
    assert not serving.casts
    assert serving.casts.program is serving.materialized.program


def test_casts_stand_between_producer_and_consumer():
    analysis = _analysis(_FP8_MODEL)
    serving = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm"))
    assert serving.casts.sites
    casts = serving.casts
    # The deployment quantises matmul operands: the graft is not empty.
    assert casts
    pre = serving.pre_program
    for site in casts.sites:
        node = site.node
        consumer = pre.by_uid(site.consumer)
        # The consumer's input at the cast's position reads the cast now…
        ref = consumer.inputs_obj[site.index]
        assert ref.src is node, (consumer.display_path, site.index)
        # …and the cast reads the producer the edge carried before.
        producer = node.inputs_obj[0].src
        assert producer is not None
        assert node.op == "quantise_activation"
        assert node.provenance == "cast"
        # The node declares the WHOLE format it writes — the deployment's
        # required format, granularity and scale dtype with it — so its
        # write and every read of it price narrow and its cell is keyed by
        # it; the element dtype is the format's, not a second declaration.
        assert node.output_format is not None
        assert node.output_format.dtype
        assert node.output_dtype is None
        # The grafted node is reachable and classified by the rebuilt
        # program's own indexes.
        assert pre.by_uid(node.uid) is node
        assert pre.stack_of(node) == pre.stack_of(consumer)


def test_conversion_bytes_move_onto_the_cast():
    analysis = _analysis(_FP8_MODEL)
    serving = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm"))
    facts = _facts(analysis, serving)
    flows = _flows_over_graft(analysis, serving, facts)
    assert serving.casts.sites
    for site in serving.casts.sites:
        consumer_traffic = flows.values[site.consumer]
        # One format change, one price: with the cast standing on the edge
        # the consumer pays nothing for it.
        assert same_law(consumer_traffic.convert_bytes, 0)
        assert consumer_traffic.cast is None
        cast_traffic = flows.values.get(site.node.uid)
        assert cast_traffic is not None
        dims = facts.full_shapes.values[site.node.uid].out
        elements = product(dims)
        record = _record_bytes_by_hand(dims, site.node.output_format)
        # The cast reads the operand as the edge holds it (wide, BF16) and
        # writes the narrow copy it declares — FP8 elements and the scales
        # beside them; the consumer then reads that same record.
        assert same_law(cast_traffic.act_reads, 2 * elements)
        assert Fraction(str(value_of(cast_traffic.act_writes))) == record
        edge = consumer_traffic.in_edges[site.index]
        assert Fraction(str(value_of(edge.elements * edge.width))) == record


def test_one_cast_one_price_before_and_after_the_graft():
    """The same launch, priced on both sides of the graft: the record the
    derivation keeps before it (`CastLaunch.narrow_write` / `narrow_read`,
    on the consumer that needs the conversion) and the grafted node's own
    write and its consumer's read of it. All four are one tensor — FP8
    elements and the scales beside them — and each is checked against the
    record counted by hand (`_record_bytes_by_hand`) rather than against
    another output of the derivation. DeepSeek-V3's format carries an F32
    scale per 128 channels (3% of the write), V4-Flash's a UE8M0 one (<1%):
    both widths of scale are exercised."""
    for model in ("deepseek-ai/DeepSeek-V3", _FP8_MODEL):
        analysis = _analysis(model)
        # Before the graft: the walked program, whose consumers still read
        # their producers, records each conversion as a launch.
        before = traffic_layer.derive(analysis)
        serving = rewrite_serving_graph(
            analysis, framework_binding=get_binding("vllm"))
        facts = _facts(analysis, serving)
        after = _flows_over_graft(analysis, serving, facts)
        assert serving.casts.sites, model
        for site in serving.casts.sites:
            dims = facts.full_shapes.values[site.node.uid].out
            record = _record_bytes_by_hand(dims, site.node.output_format)
            edge = after.values[site.consumer].in_edges[site.index]
            prices = {
                "grafted write": after.values[site.node.uid].act_writes,
                "reader read": edge.elements * edge.width,
            }
            # A cast still on the edge it was grafted for is the launch the
            # walk recorded there; one moved above a copy (the routed
            # experts' cast, above the dispatch) quantises the copy's input,
            # and the walk's record is of the rows it no longer reads.
            launch = before.values[site.consumer].cast
            if launch is not None:
                assert launch.target == site.node.output_format
                prices["pre-graft write"] = launch.narrow_write
                prices["pre-graft read"] = launch.narrow_read
            for name, price in prices.items():
                assert Fraction(str(value_of(price))) == record, (
                    model, site.node.name, name, value_of(price), record)


def test_consumers_sharing_an_edge_share_the_cast():
    """A gated FFN's two projections convert the same tensor to the same
    format: one launch makes the copy (the frameworks' merged GEMM's single
    quantise call), so one node stands on the edge and both consumers read
    it — and the parallel fusion merging the two GEMMs still matches, which
    two per-consumer casts would have broken (the siblings no longer read
    the same tensor)."""
    analysis = _analysis(_FP8_MODEL)
    serving = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm"))
    sites_by_node: Dict[Any, list] = {}
    for site in serving.casts.sites:
        sites_by_node.setdefault(site.node.uid, []).append(site)
    shared = [sites for sites in sites_by_node.values() if len(sites) > 1]
    assert shared, "the gated FFNs' projections share an edge's cast"
    for sites in shared:
        # Distinct consumers of one edge, one node between them and it.
        assert len({site.consumer for site in sites}) == len(sites)
    assert any(match["fused_op"] == "merged_column_parallel_linear"
               for match in serving.fusion_matches)


def test_cast_facts_are_the_consumers_own():
    analysis = _analysis(_FP8_MODEL)
    serving = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm"))
    facts = _facts(analysis, serving)
    local_shapes = physical_kernel_shapes(analysis)
    for site in serving.casts.sites:
        uid = site.node.uid
        # The cast moves the operand the consumer's kernel is handed, per
        # rank and global, and runs as many times as the consumer runs.
        assert (facts.local_shapes.values[uid].out
                == tuple(local_shapes.values[site.consumer].ins[site.index]))
        assert (facts.full_shapes.values[uid].out
                == tuple(analysis.shapes.values[site.consumer].ins[site.index]))
        assert (facts.instances.values[uid]
                == analysis.instances.values[site.consumer])
        # Two elementwise passes over the tensor, as the op declares.
        elements = product(facts.full_shapes.values[uid].out)
        assert same_law(facts.full_cost.values[uid].flops, 2 * elements)


def test_cast_rows_in_the_priced_tables():
    """End to end (`pipeline.analyze`): the casts are rows of the pre- and
    post-fusion tables, counted like their consumers, and no row carries a
    conversion number beside them."""
    from backend.inference.pipeline import analyze

    resolved = resolve_snapshot(_FP8_MODEL)
    result = analyze(
        resolved, parallel=ParallelConfig(),
        framework_binding=get_binding("vllm"),
        batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
        breakdown=True)
    for rows in (result.pre_per_op_timing, result.per_op_timing):
        casts = [r for r in rows if r["op"] == "quantise_activation"]
        assert casts
        for row in casts:
            assert int(row["access_bytes_per_rank"]) > 0
            assert int(row["convert_bytes_per_rank"] or 0) == 0
    # A cast runs as often as the consumer it casts for (named for it,
    # `<consumer>__cast`, in its block template; it stands in the producer's
    # scope, so the path above the name is the producer's). Checked on the
    # pre-fusion table: after the rewrite the consumer may be an absorbed
    # member with no row of its own (a merged GEMM swallows the gated FFN's
    # two projections; their shared cast keeps its own).
    def leaf(row):
        template, _, path = row["aggregation_id"].rpartition("|")
        return template, path.rpartition(".")[2]

    by_leaf: Dict[Any, list] = {}
    for r in result.pre_per_op_timing:
        if r["op"] != "quantise_activation":
            template, name = leaf(r)
            by_leaf.setdefault((template, name), []).append(r)
            # A node of a component written as an op (`dsv4_indexer`) is
            # pathed `<node>__<op>`; its cast is named for the node.
            node, _, _op = name.partition("__")
            if node != name:
                by_leaf.setdefault((template, node), []).append(r)
    for row in (r for r in result.pre_per_op_timing
                if r["op"] == "quantise_activation"):
        template, name = leaf(row)
        consumers = by_leaf.get((template, name.removesuffix("__cast")), [])
        assert consumers, row["aggregation_id"]
        assert any(c["count_per_step"] == row["count_per_step"]
                   for c in consumers), row["aggregation_id"]
        for consumer in consumers:
            assert int(consumer["convert_bytes_per_rank"] or 0) == 0


def test_cast_rows_are_keyed_by_the_format_they_write():
    """A cast cell's dtype environment is the format the cast WRITES: the
    quantised element dtype, which elements share a scale, and the scales'
    dtype. Every FP8 cast used to be keyed `{activation: bf16, output:
    bf16}` — a copy no kernel serves — so every provider declined all of
    them (57k H100 cells over 18 models).

    Checked by a second route that shares nothing with the traffic
    derivation that now states the format: the consumer's own resident
    weight format names the activation format its kernel eats
    (`TensorFormat.activation_format`, from the scheme yaml)."""
    import json

    from backend.arch.activation_formats import ACTIVATION_FORMATS
    from backend.arch.dtypes import corpus_token
    from backend.arch.eval.formats import segments_of
    from backend.inference.benchmark_identity import RowIdentity
    from backend.inference.pipeline import analyze

    resolved = resolve_snapshot(_FP8_MODEL)
    result = analyze(
        resolved, parallel=ParallelConfig(),
        framework_binding=get_binding("vllm"),
        batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
        breakdown=True)
    identity = RowIdentity(result)
    walk = result.analysis
    program = walk.program
    # A cast is named for its consumer (`<consumer>__cast`) in the
    # consumer's block template, and stands in the producer's scope, so the
    # consumer is the leaf of that name in that template.
    def leaf(aggregation_id):
        template, _, path = str(aggregation_id).rpartition("|")
        return template, path.rpartition(".")[2]

    by_leaf = {}
    for node in program.nodes():
        by_leaf.setdefault(leaf(node.aggregation_id), []).append(node)
    # The casts that still launch: one a fused launch took in is no cell of
    # its own (`test_a_launch_that_takes_a_cast_in_writes_its_format`).
    casts = [r for r in result.per_op_timing
             if r["op"] == "quantise_activation"]
    assert casts
    for row in casts:
        template, name = leaf(row["aggregation_id"])
        consumers = by_leaf[(template, name.removesuffix("__cast"))]
        named = {rep.resident.activation_format
                 for consumer in consumers
                 for rep in segments_of(walk.formats.values, consumer.uid)
                 if rep.resident is not None}
        named.discard(None)
        assert len(named) == 1, (row["aggregation_id"], named)
        fmt = ACTIVATION_FORMATS[named.pop()]
        env = json.loads(identity.dtype_env(row))
        assert env == {
            "activation": "bf16",
            "output": corpus_token(fmt.dtype),
            "metadata": corpus_token(fmt.scale_dtype),
            "activation_granularity": fmt.granularity,
        }, row["aggregation_id"]


def test_a_launch_that_takes_a_cast_in_writes_its_format():
    """Fusion takes a cast in where a plan names `quantise_activation` as a
    member (DeepSeek-V4-Flash's clamped SwiGLU epilogue before `down_proj`).
    The cast's launch is then gone from the fused table, and the launch
    that absorbed it writes — and is keyed by — the format the cast was
    grafted for: one declaration, read where the tensor now lands
    (`traffic._written_formats`)."""
    from backend.arch.activation_formats import ACTIVATION_FORMATS
    from backend.inference.pipeline import analyze

    result = analyze(
        resolve_snapshot(_FP8_MODEL), parallel=ParallelConfig(),
        framework_binding=get_binding("vllm"),
        batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
        breakdown=True)
    pre_casts = [r for r in result.pre_per_op_timing
                 if r["op"] == "quantise_activation"]
    post_casts = [r for r in result.per_op_timing
                  if r["op"] == "quantise_activation"]
    assert len(post_casts) < len(pre_casts)
    grafted = {r["output_activation_format"] for r in pre_casts}
    absorbing = [r for r in result.per_op_timing
                 if r["op"] != "quantise_activation"
                 and r.get("output_activation_format")]
    assert {r["op"] for r in absorbing} >= {"silu_and_mul_with_clamp"}
    for row in absorbing:
        assert row["output_activation_format"] in grafted, row["aggregation_id"]
    # The launch that took the cast in writes the record the cast would
    # have — elements and scales, counted by hand over the tensor it
    # publishes — and nothing else where that is its only output (the
    # epilogue). A launch with a second output (the indexer query's, which
    # writes the scaled head weights beside the query) writes the record
    # and more.
    for row in absorbing + post_casts:
        fmt = ACTIVATION_FORMATS[row["output_activation_format"]]
        record = _record_bytes_by_hand(row["shape_out"], fmt)
        count = int(row["count_per_step"])
        if row.get("shape_outs"):
            assert row["act_out_bytes_per_rank"] > record * count, (
                row["aggregation_id"])
            continue
        assert row["act_out_bytes_per_rank"] == record * count, (
            row["aggregation_id"], row["act_out_bytes_per_rank"],
            record * count)


def test_a_cast_behind_views_joins_the_launch_that_writes_its_tensor():
    """DeepSeek-V4's attention output reaches `wo_a`'s FP8 cast through two
    view components (`head_merge`, then `head_split` into the output-LoRA
    groups). A view moves no bytes and these keep every 128-element scale
    group whole, so `fused_inv_rope` — vLLM's `fused_inv_rope_fp8_quant` —
    takes the cast in across them (`fusion.ViewChains`): the cast is no
    launch of its own after fusion, and the un-rotation writes o in FP8.
    A cast behind views no plan can absorb (`wo_b`'s, after the `wo_a`
    GEMM) stays as the serving graph grafted it.

    The bytes are checked by a second route that shares nothing with the
    traffic derivation: o's size from the checkpoint config (heads × head
    dim per token), the element widths of bf16 and FP8, and the format's
    one scale per `group` elements of a row, counted by hand."""
    import json
    from pathlib import Path

    from backend.inference.pipeline import analyze

    result = analyze(
        resolve_snapshot(_FP8_MODEL), parallel=ParallelConfig(),
        phase="decode", framework_binding=get_binding("vllm"),
        batch_size=1, avg_input_tokens=128, avg_output_tokens=2,
        breakdown=True)

    def named(rows, suffix):
        return [r for r in rows if r["aggregation_id"].endswith(suffix)]

    pre, post = result.pre_per_op_timing, result.per_op_timing
    grafted = named(pre, "wo_a__cast")
    assert grafted
    assert named(post, "wo_a__cast") == []
    (fmt,) = {r["output_activation_format"] for r in grafted}

    inv = [r for r in post if r["op"] == "fused_inv_rope"]
    assert inv
    assert {r["output_activation_format"] for r in inv} == {fmt}
    snapshot = (Path(__file__).resolve().parents[3] / "backend" / "arch"
                / "snapshots" / "deepseek-ai__DeepSeek-V4-Flash.json")
    config = json.loads(snapshot.read_text())["config"]
    o = config["num_attention_heads"] * config["head_dim"]
    from backend.arch.activation_formats import ACTIVATION_FORMATS
    record_fmt = ACTIVATION_FORMATS[fmt]
    assert o % record_fmt.group == 0
    scale_bytes = {"F8_E8M0": 1, "F32": 4}[record_fmt.scale_dtype]
    record = o * 1 + (o // record_fmt.group) * scale_bytes
    # One decode token per layer: o read at bf16 and written once as the FP8
    # record (elements and scales) — the standalone cast's own bytes, where
    # the launch before it also wrote o at bf16 and the cast read it back.
    for row in inv:
        assert (int(row["access_bytes_per_rank"])
                == (o * 2 + record) * row["count_per_step"]), row
    for row in grafted:
        assert (int(row["access_bytes_per_rank"])
                == (o * 2 + record) * row["count_per_step"]), row
    # One launch per layer the cast ran in.
    assert (sum(r["count_per_step"] for r in inv)
            == sum(r["count_per_step"] for r in grafted))
    # `wo_a` reads the same FP8 tensor before and after.
    for before, after in zip(named(pre, ".wo_a"), named(post, ".wo_a"),
                             strict=True):
        assert before["act_in_bytes_per_rank"] == after["act_in_bytes_per_rank"]

    # `wo_b`'s cast, behind `head_merge` after a GEMM, is untouched.
    standing = named(post, "wo_b__cast")
    assert standing
    assert ([r["access_bytes_per_rank"] for r in standing]
            == [r["access_bytes_per_rank"] for r in named(pre, "wo_b__cast")])


def test_the_drawn_tree_puts_the_launch_where_the_program_does():
    """The launch that took `wo_a`'s cast in across `head_merge` /
    `head_split` is drawn inside the attention component its members share
    (their lowest common ancestor), as `FusedNode.component_path` states,
    and the view components it took whole leave no empty box behind."""
    from backend.arch.eval.fusion import materialize_tree

    serving = _serving()
    tree = materialize_tree(serving.pre_program, serving.program,
                            serving.rewrite)

    def walk(nodes):
        for node in nodes:
            yield node
            yield from walk(node.children or ())

    drawn = list(walk([root for _meta, roots in tree.blocks
                       for root in roots]))
    launches = [n for n in drawn if n.op == "fused_inv_rope"]
    assert launches
    for node in launches:
        assert node.parent is not None and node.parent.name == "attn"
    assert ({n.component_path for n in launches}
            == {n.component_path for n in serving.program.nodes()
                if n.op == "fused_inv_rope"})
    assert [n.display_path for n in drawn
            if n.is_component and not n.children] == []


# ── One quantised copy per tensor and format, moved above pure copies ───────
#
# The graft is a naive cast on every quantising GEMM's edge, canonicalised:
# casts of one tensor to one format are one (CSE), and a cast moves above an
# element copy that keeps the scale-group axis when it is all that reads it
# (`traffic.ElementCopy`, `materialize_cast._Pending.commute`). A MoE
# dispatch is such a copy: the routed experts' cast lands on its token
# input, once per token, and the dispatch and the exchange after it move the
# narrow record — the form vLLM and sglang serve (vLLM
# `prepare_finalize/no_dp_ep.py:81`, `naive_dp_ep.py:37` then the dispatch of
# `a1q` and its scales at `:181`; sglang `fused_moe_triton_kernels.py:
# 866-868`). Made on the routed rows, the cast ran over `topk` times the
# tokens (DeepSeek-V3: 8x).

_MOE_FP8_MODEL = "deepseek-ai/DeepSeek-V3"


def _moe_flows(parallel: ParallelConfig, batch: int,
               model: str = _MOE_FP8_MODEL):
    resolved = resolve_snapshot(model)
    analysis = evaluate_step(
        resolved, parallel, phase="prefill",
        batch=batch, input_tokens=64, output_tokens=1,
    ).analysis
    serving = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm"))
    facts = _facts(analysis, serving)
    return analysis, serving, facts, _flows_over_graft(analysis, serving,
                                                       facts)


def _dispatches(program):
    """Each MoE dispatch and its token input, read off the declarations."""
    from backend.arch.ops import OP_REGISTRY

    found = [(node, OP_REGISTRY[node.op].gather_source)
             for node in program.nodes()
             if (spec := OP_REGISTRY.get(node.op or "")) is not None
             and spec.is_dispatch_boundary
             and spec.gather_source is not None]
    assert found
    return found


@pytest.mark.parametrize("parallel, batch", [
    # The exchange stands between the dispatch and the experts: the cast
    # moves above it too (the case the earlier graft grafted nothing for).
    (ParallelConfig(ep_moe=8, dp_attention=8), 8),
    (ParallelConfig(tp_attention=8, tp_dense_ffn=8), 1),
    # Every rank holds every token and routes `topk / ep` of each to its own
    # experts: the dispatch writes fewer rows than it holds.
    (ParallelConfig(ep_moe=16, tp_attention=16, tp_dense_ffn=16), 1),
])
def test_routed_experts_cast_once_per_token_before_the_dispatch(parallel,
                                                                batch):
    analysis, serving, facts, flows = _moe_flows(parallel, batch)
    pre = serving.pre_program
    for dispatch, token in _dispatches(pre):
        cast = dispatch.inputs_obj[token].src
        # The copy is made on the dispatch's token input…
        assert cast is not None and cast.provenance == "cast", (
            str(dispatch.display_path))
        assert any((s.consumer, s.index) == (dispatch.uid, token)
                   for s in serving.casts.sites if s.node is cast)
        # …over the tokens, once each — not over the routed rows.
        tokens = tuple(analysis.shapes.values[dispatch.uid].ins[token])
        assert facts.full_shapes.values[cast.uid].out == tokens
        narrow = _narrow(cast.output_format, tokens[-1])
        # The dispatch moves the narrow copy and its scales, in and out.
        moved = flows.values[dispatch.uid]
        assert moved.in_edges[token].width == narrow
        assert moved.out_terms[0].width == narrow
        # Every reader of the routed rows — directly, or through the
        # exchange grafted after the dispatch — takes the copy as it
        # arrives: no conversion of its own, and it reads it narrow.
        hops = {dispatch.uid} | {
            node.uid for node in pre.nodes()
            if node.provenance == "comm"
            and node.inputs_obj[0].src is dispatch}
        readers = [uid for uid, traffic in flows.values.items()
                   if traffic.in_edges
                   and traffic.in_edges[0].producer in hops
                   and uid not in hops]
        assert readers, str(dispatch.display_path)
        for uid in readers:
            traffic = flows.values[uid]
            assert same_law(traffic.convert_bytes, 0), (
                str(pre.by_uid(uid).display_path))
            assert traffic.cast is None
            assert traffic.in_edges[0].width == narrow
        # And nothing casts the routed rows themselves.
        assert not any(node.provenance == "cast"
                       and node.inputs_obj[0].src is not None
                       and node.inputs_obj[0].src.uid in hops
                       for node in pre.nodes())


@pytest.mark.parametrize("model_id", [
    "deepseek-ai/DeepSeek-V3",          # shared + routed experts, one copy
    "deepseek-ai/DeepSeek-V4-Flash",    # the attention's and indexer's wq_b
])
def test_one_quantised_copy_per_tensor_and_format(model_id):
    """No two casts copy one tensor into one format: every reader that wants
    it in that format reads the one copy (CSE). DeepSeek-V3's post-norm row
    is quantised once for the shared expert and the routed experts alike
    — a framework that quantises it twice runs a launch the graph does not
    model — and DeepSeek-V4's `q_norm` once for both `wq_b`s."""
    serving = _serving(model_id)
    pre = serving.pre_program
    seen: Dict[Any, str] = {}
    for node in pre.nodes():
        if node.provenance != "cast" or node.output_format is None:
            continue
        ref = node.inputs_obj[0]
        key = (ref.src.uid, ref.port or "", node.output_format.name)
        assert key not in seen, (str(node.display_path), seen.get(key))
        seen[key] = str(node.display_path)
    # The MoE layer's copy is read by its dispatch and its shared expert.
    for dispatch, token in _dispatches(pre):
        cast = dispatch.inputs_obj[token].src
        readers = {str(node.display_path) for node in pre.nodes()
                   if not node.children and any(
                       ref.src is cast for ref in (node.inputs_obj or ()))}
        assert any("shared_experts" in path for path in readers), readers


def test_a_shared_copy_runs_once_per_tensor_it_copies():
    """A copy several readers take is made once per step for each time its
    writer runs, however often each reader runs. DeepSeek-V2-Lite's MoE
    layers quantise the post-norm once for the dispatch and both shared
    experts (two instances each per layer): the copy runs once per MoE
    layer, counted here off the config rather than off the graph."""
    import json
    from pathlib import Path

    from backend.arch.precision_profile import PrecisionProfile
    from backend.inference.framework_bindings import get_binding as binding
    from backend.inference.pipeline import analyze

    stem = "deepseek-ai__DeepSeek-V2-Lite"
    config = json.loads((Path(__file__).resolve().parents[3] / "backend"
                         / "arch" / "snapshots" / f"{stem}.json"
                         ).read_text())["config"]
    moe_layers = (config["num_hidden_layers"]
                  - config["first_k_dense_replace"])
    result = analyze(
        resolve_snapshot(stem), parallel=ParallelConfig(),
        framework_binding=binding("vllm"), batch_size=8,
        avg_input_tokens=64, avg_output_tokens=1, phase="decode",
        precision=PrecisionProfile.from_request({"preset": "w_fp8_kv_fp8"}),
        breakdown=True)
    shared = [row for row in result.per_op_timing
              if row.get("op") == "quantise_activation"
              and "moe_ffn" in str(row["aggregation_id"])
              and str(row["aggregation_id"]).endswith("|gate_proj__cast")]
    assert len(shared) == 1, [row["aggregation_id"] for row in shared]
    assert shared[0]["count_per_step"] == moe_layers == 26


def test_the_exchange_after_the_dispatch_moves_the_narrow_copy():
    """DP-attention puts an all-to-all between the dispatch and the experts.
    It moves the tensor it reads — the dispatch's rows of the quantised
    tokens — so its wire payload is the routed rows at the narrow width and
    its scale share, the representation the dispatch writes in HBM; the
    payload is restated over the grafted graph (`exchange_widths`) by the
    rule it was chosen by at graft (`traffic.payload_width`). The exchange
    back (`combine`'s) moves the experts' BF16 output."""
    from backend.arch.layout.materialize_comm import payload_bytes

    _, serving, _, flows = _moe_flows(
        ParallelConfig(ep_moe=8, dp_attention=8), batch=8)
    pre = serving.pre_program
    materialized = serving.materialized
    events = {materialized.node_uid_by_event[event.event_id]: event
              for event in materialized.events}
    checked = 0
    for dispatch, token in _dispatches(pre):
        cast = dispatch.inputs_obj[token].src
        for node in pre.nodes():
            if node.provenance != "comm" or node.uid not in events:
                continue
            if node.inputs_obj[0].src is not dispatch:
                continue
            event = events[node.uid]
            narrow = _narrow(cast.output_format,
                             value_of(event.payload_shape[-1]))
            assert event.payload_dtype_bytes == narrow
            assert 1.0 < narrow < 2.0
            assert value_of(event.payload_bytes) == value_of(
                payload_bytes(event.payload_shape, narrow))
            meta = materialized.comm_meta.values[node.uid]
            assert meta["payload_dtype_bytes"] == narrow
            # The wire and HBM sides of the one exchange agree.
            assert flows.values[node.uid].in_edges[0].width == narrow
            checked += 1
    assert checked
    backs = [combine.inputs_obj[0].src for combine in pre.nodes()
             if combine.op == "combine"]
    backs = [back for back in backs
             if back is not None and back.uid in events]
    assert backs
    for back in backs:
        assert events[back.uid].payload_dtype_bytes == 2.0


def test_the_priced_tables_carry_the_narrow_exchange():
    """End to end (`pipeline.analyze`) under DP-attention: the exchange
    after the dispatch is a row whose payload is the narrow copy with its
    scales."""
    from backend.arch.activation_formats import ACTIVATION_FORMATS
    from backend.inference.pipeline import analyze

    result = analyze(
        resolve_snapshot(_MOE_FP8_MODEL),
        parallel=ParallelConfig(ep_moe=8, dp_attention=8),
        framework_binding=get_binding("vllm"), phase="decode",
        batch_size=8, avg_input_tokens=128, avg_output_tokens=2,
        breakdown=True)
    rows = [r for r in result.per_op_timing
            if r["op"] == "all_to_all"
            and str(r["aggregation_id"]).endswith("__on__experts")]
    assert rows
    narrow = _narrow(ACTIVATION_FORMATS["fp8_token_group128"], 7168)
    for row in rows:
        assert row["comm_payload_dtype_bytes"] == narrow, (
            row["aggregation_id"])


def test_which_copies_a_cast_moves_above_is_read_off_declarations():
    """`ElementCopy` on real nodes: the dispatch copies whole token rows and
    the exchange after it moves them (both commute), the attention's head
    merge regroups the innermost axis (a copy that does not keep rows), a
    norm computes, and a reducing collective sums (no copy at all)."""
    from backend.arch.eval.traffic import ElementCopy
    from backend.arch.eval.layer import Layer

    for parallel, batch in ((ParallelConfig(ep_moe=8, dp_attention=8), 8),
                            (ParallelConfig(tp_attention=8,
                                            tp_dense_ffn=8), 1)):
        analysis, serving = _moe_flows(parallel, batch)[:2]
        pre = serving.pre_program
        shapes = Layer(pre, {**analysis.shapes.values,
                             **serving.materialized.shapes.values})
        kinds: Dict[str, Any] = {}
        for node in pre.nodes():
            copy = ElementCopy.of(node, shapes)
            if node.op == "permute":
                kinds["dispatch"] = copy
            elif node.op == "all_to_all":
                kinds.setdefault("exchange", copy)
            elif node.op == "all_reduce":
                kinds["reduce"] = copy
            elif node.op == "rmsnorm":
                kinds["norm"] = copy
            elif node.op == "view_merge":
                kinds["merge"] = copy
        assert kinds["dispatch"].commutes
        assert kinds["norm"] is None
        assert kinds["merge"] is not None
        assert not kinds["merge"].keeps_rows
        if "exchange" in kinds:
            assert kinds["exchange"].commutes
        if "reduce" in kinds:
            assert kinds["reduce"] is None


def _fake_tree(published=()):
    """A component `wrapper` holding a view `x_view` and a GEMM `gemm` that
    reads it, a second GEMM `other` outside the wrapper, and a program over
    them answering `nodes()` and `output_leaf()`."""
    from types import SimpleNamespace

    from backend.arch.spec_tree.nodes import InputRef

    def node(uid, name, parent=None, inputs=(), children=()):
        made = SimpleNamespace(uid=uid, name=name, parent=parent,
                               inputs_obj=tuple(inputs),
                               children=list(children),
                               secondary_outputs={}, template="t",
                               display_path=name)
        return made

    source = node(1, "norm")
    wrapper = node(2, "mlp")
    view = node(3, "x_view", parent=wrapper,
                inputs=(InputRef(src=source, port=""),))
    gemm = node(4, "gemm", parent=wrapper,
                inputs=(InputRef(src=view, port=""),))
    wrapper.children = [view, gemm]
    wrapper.inputs_obj = (InputRef(src=source, port=""),)
    leaf_out = {wrapper.uid: published[0] if published else gemm}
    program = SimpleNamespace(
        nodes=lambda: [source, wrapper, view, gemm],
        output_leaf=lambda comp, port="out": (leaf_out[comp.uid], ""))
    return program, source, wrapper, view, gemm, node


def test_a_cast_moves_above_a_copy_only_when_it_reads_all_of_it():
    """The commute's reader condition (`traffic.TensorReaders.only`, which
    fusion's view chains ask too) counts every reader of the copy's output,
    not only leaf edges: a second leaf reader, a component entry whose body
    holds none of the cast's readers, a component exit (the copy is what
    the component publishes), and a tensor read past the graph's edges —
    cached, or a stream a later layer reads (`pinned`) — each keep the cast
    where it is."""
    from backend.arch.eval.traffic import TensorReaders
    from backend.arch.spec_tree.nodes import InputRef

    def only(program, view, pinned=frozenset()):
        return TensorReaders(program, pinned_uids=frozenset(pinned)).only(
            view, "", [(gemm, 0)])

    program, source, wrapper, view, gemm, node = _fake_tree()
    # The cast is all that reads the view: it may move.
    assert only(program, view)
    # Cached, or written as a stream a later layer reads.
    assert not only(program, view, {view.uid})
    # A second leaf reads the view wide.
    other = node(5, "other", inputs=(InputRef(src=view, port=""),))
    program.nodes = lambda: [source, wrapper, view, gemm, other]
    assert not only(program, view)
    # A component entry lists the view, and none of the cast's readers is
    # inside it.
    outer = node(6, "outer", inputs=(InputRef(src=view, port=""),))
    program.nodes = lambda: [source, wrapper, view, gemm, outer]
    assert not only(program, view)
    outer.children = [node(7, "inner")]
    assert not only(program, view)
    # The view is what its component publishes.
    program, source, wrapper, view, gemm, node = _fake_tree()
    program.output_leaf = lambda comp, port="out": (view, "")
    assert not only(program, view)
    # A component reader hands the tensor to its body: the body's edge is
    # part of its reader's region, not a second reader.
    program, source, wrapper, view, gemm, node = _fake_tree()
    body = node(9, "body", inputs=(InputRef(src=view, port=""),))
    entry = node(8, "entry", inputs=(InputRef(src=view, port=""),),
                 children=[body])
    body.parent = entry
    program.nodes = lambda: [source, wrapper, view, entry, body]
    assert TensorReaders(program, pinned_uids=frozenset()).only(view, "", [(entry, 0)])


@pytest.mark.parametrize("model_id, phase, parallel, batch", [
    # MLA's prefill form reads its cached latent through a KV marker view
    # straight into `kv_b_proj`, a quantised GEMM: the case a cast would
    # wrongly move above.
    ("deepseek-ai/DeepSeek-V3", "prefill",
     ParallelConfig(ep_moe=8, dp_attention=8), 8),
    ("deepseek-ai/DeepSeek-V4-Flash", "decode", ParallelConfig(), 1),
])
def test_no_cast_moves_above_a_copy_read_past_the_graph_edges(
        model_id, phase, parallel, batch):
    """On a real model: no element copy whose output is read past the
    graph's edges reads a cast — none was moved above it. Those tensors are
    named here by a second route, off the declarations directly rather than
    the graft's pinned set: the cache's declared tensors
    (`cache.cached_node_uids`), the streams the tree says its layers write
    (`SpecIRTree.stream_writes`), each block template's declared output
    roots, the model-level nodes, and every view that marks a KV stream
    (`OpSpec.kv_stream_marker`: it reads the cache's positions)."""
    from backend.arch.eval.layer import Layer
    from backend.arch.eval.traffic import ElementCopy
    from backend.arch.ops import OP_REGISTRY

    analysis = evaluate_step(
        resolve_snapshot(model_id), parallel, phase=phase,
        batch=batch, input_tokens=64, output_tokens=1).analysis
    serving = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm"))
    pre = serving.pre_program
    read_past = (set(cache_mod.cached_node_uids(pre))
                 | set(pre.ir.stream_writes.values())
                 | {root.uid for template in pre.blocks
                    for root in template.output_roots()}
                 | {node.uid for node in pre.ir.top_level}
                 | {node.uid for node in pre.nodes()
                    if node.op in OP_REGISTRY
                    and OP_REGISTRY[node.op].kv_stream_marker})
    shapes = Layer(pre, {**analysis.shapes.values,
                         **serving.materialized.shapes.values})
    markers = 0
    for uid in read_past:
        node = pre.by_uid(uid)
        if node is None or ElementCopy.of(node, shapes) is None:
            continue
        markers += bool(node.op in OP_REGISTRY
                        and OP_REGISTRY[node.op].kv_stream_marker)
        for ref in node.inputs_obj or ():
            assert ref.src is None or ref.src.provenance != "cast", (
                str(node.display_path))
    # The copies a cast would wrongly move above are here: a KV marker view
    # an element copy by its declaration, read as the cache's positions.
    assert markers


def test_a_context_gather_keeps_the_wide_tensor_it_publishes():
    """A CP all-gather carries a cached tensor's context, so a later cast
    cannot commute above it: the collective still publishes the wide
    tensor, and the cache can store what it gathered. This graph has the
    cast — the assertion that a collective exists makes the absence of a
    cast input a regression, not an empty pass."""
    analysis = evaluate_step(
        resolve_snapshot("deepseek-ai/DeepSeek-V3"), ParallelConfig(cp=2),
        phase="prefill", batch=1, input_tokens=256,
        output_tokens=1).analysis
    serving = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm"))
    from backend.arch.eval.traffic import TensorReaders

    pre = serving.pre_program
    readers = TensorReaders.of(analysis, pre)
    gathers = [node for node in pre.nodes()
               if node.provenance == "comm" and node.op == "all_gather"]
    assert gathers
    assert all(node.uid in readers.pinned for node in gathers)
    for node in pre.nodes():
        if node.provenance != "comm" or not node.inputs_obj:
            continue
        for ref in node.inputs_obj:
            assert ref.src is None or ref.src.provenance != "cast", (
                str(node.display_path))


def test_a_lookup_gathers_its_ids_without_casting_them():
    """A table's storage format says what the table's rows are, not what a
    kernel gathers them by: DeepSeek-V4.1's FP8 Engram table still takes
    I32 token ids. Only a launch that multiplies its operand eats it in
    the format the deployment declares."""
    analysis = evaluate_step(
        resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash"),
        ParallelConfig(), phase="prefill", batch=1, input_tokens=64,
        output_tokens=1).analysis
    serving = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm"))
    for site in serving.casts.sites:
        ref = site.node.inputs_obj[0]
        assert ref.src is not None or ref.external_sid != "input_ids", (
            str(site.node.display_path))


def test_a_quantised_record_is_handed_on_only_whole():
    """An element copy writes what it reads (`traffic._copied`): a
    collective moving a cast's copy holds that record, its width read over
    the collective's own tensor; one that regroups the innermost axis per
    rank (its scale groups are not the copy's) holds no one record and is
    refused, never priced at a width no tensor has."""
    from types import SimpleNamespace

    from backend.arch.activation_formats import ACTIVATION_FORMATS
    from backend.arch.eval.values import Shape
    from backend.arch.spec_tree.nodes import InputRef

    fmt = ACTIVATION_FORMATS["fp8_token_group128"]

    def node(uid, **fields):
        base = dict(uid=uid, op=None, provenance="spec", children=[],
                    inputs_obj=(), output_format=None, output_dtype=None,
                    display_path=f"n{uid}", name=f"n{uid}")
        base.update(fields)
        return SimpleNamespace(**base)

    cast = node(1, op="quantise_activation", provenance="cast",
                output_format=fmt)
    comm = node(2, op="all_gather", provenance="comm",
                inputs_obj=(InputRef(src=cast, port=""),))
    program = SimpleNamespace(by_uid={1: cast, 2: comm}.get)
    for gathered, whole in (((64, 512), (256, 512)), ((64, 72), (64, 576))):
        shapes = SimpleNamespace(values={
            1: Shape(ins=(gathered,), out=gathered),
            2: Shape(ins=(gathered,), out=whole)})
        if gathered[-1] == whole[-1]:
            held = traffic_layer.declared_output(comm, program,
                                                 shapes=shapes)
            assert held is not None and held.format is fmt
            assert held.width == _narrow(fmt, whole[-1])
        else:
            with pytest.raises(ValueError, match="without keeping"):
                traffic_layer.declared_output(comm, program, shapes=shapes)


@pytest.mark.parametrize("model_id", [
    "deepseek-ai/DeepSeek-V3",          # the MoE block's shared-expert cast
    "deepseek-ai/DeepSeek-V4-Flash",    # and the attention block's `wq_a`
])
def test_every_component_a_cast_enters_lists_it(model_id):
    """Every component the edge from a grafted cast to one of its consumers
    enters lists the cast among its inputs: a drawn graph and the
    structure view then have an edge into the component for it, and no
    cast is left without a reader inside its own scope. A component whose
    body reads both the wide tensor and the copy lists both, one formal
    each; the cast's inputs stay one to one with its formals."""
    serving = _serving(model_id)
    pre = serving.pre_program
    unlisted = set()
    for site in serving.casts.sites:
        cast = site.node
        crossed = pre.by_uid(site.consumer).parent
        while crossed is not None and crossed is not cast.parent:
            if not any(ref.src is not None and ref.src.uid == cast.uid
                       for ref in (crossed.inputs_obj or ())):
                unlisted.add((str(cast.display_path),
                              str(crossed.display_path)))
            assert len(crossed.formals) == len(crossed.inputs_obj), (
                str(crossed.display_path))
            crossed = crossed.parent
    assert unlisted == set(), sorted(unlisted)


def _inside(node, wrapper) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is wrapper:
            return True
        parent = parent.parent
    return False


# -- a cache format's stored transform ---------------------------------------
#
# GLM-5.3-Flash ships its k-pool index keys as both serving stacks store them
# (`quant_context.cache_formats.index_key: fp8_pow2_hadamard128`): FP8, one
# pow2 scale per 128 channels, Hadamard-128 rotated before the write. The
# rotation is a launch of `hadamard_transform`, grafted on the edge into the
# cache and on the query the indexer score multiplies the cached key with.

_TRANSFORM_MODEL = "zai-org/GLM-5.3-Flash"
_ROTATION = "hadamard_transform"


def _deployed(framework: str):
    """The binding as an H100 deployment answers it: the `device: cuda`
    fusions are in."""
    from backend.inference.framework_bindings import ServingDeployment
    return get_binding(framework).deployed(ServingDeployment.at(
        gpu_sku="H100-SXM", activation_dtype="BF16"))


def _rotations(serving):
    return [site for site in serving.casts.sites
            if site.node.op == _ROTATION]


def test_the_shipped_index_key_is_the_rotated_fp8_record():
    analysis = _analysis(_TRANSFORM_MODEL)
    index = [t for entry in cache_mod.derive(analysis).values.values()
             for t in entry.tensors if t.kind == "index_key"]
    assert index
    for tensor in index:
        assert tensor.format_name == "fp8_pow2_hadamard128"
        assert tensor.resolved_format.transform_op == _ROTATION
        # 128 FP8 values and one F32 scale a pooled key.
        assert same_law(tensor.position_bytes, 132)


def test_one_rotation_on_the_key_write_and_one_on_the_query():
    serving = rewrite_serving_graph(_analysis(_TRANSFORM_MODEL),
                                    framework_binding=get_binding("sglang"))
    pre = serving.pre_program
    rotations = _rotations(serving)
    # Per DSA layer template (the backbone's and the MTP head's): one on
    # the edge into the store marker of the cached pooled key, one on the
    # score's query operand.
    by_template: Dict[Any, list] = {}
    for site in rotations:
        consumer = pre.by_uid(site.consumer)
        by_template.setdefault(consumer.template, []).append(
            (consumer.op, site.index))
        node = site.node
        assert node.provenance == "cast"
        assert node.output_format is None and node.output_dtype is None
        # The consumer reads the rotation, or the quantised copy of it the
        # score's FP8 query is (`index_key.paired_operand`): rotated, then
        # quantised, as both stacks run it.
        read = consumer.inputs_obj[site.index].src
        if read is not node:
            assert read.op == "quantise_activation", read.display_path
            assert read.inputs_obj[0].src is node
    assert len(by_template) == 2, by_template
    for got in by_template.values():
        assert sorted(got) == [("attention_score", 0),
                               ("token_pool_compressor", 0)], got
    # The key side rotates what the pool computed, the query side what the
    # query projection's head split handed the score.
    for site in rotations:
        producer = site.node.inputs_obj[0].src
        assert producer.op in ("sum_reduce", "head_split"), (
            producer.display_path)


def test_the_rotations_facts_are_their_consumers_operands():
    analysis = _analysis(_TRANSFORM_MODEL)
    serving = rewrite_serving_graph(analysis,
                                    framework_binding=get_binding("sglang"))
    facts = _facts(analysis, serving)
    for site in _rotations(serving):
        operand = analysis.shapes.values[site.consumer].ins[site.index]
        assert tuple(facts.full_shapes.values[site.node.uid].out) == tuple(
            operand)
        # n · (log2 n + 1) per row: the fast transform's butterflies and
        # the scale, from the op's own declaration.
        rows = product(operand[:-1])
        n = operand[-1]
        assert same_law(facts.full_cost.values[site.node.uid].flops,
                        rows * n * 8) and n == 128


@pytest.mark.parametrize("framework, query_launch", [
    ("vllm", "fused_fwht128_quant"),
    ("sglang", None),
])
def test_each_framework_absorbs_the_rotations_it_runs_fused(framework,
                                                           query_launch):
    serving = rewrite_serving_graph(_analysis(_TRANSFORM_MODEL),
                                    framework_binding=_deployed(framework))
    origins = serving.rewrite.origins or {}
    launch_of = {member: serving.program.by_uid(fused).op
                 for fused, members in origins.items()
                 for member in members}
    pre = serving.pre_program
    for site in _rotations(serving):
        consumer = pre.by_uid(site.consumer)
        got = launch_of.get(site.node.uid)
        if consumer.op == "token_pool_compressor":
            # Both stacks rotate inside the pool+quant+store launch.
            assert got == "fused_kpool_compress_store", consumer.display_path
        else:
            # vLLM rotates and quantises the query in one launch; sglang
            # runs `rotate_activation` on its own.
            assert got == query_launch, (framework, consumer.display_path)


@pytest.mark.parametrize("model", [
    "deepseek-ai/DeepSeek-V4-Flash",      # index key MXFP4, no transform
    _BF16_MODEL,
])
def test_a_model_without_a_transform_format_grafts_no_rotation(model):
    serving = rewrite_serving_graph(_analysis(model),
                                    framework_binding=get_binding("vllm"))
    assert not _rotations(serving)


def test_a_transform_its_readers_cannot_pair_is_refused():
    """A rotated K/V cache: the score pairs the cached key with the query,
    but the value is read by the probability-weighted sum, which contracts
    the positions, not the rotated channels — no basis makes it right."""
    from backend.arch.precision_profile import PrecisionError, PrecisionProfile
    profile = PrecisionProfile.canonicalize({"cache": [
        {"select": {"slots": ["kv"]}, "format": "fp8_pow2_hadamard128"}]})
    analysis = evaluate_step(
        resolve_snapshot(_BF16_MODEL), ParallelConfig(), phase="decode",
        batch=1, input_tokens=128, output_tokens=1,
        precision=profile).analysis
    with pytest.raises(PrecisionError, match=r"rotated basis|marking it"):
        rewrite_serving_graph(analysis, framework_binding=get_binding("vllm"))
