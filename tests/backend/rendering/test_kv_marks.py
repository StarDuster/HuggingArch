"""KV cache marks on the rendered graphs, against the cache layer.

The inference page's serving graph is checked over every snapshot, both
phases, and three meshes: the writers' stored bytes per request, counted per
layer, must add up to the per-rank stored cache the capacity page budgets
with (`CacheView.storage_bytes_at` — each tensor at its own law, so a window
stops at its window and a compressed pool grows at its ratio), and every
node the cache layer names must be drawn. The Inspector graph is checked on
the layers worked through by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.formats import known_dtype_bytes
from backend.arch.layout.physical_cache import (
    cache_geometry, physical_cache_view)
from backend.arch.snapshots import iter_snapshot_ids
from backend.inference.framework_bindings import get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.rendering._compute_graph import serialize_post_tree
from backend.rendering.builders import build_block_graph_data
from backend.rendering.kv_marks import KVMarks
from tests.snapshots import resolve_snapshot, shared_loader

SNAPSHOTS = sorted(iter_snapshot_ids())
PHASES = ("prefill", "decode")
#: DCP is the mesh the engine may refuse for a layer (a head shard it
#: crosses, sparse top-k); TP never.
MESHES = {
    "tp1": {},
    "tp8": {"tp_attention": 8, "tp_dense_ffn": 8},
    "tp4_dcp2": {"tp_attention": 4, "tp_dense_ffn": 4, "dcp": 2},
}


def _serving(model_id: str, phase: str, mesh: str):
    resolved = resolve_snapshot(model_id, shared_loader())
    try:
        result = analyze(
            resolved, parallel=ParallelConfig.from_dict(MESHES[mesh]),
            framework_binding=get_binding("vllm"), batch_size=1, avg_input_tokens=16, avg_output_tokens=0,
            phase=phase, breakdown=False)
    except (ValueError, NotImplementedError) as exc:
        if "dcp" not in MESHES[mesh]:
            raise
        pytest.skip(f"the engine refuses this mesh: {exc}")
    graph = serialize_post_tree(
        result.post_tree, resolved, pre_tree=result.comm_tree,
        fusion_matches=result.fusion_matches, program=result.program,
        shapes=result.shapes, full_shapes=result.full_shapes,
        pre_program=result.pre_program, pre_shapes=result.pre_shapes,
        pre_full_shapes=result.pre_full_shapes,
        kv=KVMarks.of_serving(result))
    return result, graph


def _walk(nodes):
    for node in nodes:
        yield node
        yield from _walk(node.get("children") or ())


def _derived_edges(nodes):
    """The edges the old inference renderer added from `inputs[]`, per
    scope — the rule `deriveEdgesFromInputs` applied."""
    by_id = {n["id"] for n in nodes}
    out = set()
    for n in nodes:
        for ref in n.get("inputs") or ():
            parent = ref if ref in by_id else ref.rsplit(".", 1)[0]
            if parent in by_id:
                out.add((parent, n["id"]))
    return out


def _edge_scopes(nodes, edges):
    """(backend edges, derived edges) per scope, over the whole tree."""
    yield {(e["src"], e["dst"]) for e in edges}, _derived_edges(nodes)
    for node in nodes:
        if node.get("children"):
            yield from _edge_scopes(node["children"],
                                    node.get("child_edges") or [])


@pytest.mark.regression  # every snapshot × phase × mesh
@pytest.mark.parametrize("model_id", SNAPSHOTS)
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("mesh", sorted(MESHES))
def test_serving_graph_marks_add_up_to_the_cache(model_id, phase, mesh):
    result, graph = _serving(model_id, phase, mesh)

    # G1: backbone writers × layers = the per-rank stored cache, each
    # tensor at its own storage law at the graph's cache length.
    stored = aux = 0.0
    cache_lens = set()
    for node in graph["nodes"]:
        if node.get("op") != "block_stack" or not node.get("kv"):
            continue
        count = int(node.get("count") or 1)
        for entry in node["kv"]["writes"]:
            if node.get("aux_kind"):
                aux += (entry["bytes_per_request"] if entry["recurrent"]
                        else entry["bytes_per_position"]) * count
                continue
            stored += entry["bytes_per_request"] * count
            cache_lens.add(entry["cache_len"])
    if cache_lens:
        (cache_len,) = cache_lens
        assert stored == pytest.approx(physical_cache_view(
            result.analysis).storage_bytes_at(cache_len))

    # An auxiliary stack (MTP) keeps its own cache, outside the backbone
    # budget: its writers add up to the cache layer's entries for it.
    owner = {}
    for contribution in result.analysis.program.contributions:
        for node in _walk_engine(contribution.node):
            owner[node.uid] = (contribution.count, contribution.stack)
    want_aux = 0.0
    for ref, (_holder, _entry, tensor) in cache_mod.resident(
            result.analysis, geometry=cache_geometry(result.analysis)).items():
        count, stack = owner.get(ref.component, (1, ""))
        if stack:
            # The cache layer's own law for a stored position, which is what
            # the mark publishes: the data AND the dynamic scales riding the
            # position. `elements × element_bytes` is only the data half
            # (`CachedTensor.data_position_bytes`), so a per-group fp8 entry
            # — DeepSeek-V3.2's MTP latent — would be short its scales.
            per = tensor.position_bytes / tensor.positions_over
            want_aux += float(per) * count
    assert aux == pytest.approx(want_aux)

    # G4: every writer and reader the cache layer names is drawn.
    drawn = {n.get("uid") for n in _walk(graph["nodes"])}
    named = {uid for uid, value in result.pre_kv_access.values.items()
             if value.writes or value.reads}
    assert named <= drawn

    # G8 (payload): the edges the old renderer derived from `inputs[]` are
    # all backend edges, so drawing only the backend's loses none.
    for backend, derived in _edge_scopes(graph["nodes"], graph["edges"]):
        assert derived <= backend


def _walk_engine(node):
    yield node
    for child in node.children or ():
        yield from _walk_engine(child)


def _find(nodes, predicate):
    return [n for n in _walk(nodes) if predicate(n)]


def _edges(nodes, edges):
    for scope_edges in [edges] + [n.get("child_edges") or []
                                  for n in _walk(nodes)]:
        yield from scope_edges


def test_serving_graph_qwen3_decode():
    """Qwen3-8B decode: the fused qk-rope kernel writes K and publishes it
    on one port only; V is written by its head split; the fused attention
    reads both; the block shows the layer's 4 KiB per token."""
    _result, graph = _serving("Qwen/Qwen3-8B", "decode", "tp1")
    rope = _find(graph["nodes"], lambda n: n.get("op") == "fused_qk_rope")[0]
    assert [e["tensor"] for e in rope["kv"]["writes"]] == ["k_rope"]
    assert rope["kv"]["write_bytes_per_token"] == 2048
    attn = _find(graph["nodes"], lambda n: n.get("op") == "flash_attention")[0]
    assert sorted(e["tensor"] for e in attn["kv"]["reads"]) == [
        "k_rope", "v_split_heads"]
    assert attn["kv"]["read_bytes_per_token"] == 4096
    assert attn["kv"]["read_terms"] == [{"law": "s_kv", "bytes_per_token": 4096}]
    block = _find(graph["nodes"], lambda n: n.get("op") == "block_stack")[0]
    assert block["kv"]["write_bytes_per_token"] == 4096
    # Q and K leave the fused rope kernel on two ports; only K's edge is a
    # cache edge.
    rope_edges = [e for e in _edges(graph["nodes"], graph["edges"])
                  if e["src"] == rope["id"]]
    assert len(rope_edges) == 2
    assert sorted(bool(e.get("kind")) for e in rope_edges) == [False, True]
    kinds = {e.get("kind") for e in _edges(graph["nodes"], graph["edges"])}
    assert "cache" in kinds


def test_serving_graph_states_are_on_their_carriers():
    """Qwen3-Coder-Next: the gated delta rule reads and writes the recurrent
    matrix, the causal conv its window, and neither has an edge. The two
    badges sum to the layer's whole state, 32·128·128 + 8192·4 elements (the
    conv keeps its kernel_size = 4 columns, as the HF cache does), each
    at its own width: the recurrent matrix at the dtype its cache entry
    declares (fp32, the reference model's), the conv window at the
    deployment's."""
    _result, graph = _serving("Qwen/Qwen3-Coder-Next", "decode", "tp1")
    carriers = [n for n in _find(graph["nodes"], lambda n: n.get("kv"))
                if not n.get("children")
                and any(e["recurrent"] for e in n["kv"]["writes"])]
    by_op = {n["op"]: n["kv"] for n in carriers}
    assert set(by_op) == {"gated_delta_rule", "conv1d"}
    assert [e["tensor"] for e in by_op["gated_delta_rule"]["reads"]] == [
        "recurrent_state"]
    assert [e["tensor"] for e in by_op["conv1d"]["reads"]] == ["conv_state"]
    resolved = resolve_snapshot("Qwen/Qwen3-Coder-Next", shared_loader())
    gdn = resolved.spec.components.get("gated_delta_net_fused_attention") \
        or resolved.spec.get_block("gated_delta_net_fused_attention")
    [state] = [s for s in gdn.kv_states if s.carrier == "delta_rule"]
    state_width = known_dtype_bytes(state.dtype)
    assert state_width == 4
    assert by_op["gated_delta_rule"]["state_bytes_per_request"] == (
        32 * 128 * 128 * state_width)
    assert by_op["conv1d"]["state_bytes_per_request"] == 8192 * 4 * 2
    assert sum(kv["state_bytes_per_request"] for kv in by_op.values()) == (
        32 * 128 * 128 * state_width + 8192 * 4 * 2)


def test_inspector_graph_marks_the_writers_readers_and_cache_edges():
    """The Inspector's raw graph carries the same marks, and its cache edges
    run from the writers to the readers."""
    resolved = resolve_snapshot("Qwen/Qwen3-8B", shared_loader())
    blocks, _ = build_block_graph_data(resolved, 1, 1024)
    nodes = next(iter(blocks.values()))["nodes"]
    edges = next(iter(blocks.values()))["edges"]
    by_tensor = {}
    for node in _walk(nodes):
        for entry in (node.get("kv") or {}).get("writes", ()):
            if not node.get("children") or node["op"] == "head_split":
                by_tensor.setdefault(entry["tensor"], set()).add(node["op"])
    assert by_tensor["k_rope"] >= {"rope"}
    score = _find(nodes, lambda n: n.get("op") == "attention_score")[0]
    read = score["kv"]["reads"][0]
    assert read["tensor"] == "k_rope"
    assert read["bytes_per_position"] == 2048
    assert read["bytes_per_request"] == 2048 * 1024
    assert read["cache_len"] == 1024
    cache_edges = [e for e in _edges(nodes, edges) if e.get("kind") == "cache"]
    assert {e["dst"] for e in cache_edges} >= {score["id"]}


def test_a_mask_output_is_not_relabelled():
    """DSA's top-k mask stays a mask edge beside the cache edges."""
    _result, graph = _serving("deepseek-ai/DeepSeek-V3.2", "decode", "tp1")
    kinds = {e.get("kind") for e in _edges(graph["nodes"], graph["edges"])}
    assert {"cache", "mask"} <= kinds


def test_an_encode_walk_has_an_empty_catalogue():
    """An encode walk runs no language step: it keeps no cache and binds no
    `s_kv`, so its marks catalogue nothing rather than asking for a cache
    length the walk never had."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    program = Program.of(resolve_snapshot("stepfun-ai/Step-3.7-Flash"))
    (encode,) = Deployment(program).at(batch=1, media={
        "pixel_values": {"n_img": 1},
        "patch_pixel_values": {"n_crops": 2}}).encoders()
    analysis = encode.analyze()
    assert "s_kv" not in analysis.binding
    assert KVMarks.of_analysis(analysis).catalogue == {}
