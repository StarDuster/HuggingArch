"""L1 structure view (backend.views.abstraction).

The view groups a component's L2 nodes into boxes by the operand each node
feeds, using only declared operand roles and the graph's edges. These tests
pin (1) how it groups and when it refuses, on small hand-built graphs; (2) the
architectural facts it must reproduce on real components; (3) that it stays a
view — nothing upstream imports it and it reads no component or node names.
"""
from __future__ import annotations

import ast
import glob
import pathlib

import pytest
import yaml

from backend.views.abstraction import Sums, abstract
from tests.backend.views import declared_roles
from backend.arch.eval.precision import AS_SHIPPED

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _n(nid, op, inputs=(), flops=0, params=0, **kw):
    return {"id": nid, "op": op, "inputs": list(inputs), "flops": flops,
            "params": params, "weight_bytes": 2 * params,
            **declared_roles(op), **kw}


def _attn(shared_kv=False):
    """x -> q -> score; x -> kv -> {k, v}; score -> softmax -> apply -> o."""
    nodes = [
        _n("x", "input"),
        _n("q_proj", "linear", ["x"], 10, 5, flops_law="10"),
        _n("kv_proj", "linear", ["x"], 4, 2, flops_law="4"),
        _n("k", "slice", ["kv_proj"]),
        _n("score", "attention_score", ["q_proj", "k" if not shared_kv else "kv_proj"], 3, flops_law="3"),
        _n("sm", "softmax", ["score"], 1, flops_law="1"),
        _n("v", "slice", ["kv_proj"]),
        _n("apply", "attention_apply", ["sm", "v" if not shared_kv else "kv_proj"], 3, flops_law="3"),
        _n("o_proj", "linear", ["apply"], 10, 5, flops_law="10"),
    ]
    tot = Sums(flops=sum(n["flops"] for n in nodes), params=sum(n["params"] for n in nodes),
               weight_bytes=sum(n["weight_bytes"] for n in nodes))
    return nodes, tot


def test_groups_attention_by_operand_role():
    nodes, tot = _attn()
    v = abstract(nodes, tot)
    assert v.available, v.reason
    assert v.node_box["q_proj"] == "q"
    assert v.node_box["kv_proj"] == "k+v"      # feeds both K and V
    assert v.node_box["k"] == "k" and v.node_box["v"] == "v"
    assert {v.node_box[x] for x in ("score", "sm", "apply")} == {"core"}
    assert v.node_box["o_proj"] == "output"
    assert v.node_box["x"] == "input"


def test_every_node_in_exactly_one_box_and_sums_hold():
    nodes, tot = _attn()
    v = abstract(nodes, tot)
    members = [m for b in v.boxes for m in b.members]
    assert sorted(members) == sorted(n["id"] for n in nodes)
    for k in ("flops", "params", "weight_bytes"):
        assert sum(getattr(b.sums, k) for b in v.boxes) == getattr(tot, k)


def test_box_formula_is_the_sum_of_member_formulas():
    nodes, tot = _attn()
    boxes = {b.id: b for b in abstract(nodes, tot).boxes}
    assert boxes["core"].flops_symbolic == "7"
    assert boxes["output"].flops_symbolic == "10"
    # A sum that collapsed to a number is not a law over the workload.
    assert not boxes["core"].flops_is_law


def test_a_box_formula_over_workload_axes_is_a_law():
    nodes, tot = _attn()
    nodes = [dict(n, flops_law=f"{n['flops']}*B") if n.get("flops_law") else n
             for n in nodes]
    boxes = {b.id: b for b in abstract(nodes, tot).boxes}
    assert boxes["core"].flops_symbolic == "7*B"
    assert boxes["core"].flops_is_law


def test_a_fused_wrapper_lifts_inner_anchor_roles():
    """Fusion nests score/apply under flash_attention; the wrapper carries
    the inner q, k, v roles so L1 can still group the component."""
    flash = {
        "id": "flash", "op": "flash_attention", "style": "fused",
        "inputs": ["q_proj", "k", "v"],
        "input_roles": ["q", "k", "v"],
        "flops": 7, "params": 0, "weight_bytes": 0,
    }
    nodes = [
        _n("x", "input"),
        _n("q_proj", "linear", ["x"], 10, 5),
        _n("kv_proj", "linear", ["x"], 4, 2),
        _n("k", "slice", ["kv_proj"]),
        _n("v", "slice", ["kv_proj"]),
        flash,
        _n("o_proj", "linear", ["flash"], 10, 5),
    ]
    tot = Sums(flops=sum(int(n.get("flops") or 0) for n in nodes),
               params=sum(int(n.get("params") or 0) for n in nodes),
               weight_bytes=sum(int(n.get("weight_bytes") or 0) for n in nodes))
    v = abstract(nodes, tot)
    assert v.available, v.reason
    assert v.node_box["q_proj"] == "q"
    assert v.node_box["kv_proj"] == "k+v"
    assert v.node_box["flash"] == "core"
    assert v.node_box["o_proj"] == "output"


def _box_edges(nodes, view):
    """Box-to-box edges of a view, from the producers its nodes read."""
    from backend.views.abstraction import _producers, _scope_holders
    box = view.node_box
    holders = _scope_holders(nodes)
    return {(box[src], box[n["id"]]) for n in nodes
            for src in _producers(n, holders)
            if src in box and box[src] != box[n["id"]]}


def _fused_mla():
    """What a serving framework runs for MLA: q_a and kv_a merged into one
    GEMM with two output ports, and the q/k rope fused into one kernel that
    reads Q's split heads and hands back `q_out` and `k_out`."""
    def ported(nid, op, refs, **kw):
        return _n(nid, op, [r if not p else f"{r}.{p}" for r, p in refs],
                  input_ports=[p for _r, p in refs], **kw)
    nodes = [
        _n("x", "input"),
        _n("qkv_a", "merged_column_parallel_linear", ["x"], 6, 3, flops_law="6"),
        ported("q_a_norm", "rmsnorm", [("qkv_a", "output_1")]),
        _n("q_b", "linear", ["q_a_norm"], 4, 2, flops_law="4"),
        ported("kv_a_norm", "rmsnorm", [("qkv_a", "output_0")]),
        _n("kv_b", "linear", ["kv_a_norm"], 4, 2, flops_law="4"),
        _n("v", "slice", ["kv_b"]),
        ported("rope", "fused_qk_rope", [("q_b", ""), ("qkv_a", "output_0")]),
        ported("q", "concat", [("rope", "q_out"), ("q_b", "")]),
        ported("k", "concat", [("kv_b", ""), ("rope", "k_out")]),
        _n("score", "attention_score", ["q", "k"], 3, flops_law="3"),
        _n("apply", "attention_apply", ["score", "v"], 3, flops_law="3"),
        _n("o_proj", "linear", ["apply"], 10, 5, flops_law="10"),
    ]
    tot = Sums(flops=sum(n["flops"] for n in nodes), params=sum(n["params"] for n in nodes),
               weight_bytes=sum(n["weight_bytes"] for n in nodes))
    return nodes, tot


def test_an_input_through_a_named_port_is_an_edge_to_its_producer():
    """`<producer>.<port>` reads the producer: the merged projection feeds
    Q, K and V, the fused rope feeds Q and K. Dropping ported edges left both
    reached by no operand — "beside the core" — and drew a cycle."""
    nodes, tot = _fused_mla()
    v = abstract(nodes, tot)
    assert v.available, v.reason
    assert v.node_box["qkv_a"] == "q+k+v"
    assert v.node_box["rope"] == "q+k"
    assert "other" not in {b.id for b in v.boxes}


def test_a_read_of_a_compounds_member_is_an_edge_from_the_compound():
    """A consumer rewired to a fused kernel inside the router reads that
    member (`router.fused_gate`), not the router. The drawn edge leaves the
    router, so the routing operand's branch is the router; read as a node
    of no scope, the router reached no operand and fell beside the core."""
    nodes = [
        _n("x", "input"),
        _n("router", "router", ["x"], 2, 1, flops_law="2", children=[
            _n("router.linear", "linear", ["x"]),
            _n("router.fused_gate", "moe_fused_gate", ["router.linear"]),
        ]),
        _n("permute", "permute", ["x", "router.fused_gate"]),
        _n("experts", "gated_experts", ["permute"], 8, 4, flops_law="8"),
        _n("combine", "combine", ["experts", "router.fused_gate"]),
        _n("shared", "swiglu_ffn", ["x"], 4, 2, flops_law="4"),
        _n("out", "add", ["combine", "shared"]),
    ]
    tot = Sums(flops=14, params=7, weight_bytes=14)
    v = abstract(nodes, tot)
    assert v.available, v.reason
    assert v.node_box["router"] == "routing"
    assert v.node_box["shared"] == "other"


def test_the_boxes_form_no_cycle():
    """A box's producers are read by every operand its readers are, so an
    edge between boxes only runs from more operands to fewer: the boxes are
    a DAG, even around a fused kernel inside one operand's path."""
    nodes, tot = _fused_mla()
    edges = _box_edges(nodes, abstract(nodes, tot))
    left = {b for e in edges for b in e}
    while left:   # peel boxes nothing remaining feeds; a cycle never peels
        free = {b for b in left if not any(d == b and s in left for s, d in edges)}
        assert free, f"boxes in a cycle: {sorted(left)}"
        left -= free


def test_refuses_without_an_anchor():
    v = abstract([_n("x", "input"), _n("a", "linear", ["x"], 1, 1)],
                 Sums(flops=1, params=1, weight_bytes=2))
    assert not v.available and "no anchor" in v.reason


def test_refuses_when_boxes_do_not_add_up():
    nodes, tot = _attn()
    v = abstract(nodes, Sums(tot.flops + 1, tot.params, tot.weight_bytes))
    assert not v.available and "add up" in v.reason


def test_refuses_two_independent_cores():
    a, _ = _attn()
    b = [dict(n, id=f"b.{n['id']}", inputs=[f"b.{i}" if i != "x" else "x" for i in n["inputs"]])
         for n in a if n["op"] != "input"]
    nodes = a + b
    tot = Sums(*(sum(n[k] for n in nodes) for k in ("flops", "params", "weight_bytes")))
    v = abstract(nodes, tot)
    assert not v.available and "2 anchor cores" in v.reason


def test_an_unreadable_formula_declines_the_view_not_the_module():
    nodes, tot = _attn()
    nodes = [dict(n, flops_law="2*(B") if n["id"] == "o_proj" else n for n in nodes]
    v = abstract(nodes, tot)
    assert not v.available and "does not read back" in v.reason
    assert v.totals == tot


def test_refuses_an_anchor_whose_inputs_were_folded():
    nodes, tot = _attn()
    nodes = [dict(n, absorbed_params=[{"id": "p", "params": 0}]) if n["id"] == "score" else n
             for n in nodes]
    v = abstract(nodes, tot)
    assert not v.available and "folded" in v.reason


# --- architectural facts on real components --------------------------------

def _view_of(component, model_id):
    from backend.arch.eval.transforms import analyze_isolated
    from backend.arch.loader import SpecLoader
    from backend.playground import module_bench as mb
    from tests.snapshots import bench_call, resolve_snapshot
    loader = SpecLoader()
    host = mb.inject_component(resolve_snapshot(model_id, loader), component, loader)
    # The drawing and the report are of one call: the host's own call on
    # its backbone, as the bench isolates it.
    site = bench_call(host, component)
    kw = dict(batch=1, input_tokens=512, dim_overrides={}, site=site)
    rep = mb.project_block(host, component, **kw)
    nodes = mb._block_graph(host, component, precision=AS_SHIPPED, **kw)["nodes"]
    # The weights are the stored copy the bench reports: parameters, their
    # scales and buffers — the same target module-bench builds its view on.
    stored, _parts = mb._weight_storage(
        analyze_isolated(host, component, {}, site), site, bound=False)
    return abstract(nodes, Sums(int(rep.total_flops), int(rep.total_params),
                                stored["bytes"]))


def _view(component, model_id):
    v = _view_of(component, model_id)
    assert v.available, v.reason
    return {b.id: b for b in v.boxes}


def test_absorbed_mla_value_is_the_key_latent():
    """Absorbed MLA reads V straight from the shared latent K is joined
    from: what V has of its own is the zero-cost view that shows the one
    latent to every head — no weight, no FLOP. Unabsorbed MLA expands V per
    head."""
    absorbed = _view("mla_attention_absorbed", "deepseek-ai/DeepSeek-V3")
    plain = _view("mla_attention", "deepseek-ai/DeepSeek-V3")
    assert "k+v" in absorbed
    own_v = absorbed.get("v")
    assert own_v is None or (own_v.sums.params == 0 and own_v.sums.flops == 0)
    assert "v" in plain
    # The absorbed form moves W_uk into Q and W_uv to the output; the total
    # is unchanged, the K+V latent path is what shrinks.
    assert absorbed["k+v"].sums.params < plain["k+v"].sums.params
    assert sum(b.sums.params for b in absorbed.values()) == sum(b.sums.params for b in plain.values())


def test_dsa_adds_a_selection_that_shares_the_query_latent():
    boxes = _view("mla_dsa_attention", "deepseek-ai/DeepSeek-V3.2")
    assert "select" in boxes and "q+select" in boxes


def test_mha_projects_q_k_v_alike():
    boxes = _view("mha_attention", "meta-llama/Llama-2-7b-hf")
    assert boxes["q"].sums.params == boxes["k"].sums.params == boxes["v"].sums.params


def test_latent_moe_projects_tokens_before_the_experts():
    latent = _view("kimi_k3_latent_moe", "moonshotai/Kimi-K3")
    plain = _view("moe_ffn", "Qwen/Qwen3-30B-A3B")
    assert latent["tokens"].sums.params > 0 and "tokens" not in plain
    assert "routing" in latent and "routing" in plain


def test_shared_experts_run_beside_the_routed_ones():
    boxes = _view("dsv4_moe_hash", "deepseek-ai/DeepSeek-V4-Pro")
    assert boxes["other"].sums.params > 0   # shared expert: not routed, not downstream


def test_folded_parameters_keep_operand_positions():
    """`mamba2_scan` takes per-head parameters (A_log at 2, D and dt_bias at
    5-6) between its operands. The renderer folds them into the scan and
    drops them from `inputs`; the roles must still land on x, dt, B and C."""
    boxes = _view("nemotron_h_mamba_mixer", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16")
    reached = {r for bid in boxes for r in bid.split("+")}
    assert {"q", "k", "v", "decay"} <= reached


def test_ports_skip_the_positions_folded_parameters_held():
    from backend.views.abstraction import _ports
    node = {"inputs": ["x", "dt", "b", "c"],
            "absorbed_params": [{"id": "A", "input_positions": [2]},
                                {"id": "D", "input_positions": [5]},
                                {"id": "dt_bias", "input_positions": [6]}]}
    assert _ports(node, {}) == [(0, "x"), (1, "dt"), (3, "b"), (4, "c")]
    assert _ports({"inputs": ["x"], "absorbed_params": [{"id": "A"}]}, {}) is None


# --- stays a view ------------------------------------------------------------

#: Attention components whose core op declares no operand roles yet (linear
#: attention / state-space). Removing one from here means it gained roles.
ANCHORLESS_ATTENTION = {
    # core is a spec-declared custom op (`gla_simple_recurrence`), not a
    # registry op, so it has no OpSpec to carry roles
    "gla_qknorm_partial_rope_sigmoid_gated_attention",
}


def test_every_attention_component_has_an_anchor_or_is_listed():
    import backend.arch.ops_registry  # noqa: F401
    from backend.arch.ops_base import OP_REGISTRY

    def ops_in(obj, acc):
        if isinstance(obj, dict):
            if isinstance(obj.get("op"), str):
                acc.add(obj["op"])
            for v in obj.values():
                ops_in(v, acc)
        elif isinstance(obj, list):
            for v in obj:
                ops_in(v, acc)
        return acc

    anchorless = set()
    for f in glob.glob(str(ROOT / "backend/arch/components/library/*.yaml")):
        doc = yaml.safe_load(pathlib.Path(f).read_text()) or {}
        if doc.get("role") != "attention":
            continue
        ops = ops_in(doc.get("graph") or {}, set())
        if not any(OP_REGISTRY[o].input_roles for o in ops if o in OP_REGISTRY):
            anchorless.add(pathlib.Path(f).stem)
    assert anchorless == ANCHORLESS_ATTENTION


def test_nothing_upstream_imports_the_views():
    offenders = []
    for f in glob.glob(str(ROOT / "backend/arch/**/*.py"), recursive=True) + \
             glob.glob(str(ROOT / "backend/spec_worker/**/*.py"), recursive=True):
        tree = ast.parse(pathlib.Path(f).read_text())
        for node in ast.walk(tree):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, (ast.Import, ast.ImportFrom)) else []
            if (mod and mod.startswith("backend.views")) or any(n.startswith("backend.views") for n in names):
                offenders.append(f)
    assert not offenders, offenders


def test_the_view_reads_no_component_or_node_names():
    """Every string literal in the pass is a box name, a node-dict key, or
    documentation — never a component, op or node name to match against."""
    from backend.arch.ops_base import INPUT_ROLE_VOCABULARY
    src = (ROOT / "backend/views/abstraction.py").read_text()
    allowed = set(INPUT_ROLE_VOCABULARY) | {
        "attention", "experts", "recurrence",            # core kinds (from roles)
        "input", "core", "output", "other",              # box ids / the input pseudo-op
        "flops", "params", "weight_bytes",               # summed keys
        "id", "op", "inputs", "input_ports", "absorbed_params", "input_positions", "flops_symbolic",
        "children",
        "flops_law",
        "flops_is_law",
        "input_roles",
        "members", "ops", "available", "reason", "boxes", "node_box", "totals",
        "other_merges", "label", "rank", "", "kv",
        "Output", "Core",                                # display names of the fixed boxes
        "+", "0", ".",                                   # box id joiner, a zero law, the port separator
    }
    tree = ast.parse(src)
    doc_nodes = {id(n.body[0].value) for n in ast.walk(tree)
                 if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
                 and n.body and isinstance(n.body[0], ast.Expr)
                 and isinstance(n.body[0].value, ast.Constant)}
    strays = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in doc_nodes
                and node.value not in allowed
                and not any(c in node.value for c in " {")):
            strays.add(node.value)
    assert not strays, strays


def test_the_core_is_named_by_its_anchors_roles():
    nodes, tot = _attn()
    assert abstract(nodes, tot).core == "attention"
    moe = [
        _n("x", "input"),
        _n("gate", "linear", ["x"], 2, 1, flops_law="2"),
        _n("permute", "permute", ["x", "gate"]),
        _n("experts", "swiglu_ffn", ["permute"], 8, 4, flops_law="8"),
        _n("combine", "combine", ["experts", "gate"], 1, flops_law="1"),
    ]
    tot = Sums(11, 5, 10)
    v = abstract(moe, tot)
    assert v.available and v.core == "experts"
    assert v.node_box["gate"] == "routing" and v.node_box["experts"] == "core"


@pytest.mark.parametrize("component,model_id", [
    ("mla_attention", "deepseek-ai/DeepSeek-V3"),
    ("mla_dsa_attention", "deepseek-ai/DeepSeek-V3.2"),
    ("moe_ffn_gate_routing_bias", "XiaomiMiMo/MiMo-V2-Flash"),
    ("swa_bias_sink_attention", "openai/gpt-oss-20b"),
    ("gated_delta_net_attention", "Qwen/Qwen3-Coder-Next"),
    ("kimi_k3_kda_attention", "moonshotai/Kimi-K3"),
])
def test_a_box_formula_evaluates_to_its_box_flops(component, model_id):
    """Proposal §3.4: a box's formula, with the component's resolved dims
    substituted, is the sum of its members' FLOPs."""
    from backend.arch.law_text import read_law
    from backend.arch.loader import SpecLoader
    from backend.playground import module_bench as mb
    from tests.snapshots import bench_call, resolve_snapshot
    loader = SpecLoader()
    base = resolve_snapshot(model_id, loader)
    row = mb.bench_component(base, component, batch=1, site=bench_call(base, component),
                             input_tokens=512, dim_overrides={}, loader=loader, precision=AS_SHIPPED)
    assert row["l1"]["available"], row["l1"]["reason"]
    dims = row["dims_used"]
    for box in row["l1"]["boxes"]:
        expr = read_law(box["flops_symbolic"])
        missing = {str(s) for s in expr.free_symbols} - set(dims)
        assert not missing, (box["id"], missing)
        assert round(float(expr.subs(dims))) == box["flops"], box["id"]


def test_linear_attention_groups_its_recurrence_operands():
    """Gated DeltaNet and KDA feed one recurrence q, k, v, a state decay and a
    write strength. KDA's decay is per channel (a learned projection) where
    GDN's is one scalar per head, so KDA's decay path carries far more weight."""
    gdn = _view("gated_delta_net_attention", "Qwen/Qwen3-Coder-Next")
    kda = _view("kimi_k3_kda_attention", "moonshotai/Kimi-K3")
    for boxes in (gdn, kda):
        assert {"core", "output"} <= set(boxes)
        assert boxes["core"].ops == {"gated_delta_rule": 1}
        assert any("decay" in b.split("+") for b in boxes)
        assert any("write" in b.split("+") for b in boxes)
    decay = lambda bx: sum(b.sums.params for k, b in bx.items() if "decay" in k.split("+"))
    assert decay(kda) > 10 * decay(gdn)


def test_the_core_kind_of_linear_attention_is_recurrence():
    assert _view_of("gated_delta_net_attention", "Qwen/Qwen3-Coder-Next").core == "recurrence"
    assert _view_of("kimi_k3_kda_attention", "moonshotai/Kimi-K3").core == "recurrence"


def test_a_renaming_wrapper_is_drawn_as_what_it_renames():
    """ChatGLM's attention is a one-node wrapper over `gqa_fused_qkv_attention`
    (BlockGraph.alias_target). The Playground draws the renamed component, so
    the anchors are at the top level, and the view still adds up to the
    wrapper's own totals."""
    from backend.arch.loader import SpecLoader
    from backend.playground import module_bench as mb
    from tests.snapshots import resolve_snapshot
    loader = SpecLoader()
    out = mb.bench_component(resolve_snapshot("THUDM/glm-4-9b-chat", loader),
                             "chatglm_attention", batch=1, input_tokens=512,
                             dim_overrides={}, loader=loader,
                             precision=AS_SHIPPED)
    assert out["l1"]["available"], out["l1"]["reason"]
    assert out["l1"]["core"] == "attention"
    assert out["l1"]["totals"]["flops"] == out["phases"]["prefill"]["flops"]


def test_a_one_node_block_that_binds_something_is_not_a_rename():
    """`chatglm_block` is one `pre_norm` node binding `attn` / `ffn`: the
    target alone is not the block, so it is drawn as itself (and benches)."""
    from backend.arch.loader import SpecLoader
    from backend.playground import module_bench as mb
    from tests.snapshots import resolve_snapshot
    loader = SpecLoader()
    names = set(loader.component_names())
    resolve = lambda op: loader.component(op) if op in names else None
    assert loader.component("chatglm_attention").renamed_component(resolve) == "gqa_fused_qkv_attention"
    assert loader.component("chatglm_block").renamed_component(resolve) is None
    out = mb.bench_component(resolve_snapshot("THUDM/glm-4-9b-chat", loader),
                             "chatglm_block", batch=1, input_tokens=512,
                             dim_overrides={}, loader=loader,
                             precision=AS_SHIPPED)
    assert out["phases"]["prefill"]["flops"] > 0


def test_a_graph_without_cost_leaves_box_cost_unknown():
    """A serving graph draws topology without per-op cost: its boxes state no
    params / FLOPs (None) rather than summing the absent figures to zero."""
    nodes, _ = _attn()
    bare = [{k: v for k, v in n.items()
             if k not in ("flops", "params", "weight_bytes", "flops_law")}
            for n in nodes]
    v = abstract(bare, Sums(None, None, None))
    assert v.available, v.reason
    for b in v.to_dict()["boxes"]:
        assert (b["flops"], b["params"], b["weight_bytes"], b["flops_symbolic"]) \
            == (None, None, None, None), b["id"]
