"""Compute-graph diff (backend.views.diff).

The diff lines several components' L2 graphs up from declarations alone and
reports the skeleton they share, the paths only some have, and the paired
nodes whose attributes differ. These tests pin (1) the walk's refusals and
its "do not guess" rule on small hand-built graphs; (2) the algebraic gates
over the attention corpus — self-diff empty, mirror, order-free, no class
holding two nodes of one entry; (3) the payload's own consistency — every
node in one class, regions adding up to the totals, a full expansion giving
back each entry's graph, a reason for every changed node; (4) the structural
stories on real components; (5) that it reads no names.
"""
from __future__ import annotations

import ast
import itertools
import pathlib
import random

import pytest

from backend.arch.schema.custom_ops import CustomOpSpec
from backend.playground import module_bench as mb
from backend.views.abstraction import Sums, abstract
from backend.arch.eval.precision import AS_SHIPPED
from backend.views.diff import (
    ANCHOR, ANCHOR_SUBSTITUTED, BACKTRACK, BY_BOX, BY_LAYER, FORWARD, INPUT,
    OUTPUT, SAME_SLOT, Alignment, DiffRefused, Entry, GraphDiff,
)
from backend.rendering._node_build import declaration_card
from tests.backend.views import declared_roles

ROOT = pathlib.Path(__file__).resolve().parents[3]

# Building the attention corpus benched every shipped attention component
# before the first assertion: 597s on the loaded dev box, right at the default
# suite's 600s per-test ceiling.  This is expensive, not stuck — the assertions
# after it run in fractions of a second — so give this legitimate corpus build
# the same 1200s ceiling the regression workflow uses for its long cases.
pytestmark = pytest.mark.timeout(1200)


# --- hand-built graphs --------------------------------------------------------

def _n(nid, op, inputs=(), flops=0, params=0, **kw):
    return {"id": nid, "op": op, "tensor_name": nid, "inputs": list(inputs),
            "flops": flops, "params": params, "weight_bytes": 2 * params,
            **declared_roles(op), **kw}


def _entry(key, nodes, *, custom_ops=None, outputs=()):
    custom_ops = custom_ops or {}
    nodes = [dict(n, declaration=declaration_card(custom_ops[n["op"]], n["op"]))
             if n["op"] in custom_ops else n for n in nodes]
    tot = Sums(sum(n["flops"] for n in nodes), sum(n["params"] for n in nodes),
               sum(n["weight_bytes"] for n in nodes))
    return Entry(key=key, block_graph={"nodes": nodes, "edges": []},
                 l1=abstract(nodes, tot).to_dict(), dims={},
                 inputs=["x"], outputs=list(outputs), reads=[])


def _attn(select=False):
    nodes = [
        _n("x", "input"),
        _n("q", "linear", ["x"], 10, 5),
        _n("kv", "linear", ["x"], 4, 2),
        _n("k", "slice", ["kv"]),
        _n("v", "slice", ["kv"]),
    ]
    if select:
        nodes += [_n("idx", "linear", ["x"], 3, 1),
                  _n("mask", "dsa_topk_mask", ["idx"])]
        nodes += [_n("score", "dsa_attention_score", ["q", "k", "mask"], 3),
                  _n("sm", "dsa_softmax", ["score"], 1),
                  _n("apply", "dsa_attention_apply", ["sm", "v"], 3)]
    else:
        nodes += [_n("score", "attention_score", ["q", "k"], 3),
                  _n("sm", "softmax", ["score"], 1),
                  _n("apply", "attention_apply", ["sm", "v"], 3)]
    nodes.append(_n("o", "linear", ["apply"], 10, 5))
    return nodes


def _recurrence(**over):
    decl = {"signature": {"x": ["B", "S", "D"]}, "shape_kind": "preserve",
            "couples_on": [], "flops": "B*S*D",
            "description": "a stand-in recurrence no registered primitive composes"}
    decl.update(over)
    return CustomOpSpec(**decl)


def test_a_dense_and_a_sparse_attention_share_everything_but_the_selection():
    al = Alignment.of(_entry("a", _attn(), outputs=["o"]),
                      _entry("b", _attn(select=True), outputs=["o"]))
    assert al.pairs == {i: i for i in ("x", "q", "kv", "k", "v", "score", "sm", "apply", "o")}
    assert al.basis["x"] == INPUT and al.basis["o"] == OUTPUT
    # score and apply declare operand roles and are different ops: substituted
    # anchors; softmax sits between two paired nodes with an equal signature.
    assert al.basis["score"] == al.basis["apply"] == ANCHOR_SUBSTITUTED
    assert al.basis["sm"] == SAME_SLOT
    assert al.basis["q"] == al.basis["k"] == BACKTRACK
    gd = GraphDiff.of([_entry("a", _attn(), outputs=["o"]),
                       _entry("b", _attn(select=True), outputs=["o"])]).to_dict()
    (path,) = gd["paths"]
    assert path["owners"] == [1]
    assert [t["port"] for t in path["into"]] == ["select"]


def test_indistinguishable_consumers_are_not_guessed():
    """Two identical projections read the input and nothing downstream tells
    them apart: the walk pairs neither rather than pick."""
    two = [_n("x", "input"), _n("l1", "linear", ["x"], 1, 1), _n("l2", "linear", ["x"], 1, 1)]
    al = Alignment.of(_entry("a", two), _entry("b", [dict(n) for n in two]))
    assert al.pairs == {"x": "x"}
    one = [_n("x", "input"), _n("l1", "linear", ["x"], 1, 1)]
    assert Alignment.of(_entry("a", one), _entry("b", one)).pairs == {"x": "x", "l1": "l1"}
    assert Alignment.of(_entry("a", one), _entry("b", one)).basis["l1"] == FORWARD


def test_a_custom_op_is_its_declaration_not_its_name():
    def graph(op):
        return [_n("x", "input"), _n("r", op, ["x"], 1), _n("o", "linear", ["r"], 2, 1)]

    same = Alignment.of(_entry("a", graph("rec_a"), custom_ops={"rec_a": _recurrence()}, outputs=["o"]),
                        _entry("b", graph("rec_b"), custom_ops={"rec_b": _recurrence()}, outputs=["o"]))
    assert same.pairs["r"] == "r" and same.basis["r"] == BACKTRACK
    # One name, two declarations: not the same op. Its neighbours are paired
    # and the signatures agree, so it is a substitution in the same slot.
    other = GraphDiff.of([
        _entry("a", graph("rec"), custom_ops={"rec": _recurrence()}, outputs=["o"]),
        _entry("b", graph("rec"), custom_ops={"rec": _recurrence(flops="2*B*S*D")}, outputs=["o"]),
    ]).to_dict()
    r = next(c for c in other["classes"] if c["members"] == ["r", "r"])
    assert r["basis"] == [{"pair": [0, 1], "basis": SAME_SLOT}] and r["because"]["declaration"]
    assert r["diff"]["op"] == ["rec", "rec"]


def test_refuses_a_leaf_with_no_declaration():
    nodes = [_n("x", "input"), _n("mystery", "not_an_op_anywhere", ["x"])]
    with pytest.raises(DiffRefused, match="mystery"):
        _entry("a", nodes)


def test_refuses_inputs_folded_without_positions():
    nodes = _attn()
    nodes[5] = dict(nodes[5], absorbed_params=[{"id": "p"}])
    with pytest.raises(DiffRefused, match="folded"):
        _entry("a", nodes)


def test_refuses_a_formula_that_does_not_read_back():
    """A changed node's "because" reads its formulas; one outside the law
    language declines the diff with the reason, it does not raise out."""
    a, b = _attn(), _attn()
    b[-1] = dict(b[-1], params=6, flops_symbolic="not_an_operator(d_model)")
    gd = GraphDiff.of([_entry("a", a, outputs=["o"]), _entry("b", b, outputs=["o"])])
    with pytest.raises(DiffRefused, match="does not read back"):
        gd.to_dict()


def test_refuses_fewer_than_two_entries():
    with pytest.raises(DiffRefused):
        GraphDiff.of([_entry("a", _attn())])


# --- the attention corpus -----------------------------------------------------

@pytest.fixture(scope="module")
def loader():
    from backend.arch.loader import SpecLoader
    return SpecLoader()


def _call(model, component, loader):
    """The call a test evaluates `component` at in `model` (`bench_call`)."""
    from tests.snapshots import bench_call
    return bench_call(model, component, loader).call


def _bench_entry(loader, component, model_id, *, key=None, dims=None, base_id=None):
    """One entry the way the Playground benches it: on a base (the binding
    model unless given), with the binding's dims."""
    from backend.arch.eval.transforms import CallSite
    from tests.snapshots import resolve_snapshot
    base = resolve_snapshot(base_id or model_id, loader)
    model = resolve_snapshot(model_id, loader)
    call = _call(model, component, loader)
    ov = mb.component_dims_from_model(component, model_id, call)
    ov.update(dims or {})
    row = mb.bench_component(base, component, batch=1, input_tokens=512,
                             dim_overrides=ov, loader=loader, precision=AS_SHIPPED,
                             site=CallSite.of(model, component, call=call))
    row["entry"] = key or component
    return mb._diff_entry(base, row, loader), row


@pytest.fixture(scope="module")
def corpus(loader):
    """Every attention library component a shipped model binds in its
    language model, on its first binding's dims. An encoder's attention is
    called in the encode phase, over the media's own axes, and is not
    benched alone at a language-model step."""
    from tests.snapshots import bench_call, resolve_snapshot
    bound = mb._component_to_models()
    out = {}
    for c in mb.list_library_components():
        if c.get("role") != "attention" or not bound.get(c["name"]):
            continue
        model = resolve_snapshot(bound[c["name"]][0], loader)
        if bench_call(model, c["name"], loader).encoder is not None:
            continue
        out[c["name"]] = _bench_entry(loader, c["name"], bound[c["name"]][0])[0]
    return out


def _classes(gd: GraphDiff):
    """Order-free: each class as the set of (entry, node) it holds."""
    return frozenset(frozenset((gd.entries[i].key, nid) for i, nid in enumerate(row) if nid)
                     for row in gd.classes)


def test_self_diff_is_empty(corpus):
    for name, e in corpus.items():
        gd = GraphDiff.of([e, e]).to_dict()
        assert all(c["members"][0] == c["members"][1] for c in gd["classes"]), name
        assert len(gd["classes"]) == len(e.nodes), name
        assert not gd["paths"] and not any(c["diff"] for c in gd["classes"]), name


def test_the_diff_of_b_and_a_mirrors_the_diff_of_a_and_b(corpus):
    names = sorted(corpus)
    for x, y in itertools.combinations(names, 2):
        ab, ba = Alignment.of(corpus[x], corpus[y]), Alignment.of(corpus[y], corpus[x])
        assert ab.pairs == ba.back, (x, y)
        assert ab.basis == {u: ba.basis[v] for u, v in ab.pairs.items()}, (x, y)


def test_no_class_holds_two_nodes_of_one_entry(corpus):
    """GraphDiff refuses such a class; over every triple it never has to.
    The all-agree rule is what keeps it so: the raw union of the pairwise
    alignments does merge two nodes of one entry somewhere in the corpus."""
    names = sorted(corpus)
    raw_conflicts = 0
    for triple in itertools.combinations(names, 3):
        gd = GraphDiff.of([corpus[n] for n in triple])
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                x = parent[x]
            return x
        for (i, j), al in gd.alignments.items():
            for u, v in al.pairs.items():
                parent[find((i, u))] = find((j, v))
        groups = {}
        for i, e in enumerate(gd.entries):
            for nid in e.nodes:
                groups.setdefault(find((i, nid)), []).append(i)
        raw_conflicts += any(len(set(g)) < len(g) for g in groups.values())
    assert raw_conflicts >= 1


def test_entry_order_does_not_change_the_classes(corpus):
    names = sorted(corpus)
    rng = random.Random(0)
    for _ in range(100):
        pick = rng.sample(names, 5)
        shuffled = rng.sample(pick, 5)
        assert _classes(GraphDiff.of([corpus[n] for n in pick])) == \
            _classes(GraphDiff.of([corpus[n] for n in shuffled])), pick


def test_one_component_under_two_bindings_pairs_node_for_node(loader):
    """The same-component case is not special-cased: equal declarations and
    equal neighbours walk out to the node-id correspondence."""
    v32 = "deepseek-ai/DeepSeek-V3.2"
    a, _ = _bench_entry(loader, "mla_dsa_attention", v32, key="v32")
    b, _ = _bench_entry(loader, "mla_dsa_attention", "zai-org/GLM-5", key="glm5", base_id=v32)
    gd = GraphDiff.of([a, b])
    assert all(row[0] == row[1] for row in gd.classes)
    assert len(gd.classes) == len(a.nodes) == len(b.nodes)
    payload = gd.to_dict()
    changed = [c for c in payload["classes"] if c["diff"]]
    # The nodes whose extents differ and the weights GLM-5's checkpoint
    # stores wider than V3.2's (each entry is priced at its own model's
    # checkpoint, as the Playground prices a bound entry) differ; each says
    # why below.
    assert changed and len(changed) < len(payload["classes"])
    for c in changed:
        assert c["because"]["dims"] or c["because"]["weight_dtype"], c["members"]
        assert not c["because"]["declaration"]
    q_b = next(c for c in changed if c["members"][0].endswith("|q_b_proj"))
    assert set(q_b["because"]["dims"]) == {"d_q_lora", "d_qk_nope", "n_heads"}

    qwen = "Qwen/Qwen3-30B-A3B"
    x, _ = _bench_entry(loader, "moe_ffn", qwen, key="qwen")
    y, _ = _bench_entry(loader, "moe_ffn", "deepseek-ai/DeepSeek-V2-Lite", key="lite", base_id=qwen)
    gd = GraphDiff.of([x, y])
    paired = [row for row in gd.classes if row[0] and row[1]]
    assert len(paired) == 5 and all(row[0] == row[1] for row in paired)
    # The routers pair by position — the same routing site — and the payload
    # says what differs there: Qwen3 renormalizes its top-k weights
    # (norm_topk_prob=True), V2-Lite does not, and their expert counts differ.
    # V2-Lite alone has shared experts, and so alone adds them to `combine`
    # in `moe_out`.
    gate = next(c for c in gd.to_dict()["classes"]
                if c["members"][0] == "moe_ffn_site|gate")
    assert gate["because"]["declaration"]
    assert set(gate["because"]["dims"]) == {
        "n_experts_active", "n_route_targets"}
    assert {tuple(row) for row in gd.classes if not (row[0] and row[1])} == {
        (None, "moe_ffn_site|shared_experts"),
        (None, "moe_ffn_site|moe_out__moe_ffn")}


# --- the payload through module_bench ----------------------------------------

_ATTENTION = [
    {"name": "gqa_attention", "model_id": "Qwen/Qwen2.5-7B"},
    {"name": "mla_attention", "model_id": "deepseek-ai/DeepSeek-V3"},
    {"name": "mla_dsa_attention", "model_id": "deepseek-ai/DeepSeek-V3.2"},
    {"name": "mla_dsa_attention_absorbed", "model_id": None},
    {"name": "mla_dsa_attention_shared", "model_id": "zai-org/GLM-5.2"},
]
_LINEAR = [
    {"name": "gated_delta_net_attention", "model_id": "Qwen/Qwen3.5-35B-A3B"},
    {"name": "kimi_k3_kda_attention", "model_id": "moonshotai/Kimi-K3"},
    {"name": "nemotron_h_mamba_mixer",
     "model_id": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"},
]


def _bench(base, entries):
    from tests.snapshots import resolve_snapshot, snapshot_config
    snapshot_config(base)
    for e in entries:
        if e["model_id"]:
            snapshot_config(e["model_id"])
    loader = mb.get_default_loader()
    entries = [{**e, "call": _call(resolve_snapshot(e["model_id"] or base, loader),
                                   e["name"], loader)} for e in entries]
    out = mb.module_bench(base_model_id=base, entries=entries, batch=1,
                          input_tokens=512, precision=AS_SHIPPED)
    assert not out["errors"], out["errors"]
    assert out["graph_diff"]["available"], out["graph_diff"]["reason"]
    # The same entries module_bench diffed, for the tests that re-run the diff.
    host = resolve_snapshot(base, loader)
    out["diff_entries"] = [mb._diff_entry(host, r, loader) for r in out["results"]]
    return out


@pytest.fixture(scope="module")
def attention():
    return _bench("deepseek-ai/DeepSeek-V3.2", _ATTENTION)


@pytest.fixture(scope="module")
def linear():
    return _bench("Qwen/Qwen3-Coder-Next", _LINEAR)


@pytest.fixture(scope="module")
def gqa_mla():
    return _bench("deepseek-ai/DeepSeek-V3", [
        {"name": "gqa_qknorm_attention", "model_id": "Qwen/Qwen3-8B"},
        {"name": "mla_attention", "model_id": "deepseek-ai/DeepSeek-V3"}])


def _tensors(marks):
    return {m["tensor"] for m in marks}


class _View:
    """Test-side reading aid: find a class or path by what a member is
    called in its own entry (the diff never does this)."""

    def __init__(self, bench):
        self.rows = bench["results"]
        self.gd = bench["graph_diff"]
        self.cls = {c["id"]: c for c in self.gd["classes"]}
        self.node = [{n["id"]: n for n in r["block_graph"]["nodes"]} for r in self.rows]

    def name(self, i, nid):
        return self.node[i][nid]["tensor_name"] if nid in self.node[i] and \
            "tensor_name" in self.node[i][nid] else nid

    def cls_named(self, i, name):
        return next(c for c in self.gd["classes"]
                    if c["members"][i] and self.name(i, c["members"][i]) == name)

    def names(self, path, i):
        return {self.name(i, self.cls[c]["members"][i]) for c in path["classes"]}

    def path_with(self, i, name):
        c = self.cls_named(i, name)["id"]
        return next(p for p in self.gd["paths"] if c in p["classes"])

    def skeleton(self):
        n = len(self.rows)
        return [c for c in self.gd["classes"] if all(c["members"][:n])]

    def into(self, path):
        return {(self.name(next(i for i, m in enumerate(self.cls[t["class"]]["members"]) if m),
                           next(m for m in self.cls[t["class"]]["members"] if m)), t["port"])
                for t in path["into"]}


def test_payload_refers_to_the_rows_by_id(attention):
    v = _View(attention)
    assert v.gd["entries"] == [r["entry"] for r in v.rows]
    for c in v.gd["classes"]:
        assert len(c["members"]) == len(v.rows)
        for i, nid in enumerate(c["members"]):
            assert nid is None or nid in v.node[i]
        for i, box in enumerate(c["boxes"]):
            assert box == (v.rows[i]["l1"]["node_box"].get(c["members"][i])
                           if c["members"][i] else None)
    # Nothing of a row is copied: node dicts and L1 figures stay in the rows.
    assert not {"nodes", "l1", "block_graph"} & set(v.gd)
    for row in v.gd["boxes"]:
        assert set(row) == {"id", "label", "rank", "present", "delta"}


@pytest.mark.parametrize("bench", ["attention", "linear", "gqa_mla"])
def test_every_node_is_in_one_class_and_the_regions_add_up(bench, request):
    """Per entry: the skeleton plus the paths it owns is exactly its graph,
    and their sums are its module totals. A drawn node's weight bytes carry
    its parameters' quant scales and its buffers, so the regions add up to
    the stored copy the row reports (`weight_bytes`)."""
    v = _View(request.getfixturevalue(bench))
    skeleton = {c["id"] for c in v.skeleton()}
    in_paths = [c for p in v.gd["paths"] for c in p["classes"]]
    assert len(in_paths) == len(set(in_paths))
    assert skeleton | set(in_paths) == set(v.cls) and not skeleton & set(in_paths)
    for i, row in enumerate(v.rows):
        members = [c["members"][i] for c in v.gd["classes"] if c["members"][i]]
        assert sorted(members) == sorted(v.node[i])
        got = Sums()
        for c in skeleton:
            got.add(v.node[i][v.cls[c]["members"][i]])
        for p in v.gd["paths"]:
            if i in p["owners"]:
                s = p["sums"][i]
                got = Sums(got.flops + s["flops"], got.params + s["params"],
                           got.weight_bytes + s["weight_bytes"])
            else:
                assert p["sums"][i] is None
        # The drawn graph is the declared component, so its FLOPs close to the
        # declared walk (the L1 totals), not to the serving row's, which also
        # counts the activation casts serving grafts (a822d51b1).
        assert (got.flops, got.params, got.weight_bytes) == \
            (row["l1"]["totals"]["flops"], row["params"],
             row["weight_bytes"]), row["entry"]


@pytest.mark.parametrize("bench", ["attention", "linear", "gqa_mla"])
def test_expanding_everything_gives_back_each_graph(bench, request):
    """Open every box and path: what belongs to an entry is its Detailed
    graph, nodes and edges."""
    v = _View(request.getfixturevalue(bench))
    for i, row in enumerate(v.rows):
        member = {c["id"]: c["members"][i] for c in v.gd["classes"] if c["members"][i]}
        drawn = sorted((member[e["src"]], member[e["dst"]])
                       for e in v.gd["edges"] if i in e["owners"])
        ids = set(v.node[i])
        detailed = sorted((e["src"], e["dst"]) for e in row["block_graph"]["edges"]
                          if e["src"] in ids and e["dst"] in ids)
        assert len(detailed) == len(row["block_graph"]["edges"])
        assert drawn == detailed, row["entry"]


@pytest.mark.parametrize("bench", ["attention", "linear", "gqa_mla"])
def test_every_changed_node_says_why(bench, request):
    v = _View(request.getfixturevalue(bench))
    for c in v.gd["classes"]:
        if c["diff"]:
            because = c["because"]
            assert (because["dims"] or because["declaration"]
                    or because["weight_dtype"]), c["members"]
            if "op" in c["diff"]:
                assert because["declaration"]
            # A checkpoint storing the node at another dtype is its own
            # reason, and the one it gives is the stored dtypes themselves.
            assert because["weight_dtype"] == c["diff"].get("weight_dtype")


def test_every_owner_pair_says_how_it_was_paired(attention):
    """A class of several owners carries one basis per pair of owners, each
    the pairwise alignment's own word for it."""
    gd, entries = attention["graph_diff"], attention["diff_entries"]
    for c in gd["classes"]:
        own = [i for i, m in enumerate(c["members"]) if m]
        assert [b["pair"] for b in c["basis"]] == [list(p) for p in itertools.combinations(own, 2)]
        for b in c["basis"]:
            i, j = b["pair"]
            assert b["basis"] == Alignment.of(entries[i], entries[j]).basis[c["members"][i]]
    skeleton = [c for c in gd["classes"] if all(c["members"])]
    assert skeleton and all(len(c["basis"]) == 10 for c in skeleton)


def test_order_of_the_probe_entries_does_not_matter(attention, linear):
    for bench in (attention, linear):
        entries = bench["diff_entries"]
        assert GraphDiff.of(entries).to_dict() == bench["graph_diff"]
        want = _classes(GraphDiff.of(entries))
        for perm in itertools.permutations(entries):
            assert _classes(GraphDiff.of(list(perm))) == want


# --- structural stories -------------------------------------------------------

def test_gqa_vs_mla(gqa_mla):
    """MLA replaces GQA's Q projection with a low-rank one and its two K/V
    projections with a compressed latent plus a decompression: both flow
    into the same core ports, so each is one alternative group."""
    v = _View(gqa_mla)
    skel = {v.name(0, c["members"][0]) for c in v.skeleton()}
    assert skel == {"x", "qk_score", "softmax", "attn_apply", "o_merge_heads", "o_proj"}
    assert all(c["diff"] for c in v.skeleton() if v.name(0, c["members"][0]) != "x")
    o_proj = v.cls_named(0, "o_proj")
    assert o_proj["diff"]["tensor_in"] == ["[1,512,4096]", "[1,512,16384]"]
    assert "tensor_out" not in o_proj["diff"] and "params" in o_proj["diff"]
    sizes = sorted((p["owners"][0], len(p["classes"])) for p in v.gd["paths"])
    assert sizes == [(0, 2), (0, 4), (0, 4), (1, 8), (1, 13)]
    q_gqa, q_mla = v.path_with(0, "q_proj"), v.path_with(1, "q_a_proj")
    assert q_gqa["group"] == q_mla["group"] is not None
    assert v.into(q_gqa) == v.into(q_mla) == {("qk_score", "q")}
    kv = v.path_with(1, "kv_a_proj_with_mqa")
    assert v.into(kv) == {("qk_score", "k"), ("attn_apply", "v")}
    assert kv["group"] == v.path_with(0, "k_proj")["group"] == v.path_with(0, "v_proj")["group"]
    assert _tensors(kv["kv"][1]["writes"]) == {"kv_a_layernorm", "k_pe_rope"}
    assert _tensors(v.path_with(0, "k_proj")["kv"][0]["writes"]) == {"k_rope"}


def test_mla_vs_dsa_and_the_dsa_variants(attention):
    """The bench draws each entry as its prefill runs it
    (`transforms.phase_form`): dense MLA expanded, every DSA entry absorbed.
    So DSA and its declared absorbed form pair node for node; against MLA the
    dense ops it swaps for sparse ones stay paired, the decompression is
    MLA's alone, and the absorbed entries share a zero-cost broadcast into
    the same ports plus two per-head projections. Only the full-indexer
    entries carry the indexer branch; the shared variant reuses another
    layer's top-k, per its declared reads."""
    v = _View(attention)
    _gqa, mla, dsa, absorbed, shared = range(5)
    skel = {v.name(mla, c["members"][mla]) for c in v.skeleton()}
    assert skel == {"x", "qk_score", "softmax", "attn_apply", "o_merge_heads", "o_proj"}
    score = v.cls_named(mla, "qk_score")
    assert score["diff"]["op"] == ["attention_score", "attention_score", "dsa_attention_score",
                                   "dsa_attention_score", "dsa_attention_score"]
    # DSA prefills in its absorbed form: the two entries are one graph.
    assert all(bool(c["members"][dsa]) == bool(c["members"][absorbed])
               for c in v.gd["classes"])
    indexer = v.path_with(dsa, "indexer_logits")
    assert indexer["owners"] == [dsa, absorbed]
    assert v.into(indexer) == {("qk_score", "select")}
    assert _tensors(indexer["kv"][dsa]["writes"]) == {"indexer.key_concat"}
    # MLA's whole key path is its own: the latent it decompresses and the
    # RoPE key it broadcasts per head.
    decompress = v.path_with(mla, "kv_b_proj")
    assert decompress["owners"] == [mla]
    assert _tensors(decompress["kv"][mla]["reads"]) == {"kv_a_layernorm",
                                                         "k_pe_rope"}
    broadcast = v.path_with(absorbed, "k_latent_broadcast")
    assert broadcast["owners"] == [dsa, absorbed, shared]
    assert broadcast["group"] == decompress["group"]
    # The absorbed forms run the stored kv_b_proj as two per-head GEMMs
    # that read it (`module: kv_b_proj`): one inserted path, the tensor and both.
    inserted = v.path_with(absorbed, "q_absorbed_nope")
    assert inserted["owners"] == [dsa, absorbed, shared]
    assert v.names(inserted, absorbed) == {"kv_b_proj", "q_absorbed_nope",
                                           "v_absorbed"}
    low_rank_q = v.path_with(mla, "q_a_proj")
    assert low_rank_q["owners"] == [mla, dsa, absorbed, shared]
    assert v.gd["reads"][shared] == ["dsa_topk"] and v.gd["reads"][dsa] == []
    # The same graph holds the same weights and does the same work.
    dsa_vs_absorbed = GraphDiff.of(attention["diff_entries"][2:4]).to_dict()
    assert all(not any(d.values()) for d in dsa_vs_absorbed["totals"]["delta"])
    # Absorbed reads its cache on the core; dense MLA reads it upstream.
    apply = v.cls_named(mla, "attn_apply")
    assert apply["kv"][absorbed]["reads"] and not apply["kv"][mla]


def test_gated_delta_net_vs_kda_vs_mamba(linear):
    """One recurrence core — a delta rule in GDN and KDA, a selective scan in
    Mamba, paired by their shared operand roles — and the output projection
    are all three share. GDN and KDA share their write strength and the
    gated output; Mamba is one packed path. The decay gates are different
    functions and stay two paths into the same port: GDN's
    −exp(A_log)·softplus(a + dt_bias) against Kimi-K3's lower-bounded
    gate_lower_bound·sigmoid(exp(A_log)·(f + dt_bias))."""
    v = _View(linear)
    gdn, kda, mamba = range(3)
    skel = {v.name(gdn, c["members"][gdn]) for c in v.skeleton()}
    assert skel == {"x", "delta_rule", "out_proj"}
    core = v.cls_named(gdn, "delta_rule")
    assert core["diff"]["op"] == ["gated_delta_rule", "gated_delta_rule", "mamba2_scan"]
    assert all(k["state_bytes_per_request"] for k in core["kv"])
    gdn_gate, kda_gate = v.path_with(gdn, "g_gate"), v.path_with(kda, "g_decay")
    assert gdn_gate["owners"] == [gdn] and kda_gate["owners"] == [kda]
    assert v.into(gdn_gate) == v.into(kda_gate) == {("delta_rule", "decay")}
    write = v.path_with(gdn, "beta")
    assert write["owners"] == [gdn, kda] and v.into(write) == {("delta_rule", "write")}
    packed = v.path_with(mamba, "in_proj")
    # The gated output norm is one `norm` call (Zamba2RMSNormGated), not a
    # norm node plus a separate gate multiply.
    assert packed["owners"] == [mamba] and len(packed["classes"]) == 14
    assert {p for _c, p in v.into(packed)} >= {"q", "k", "v", "decay"}


def test_a_custom_op_core_pairs_through_its_declaration(loader):
    """GLA's core is a spec-scoped custom op that declares its operands'
    roles (q, k, v), so it anchors the component just as a registered score
    does. Against GDN the two recurrences pair as substitutes, the V paths
    that feed them pair backwards from there, and the rest of each body is
    an alternative. Against itself with other dims, and with every name
    changed, it pairs node for node: the custom op's identity is its
    declaration, found through the spec's `custom_ops`."""
    gla = "gla_qknorm_partial_rope_sigmoid_gated_attention"
    ling = "inclusionAI/Ling-2.6-1T"
    a, _ = _bench_entry(loader, gla, ling, key="gla")
    (core,) = a.anchors()
    assert a.nodes[core]["op"] == "gla_simple_recurrence"
    b, _ = _bench_entry(loader, gla, ling, key="gla-narrow", dims={"n_heads": 32, "n_kv_heads": 32})
    gd = GraphDiff.of([a, b])
    assert all(row[0] == row[1] for row in gd.classes)
    payload = gd.to_dict()
    assert payload["grouping"] == BY_LAYER and payload["skeleton_boxes"] == []
    rec = next(c for c in payload["classes"] if a.nodes[c["members"][0]]["op"] == "gla_simple_recurrence")
    assert [b["basis"] for b in rec["basis"]] == [ANCHOR] and "because" in rec
    renamed = _renamed(b, "r_")
    assert _canon(GraphDiff.of([a, renamed]), {1: renamed.back}) == _canon(gd)
    c, _ = _bench_entry(loader, "gated_delta_net_attention", "Qwen/Qwen3.5-35B-A3B", key="gdn")
    both = GraphDiff.of([a, c]).to_dict()
    skel = [cl for cl in both["classes"] if all(cl["members"])]
    assert [[b["basis"] for b in cl["basis"]] for cl in skel] == [
        [INPUT], [BACKTRACK], [BACKTRACK], [ANCHOR_SUBSTITUTED], [OUTPUT]]
    (cores,) = [cl for cl in skel if cl["basis"][0]["basis"] == ANCHOR_SUBSTITUTED]
    assert (a.nodes[cores["members"][0]]["op"], c.nodes[cores["members"][1]]["op"]) \
        == ("gla_simple_recurrence", "gated_delta_rule")
    assert all(len(g["paths"]) == 2 for g in both["groups"])


# --- reads no names -----------------------------------------------------------

def _renamed(e: Entry, prefix: str) -> Entry:
    """The same entry with every node id, label, tensor name, component name
    (the id prefix) and custom op name replaced; `.back` maps new ids to old."""
    ids = {nid: f"{prefix}{k}" for k, nid in enumerate(e.nodes)}
    ops: dict = {}
    fresh = itertools.count()

    def node(n, top):
        m = dict(n)
        m["id"] = ids[n["id"]] if top else f"{prefix}child{next(fresh)}"
        m["label"] = m["tensor_name"] = m["id"]
        m["inputs"] = [ids.get(i, i) for i in n.get("inputs") or []]
        declaration = n.get("declaration") or {}
        if isinstance(declaration.get("identity"), str):
            m["op"] = ops.setdefault(n["op"], f"{prefix}op{len(ops)}")
        if n.get("children"):
            m["children"] = [node(c, False) for c in n["children"]]
        return m

    nodes = [node(n, True) for n in e.nodes.values()]
    l1 = dict(e.l1)
    if e.has_l1():
        l1["node_box"] = {ids[k]: v for k, v in e.l1["node_box"].items()}
    out = Entry(key=f"{prefix}{e.key}", block_graph={"nodes": nodes, "edges": []}, l1=l1,
                dims=e.dims,
                inputs=[ids[s] for s in e.sources], outputs=[ids[s] for s in e.sinks],
                reads=e.reads)
    out.back = {v: k for k, v in ids.items()}
    return out


def _canon(gd, backs=None):
    """Classes by entry position, members mapped back through `backs`."""
    backs = backs or {}
    return frozenset(frozenset((i, backs.get(i, {}).get(nid, nid))
                               for i, nid in enumerate(row) if nid) for row in gd.classes)


def test_renaming_every_name_changes_no_pairing(corpus):
    names = sorted(corpus)
    rng = random.Random(1)
    pairs = [tuple(rng.sample(names, 2)) for _ in range(25)]
    pairs.append(("gla_qknorm_partial_rope_sigmoid_gated_attention", "gated_delta_net_attention"))
    for x, y in pairs:
        a, b = corpus[x], corpus[y]
        ra, rb = _renamed(a, "a_"), _renamed(b, "b_")
        want = Alignment.of(a, b)
        got = Alignment.of(ra, rb)
        assert {ra.back[u]: rb.back[v] for u, v in got.pairs.items()} == want.pairs, (x, y)
        assert {ra.back[u]: w for u, w in got.basis.items()} == want.basis, (x, y)


def test_the_diff_reads_no_component_op_or_node_names():
    """Every string literal is a payload key, a basis or grouping word, a
    node-dict key or a message — never a name to match against. Listing an
    op here (a view op, say) is what this test exists to stop."""
    src = (ROOT / "backend/views/diff.py").read_text()
    allowed = {
        # basis / grouping words, and the input pseudo-op
        INPUT, OUTPUT, ANCHOR, ANCHOR_SUBSTITUTED, BACKTRACK, FORWARD, SAME_SLOT,
        BY_BOX, BY_LAYER,
        # node-dict and L1 keys read
        "id", "op", "children", "tensor_name", "tensor_in", "tensor_out", "params",
        "flops", "weight_dtype", "weight_bytes", "kv", "nodes", "available",
        "node_box", "boxes", "rank", "label", "totals",
        "flops_symbolic", "params_symbolic", "tensor_in_symbolic", "tensor_out_symbolic",
        # kv payload keys read
        "writes", "reads", "tensor", "write_bytes_per_token", "read_bytes_per_token",
        "state_bytes_per_request",
        # payload keys written
        "reason", "entries", "grouping", "classes", "edges", "paths", "groups",
        "skeleton_boxes", "members", "basis", "pair", "diff", "because", "dims", "declaration",
        "view", "partition", "src", "dst", "port", "owners", "from", "into", "class",
        "chain", "sums", "group", "present", "delta", "regrouped", "output_kind",
        # declaration-card keys
        "identity", "sigma", "roles",
        # CustomOpSpec.layout_family's closed vocabulary
        "geometry",
        # id prefixes
        "c", "p", "g", "",
    }
    tree = ast.parse(src)
    doc_nodes = {id(n.body[0].value) for n in ast.walk(tree)
                 if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
                 and n.body and isinstance(n.body[0], ast.Expr)
                 and isinstance(n.body[0].value, ast.Constant)}
    strays = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in doc_nodes and node.value not in allowed
                and not any(ch in node.value for ch in " {")):
            strays.add(node.value)
    assert not strays, strays


# --- refusals ------------------------------------------------------------------

def test_a_node_the_diff_cannot_declare_refuses_it():
    """A source the renderer draws without a declaration card (a stream read
    outside the Playground's isolation) is named in the refusal."""
    with pytest.raises(DiffRefused, match="stream_in"):
        Entry(key="broken", block_graph={"nodes": [
            {"id": "x", "op": "input", "inputs": []},
            {"id": "y", "op": "stream_in", "inputs": []}], "edges": []},
            l1={"available": False}, dims={}, inputs=["x"],
            outputs=[], reads=[])
