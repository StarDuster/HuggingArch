"""Multi-output primitive op support.

Covers the four pieces of multi-output ops:

  - `OpSpec.weight_tensor_suffix` controls leaf tensor naming
    (`""` for `parameter_vector` / `linear_param` nn.Parameter-style
    storage, `".weight"` for nn.Module-wrapped Linear/Embedding/etc.)
  - `OpSpec.secondary_outputs` registers per-port shape templates on
    multi-output primitives like `hc_split_sinkhorn`
  - `parse_input_ref` resolves `<node>.<port>` refs against the
    surrounding graph; falls back to bare-ref when the suffix isn't
    a declared port
  - `Analysis.shapes` records named ports for multi-output primitives,
    and downstream evaluation follows port refs to the right shape

Plus the two new primitives themselves: `linear_param` (single-output
GEMM with bare checkpoint name) and `hc_split_sinkhorn` (V4-Pro
hyper-connection 3-way split, the first multi-output primitive).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.arch.axis_propagation import compute_shape_out
from backend.arch.ops import OP_REGISTRY
from backend.arch.schema import (
    BlockGraph,
    GraphNode,
    parse_input_ref,
)


class TestNewPrimitivesRegistered(unittest.TestCase):
    """`linear_param` and `hc_split_sinkhorn` are present in the
    registry with the expected fields populated."""

    def test_linear_param_is_registered(self):
        op = OP_REGISTRY.get("linear_param")
        self.assertIsNotNone(op, "linear_param must be registered")
        self.assertEqual(op.weight_tensor_suffix, "")
        self.assertEqual(op.secondary_outputs, {})
        self.assertEqual(op.shape_in, ("B", "s_ctx", "in"))
        self.assertEqual(op.shape_out, ("B", "s_ctx", "out"))

    def test_linear_param_flops_match_linear_no_bias(self):
        # Same GEMM cost as `linear` without bias: 2·B·S·in·out.
        lp = OP_REGISTRY["linear_param"]
        ln = OP_REGISTRY["linear"]
        d = {"B": 2, "s_ctx": 16, "in": 32, "out": 64, "bias": 0}
        self.assertEqual(int(lp.flops_fn(d)), int(ln.flops_fn(d)))
        self.assertEqual(int(lp.params_fn(d)), int(ln.params_fn(d)))

    def test_hc_split_sinkhorn_has_three_ports(self):
        op = OP_REGISTRY.get("hc_split_sinkhorn")
        self.assertIsNotNone(op)
        # Main `out` port → `pre`. Secondary ports → `post`, `comb`.
        self.assertEqual(set(op.secondary_outputs), {"post", "comb"})

    def test_hc_split_sinkhorn_main_and_secondary_shapes(self):
        op = OP_REGISTRY["hc_split_sinkhorn"]
        shape_in = [(2, 16, 24), (3,), (24,)]   # B=2 S=16 Mc=24
        dims = {"B": 2, "s_ctx": 16, "Mc": 24, "n_hidden_copies": 4}
        # Main: pre [B, S, Hc]
        out = compute_shape_out(op, shape_in, dims)
        self.assertEqual(out, (2, 16, 4))
        self.assertEqual(op.secondary_outputs["post"],
                         ("B", "s_ctx", "n_hidden_copies"))
        self.assertEqual(op.secondary_outputs["comb"],
                         ("B", "s_ctx", "n_hidden_copies", "n_hidden_copies"))

    def test_mamba2_scan_is_registered(self):
        op = OP_REGISTRY.get("mamba2_scan")
        self.assertIsNotNone(op, "mamba2_scan must be registered")
        # Zero-param recurrence: SSM state lives outside the param budget.
        d = {"B": 1, "s_ctx": 1, "n_ssm_heads": 1, "d_ssm_head": 1, "d_ssm_state": 1, "n_ssm_groups": 1}
        self.assertEqual(int(op.params_fn(d)), 0)
        # required_dims surfaces the SSM-specific symbols call sites must bind.
        self.assertEqual(set(op.required_dims), {"n_ssm_heads", "d_ssm_head", "d_ssm_state", "n_ssm_groups"})
        # 7 positional inputs mirror the mamba_chunk_scan_combined call site:
        # (hidden_states, dt, A_log, B, C, D, dt_bias).
        self.assertEqual(len(op.shape_in), 7)

    def test_mamba2_scan_output_shape_mirrors_hidden_states(self):
        op = OP_REGISTRY["mamba2_scan"]
        # hidden_states is positional arg 0 — output mirrors its full shape.
        shape_in = [
            (2, 16, 32, 8),          # hidden_states [B, S, Nmh, Dmh]
            (2, 16, 32),             # dt            [B, S, Nmh]
            (32,),                   # A_log         [Nmh]
            (2, 16, 4, 64),          # B             [B, S, Nmg, Dss]
            (2, 16, 4, 64),          # C             [B, S, Nmg, Dss]
            (32,),                   # D             [Nmh]
            (32,),                   # dt_bias       [Nmh]
        ]
        dims = {"B": 2, "s_ctx": 16, "n_ssm_heads": 32, "d_ssm_head": 8, "d_ssm_state": 64, "n_ssm_groups": 4}
        out = compute_shape_out(op, shape_in, dims)
        self.assertEqual(out, (2, 16, 32, 8))

    def test_mamba2_scan_flops_scale_with_state_size(self):
        # The chunk-scan kernel's dominant cost is B·S·Nmh·Dmh·Dss
        # (state-update + output-read matmul terms summed). Doubling
        # Dss doubles the matmul term but leaves the D⊙x skip term
        # alone — total grows but less than 2×.
        op = OP_REGISTRY["mamba2_scan"]
        base = {"B": 1, "s_ctx": 64, "n_ssm_heads": 8, "d_ssm_head": 16, "d_ssm_state": 32, "n_ssm_groups": 2}
        doubled = {**base, "d_ssm_state": 64}
        f_base = int(op.flops_fn(base))
        f_doubled = int(op.flops_fn(doubled))
        # Strict bounds: more compute, but not quite 2× (the D⊙x skip stays put).
        self.assertGreater(f_doubled, f_base)
        self.assertLess(f_doubled, 2 * f_base)
        # Sanity: zeroing Nmh zeros total FLOPs.
        zero = {**base, "n_ssm_heads": 0}
        self.assertEqual(int(op.flops_fn(zero)), 0)

    def test_head_repeat_expands_existing_head_axis(self):
        op = OP_REGISTRY["head_repeat"]
        shape_in = [(2, 16, 7, 128)]
        self.assertEqual(
            compute_shape_out(op, shape_in, {"n_heads": 48}),
            (2, 48, 7, 128),
        )
        self.assertEqual(int(op.flops_fn({})), 0)
        self.assertEqual(int(op.params_fn({})), 0)


class TestParseInputRef(unittest.TestCase):
    """`<node>.<port>` is interpreted as a port ref ONLY when the
    parent node's op declares the port. Anything else falls back to
    bare-ref interpretation, leaving the dotted name intact."""

    def setUp(self):
        self.graph = {
            "splitter": GraphNode(
                op="hc_split_sinkhorn",
                inputs=["mixes", "scale", "base"],
                dims={"Mc": "Mc", "n_hidden_copies": "n_hidden_copies"},
            ),
            "mixes": GraphNode(op="parameter_vector", dim="Mc"),
            "scale": GraphNode(op="parameter_vector", dim="3"),
            "base": GraphNode(op="parameter_vector", dim="Mc"),
            "consumer": GraphNode(op="add", inputs=["splitter.post"], dim="n_hidden_copies"),
        }

    def test_bare_ref_resolves_to_out_port(self):
        self.assertEqual(parse_input_ref("splitter", self.graph),
                         ("splitter", "out"))

    def test_declared_port_ref(self):
        self.assertEqual(parse_input_ref("splitter.post", self.graph),
                         ("splitter", "post"))
        self.assertEqual(parse_input_ref("splitter.comb", self.graph),
                         ("splitter", "comb"))

    def test_unknown_port_falls_back_to_bare_ref(self):
        # `splitter.bogus` — bogus isn't a declared port, so the whole
        # string is treated as a node name (which then fails connectivity
        # validation upstream, surfacing the typo).
        self.assertEqual(parse_input_ref("splitter.bogus", self.graph),
                         ("splitter.bogus", "out"))

    def test_port_ref_against_single_output_op_is_bare(self):
        # `consumer.foo`: consumer is `add` (no secondary_outputs), so
        # `consumer.foo` is treated as a bare ref.
        self.assertEqual(parse_input_ref("consumer.foo", self.graph),
                         ("consumer.foo", "out"))

    def test_unknown_parent_node_is_bare(self):
        self.assertEqual(parse_input_ref("unknown.port", self.graph),
                         ("unknown.port", "out"))


class TestBlockGraphConnectivityWithPorts(unittest.TestCase):
    """`validate_graph_connectivity` accepts port refs whose parent
    node is in the graph; rejects the rest."""

    def _nodes(self):
        return {
            "splitter": GraphNode(
                op="hc_split_sinkhorn",
                inputs=["mixes", "scale", "base"],
                dims={"Mc": "Mc", "n_hidden_copies": "n_hidden_copies"},
            ),
            "mixes": GraphNode(op="parameter_vector", dim="Mc"),
            "scale": GraphNode(op="parameter_vector", dim="3"),
            "base": GraphNode(op="parameter_vector", dim="Mc"),
            "consumer": GraphNode(op="add",
                                  inputs=["splitter.post", "scale"],
                                  dim="n_hidden_copies"),
        }

    def test_valid_port_refs_pass_validation(self):
        g = BlockGraph(inputs=["x"], graph=self._nodes(),
                       outputs=["consumer"])
        # round-trip the consumer node's parsed refs
        parsed = [parse_input_ref(s, g.graph)
                  for s in g.graph["consumer"].source_names]
        self.assertEqual(parsed, [("splitter", "post"), ("scale", "out")])

    def test_unknown_port_ref_rejected(self):
        nodes = self._nodes()
        nodes["bad"] = GraphNode(op="add",
                                 inputs=["nonexistent.port"],
                                 dim="n_hidden_copies")
        with self.assertRaisesRegex(Exception, "unknown input"):
            BlockGraph(inputs=["x"], graph=nodes, outputs=["consumer"])


class TestRmsnormUnscaled(unittest.TestCase):
    """`rmsnorm_unscaled` is a separate primitive — same forward as
    `rmsnorm` minus the trailing `· γ` scale. Used by V4-Pro
    Hyper-Connections (`Block.hc_pre`, `ParallelHead.hc_head`)."""

    def test_registered_with_no_weight(self):
        op = OP_REGISTRY.get("rmsnorm_unscaled")
        self.assertIsNotNone(op)
        # Same FLOPs as rmsnorm (forward is identical except the gain mul).
        self.assertEqual(int(op.flops_fn({"B": 1, "s_ctx": 8, "dim": 128})),
                         int(OP_REGISTRY["rmsnorm"].flops_fn(
                             {"B": 1, "s_ctx": 8, "dim": 128})))
        # ZERO params and no weight_shape — spec_tree must not emit
        # a `.weight` LeafTensor for this op.
        self.assertEqual(int(op.params_fn({"dim": 128})), 0)
        self.assertFalse(op.has_weight())


class TestLibraryCompoundsLoaded(unittest.TestCase):
    """The 3 HC library compounds load with the expected role + node
    counts. Topology validation is enforced by the loader; tests here
    pin the structure so accidental edits surface."""

    @classmethod
    def setUpClass(cls):
        from backend.arch.loader import SpecLoader
        cls.lib = SpecLoader()._component_lib

    def test_hc_weighted_reduce(self):
        self.assertIn("hc_weighted_reduce", self.lib)
        c = self.lib["hc_weighted_reduce"]
        # Every HC compound declares the residual stream's own role.
        self.assertEqual(c.role, "hyper_connection")
        # 3 graph nodes: broadcast + mul + sum_reduce.
        self.assertEqual(len(c.graph), 3)
        ops = sorted(n.op for n in c.graph.values())
        self.assertEqual(ops, ["broadcast", "mul", "sum_reduce"])

    def test_hc_combine(self):
        self.assertIn("hc_combine", self.lib)
        c = self.lib["hc_combine"]
        self.assertEqual(c.role, "hyper_connection")
        # 5 graph nodes: broadcast×2 + mul×1 + hc_mix×1 + add×1. Branch 2 is the
        # fused einsum `hc_mix` (matches vLLM mhc/torch.py::mhc_post_torch), NOT
        # the eager broadcast→mul→sum_reduce outer product — that materialised
        # a [B,S,Hc,Hc,H] intermediate the fused kernel never forms (the
        # M-NoPwTop bug). See
        # backend/arch/components/library/hc_combine.yaml.
        self.assertEqual(len(c.graph), 5)
        from collections import Counter
        ops = Counter(n.op for n in c.graph.values())
        self.assertEqual(ops, Counter(broadcast=2, mul=1, hc_mix=1, add=1))

    def test_hc_head(self):
        self.assertIn("hc_head", self.lib)
        c = self.lib["hc_head"]
        # A pure-function compound consumed at any scope (model root for
        # V4 main exit, MTP block for the MTP head); it carries the
        # residual stream's role. The 3 weights live at the caller's
        # scope, not inside.
        self.assertEqual(c.role, "hyper_connection")
        # 9 nodes: flat + flat_norm + mixes (linear_external_weight) +
        # scaled + biased + pre, then the weighted reduce written inline
        # (w_expanded + weighted + reduced) so a fusion can reach it.
        # Zero own params — the 3 weights are consumed via the 4-input
        # signature `(x, fn, scale, base)`.
        self.assertEqual(len(c.graph), 9)
        self.assertEqual(list(c.inputs), ["x", "fn", "scale", "base"])
        # Ops that OWN their weight on disk (params count > 0).
        # linear_external_weight consumes an external weight but
        # doesn't own one — params=0, so it's not in this set.
        weight_owning_ops = {"linear", "linear_param",
                             "parameter_vector", "parameter_tensor"}
        own_weights = [n for n in c.graph.values() if n.op in weight_owning_ops]
        self.assertEqual(len(own_weights), 0,
                         "hc_head should be a pure function — no own params")


class TestV4ProHCEndToEnd(unittest.TestCase):
    """End-to-end check that V4-Pro's Hyper-Connections build all the
    way up: (a) the 6 per-block hc_* banks are no longer orphans —
    every parameter_vector is consumed by `hc_split_sinkhorn`; (b)
    the model-root `hc_head_fn` / `hc_head_base` / `hc_head_scale`
    land at bare top-level paths via the `hc_head_block` singleton
    template + primary fold; (c) byte total stays the V4-Pro
    byte-perfect baseline."""

    @classmethod
    def setUpClass(cls):
        import json, pathlib
        from backend.arch.loader import SpecLoader
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        loader = SpecLoader()
        spec = loader._specs['deepseek_v4']
        config = json.loads(pathlib.Path(
            'backend/arch/snapshots/deepseek-ai__DeepSeek-V4-Pro.json'
        ).read_text())['config']
        cls.resolved = loader.resolve(spec, config)
        cls.materialized = list(
            Deployment(Program.of(cls.resolved)).at(
                batch=1, input_tokens=8,
            ).analyze().iter_checkpoint()
        )

    def test_dsv4_block_has_no_orphan_parameter_vectors(self):
        from backend.rendering._node_build import NodeBuilder
        nb = NodeBuilder(self.resolved, {}, {})
        # Pick the most heavily-used block specialization. dsv4_block
        # hardcodes rmsnorm internally and doesn't declare a `norm`
        # param slot, so the synth name only encodes attn + ffn.
        tname = ('dsv4_block__attn-dsv4_hca_attention_ffn-'
                 'dsv4_moe_sigmoid')
        nodes = nb.build_graph(tname)
        ids = {n['id'] for n in nodes}
        consumed = set()
        for n in nodes:
            for inp in n.get('inputs', []):
                head = inp.split('.')[0] if '.' in inp else inp
                if head in ids: consumed.add(head)
                if inp in ids: consumed.add(inp)
        sources = [n['id'] for n in nodes if n.get('is_source')]
        orphans = [s for s in sources if s not in consumed]
        self.assertEqual(orphans, [],
                         f"dsv4_block has orphan parameter_vectors: {orphans}")
        # After Option B refactor, all 6 HC nn.Parameters live at block
        # scope (mirroring V4-Pro source `Block.__init__`):
        #   hc_attn_fn / hc_ffn_fn    — parameter_tensor (2-D weights)
        #   hc_attn_base / hc_ffn_base / hc_attn_scale / hc_ffn_scale
        #     — parameter_vector (1-D)
        # All 6 must be consumed (by hc_pre via input list) — no orphans.
        # Block-internal renderer-dict node ids carry the template prefix
        # (`{template}|{display_path}`) since the keyspace was unified to
        # aggregation_id end-to-end.
        self.assertEqual(
            sorted(sources),
            [f"{tname}|hc_attn_base", f"{tname}|hc_attn_fn",
             f"{tname}|hc_attn_scale", f"{tname}|hc_ffn_base",
             f"{tname}|hc_ffn_fn", f"{tname}|hc_ffn_scale"],
        )

    def test_main_hc_head_lands_at_bare_top_level(self):
        """The 3 main-model hc_head_* tensors must be at the model root
        (no `layers.X.` prefix). Each is a top-level model layer
        (`parameter_tensor` for `hc_head_fn`, `parameter_vector` for
        the two scalars), and the `hc_head` template invocation reads
        them as explicit input refs — mirroring V4-Pro source's
        `ParallelHead.hc_head(x, hc_head_fn, hc_head_scale, hc_head_base)`
        functional call."""
        bare_names = {t.name for t in self.materialized
                      if t.name in ("hc_head_fn", "hc_head_base", "hc_head_scale")}
        self.assertEqual(
            bare_names, {"hc_head_fn", "hc_head_base", "hc_head_scale"},
            "Main hc_head_* tensors should land bare; got: "
            f"{[t.name for t in self.materialized if 'hc_head' in t.name]}",
        )
        # parameter_tensor weight should be 2D (Hc, Hc·H).
        hc_fn = next(t for t in self.materialized if t.name == "hc_head_fn")
        self.assertEqual(len(hc_fn.shape), 2,
                         f"hc_head_fn shape should be 2D from parameter_tensor; "
                         f"got {hc_fn.shape}")

    def test_byte_perfect_baseline(self):
        total = sum(t.storage_bytes * t.count
                    for t in self.materialized)
        self.assertEqual(
            # ac62d8b9 modelled the 31 HCA layers' compressor in the
            # checkpoint's no-overlap form: wkv/wgate [1024,7168]→[512,7168]
            # BF16 (−14,680,064 B) and ape [4,1024]→[128,512] F32
            # (+245,760 B) per layer, −447,463,424 B over 31 layers. The hub
            # shard headers and inference/model.py (`coff = 1 + (ratio == 4)`)
            # both carry the new shape, e.g. layers.3 vs layers.2.
            # 7528955b added the E8M0 128x128 tile scales the file stores
            # beside the FP8 shared experts (62 layers incl. MTP x 4,032 B)
            # and MTP e_proj / h_proj (2 x 3,136 B): +256,256 B.
            total, 864_704_792_696,
            f"V4-Pro spec_storage_bytes drifted; got {total:,}",
        )

    def test_mtp_block_has_no_orphans_and_correct_naming(self):
        """MTP block: hc_attn_*/hc_ffn_*/hc_head_* params must all be
        wired (no orphans), and the MTP block's leaves land under the
        `mtp.X.` checkpoint scope — NOT `layers.X.` — because the MTP
        group is an aux stack with its own checkpoint namespace
        (`mtp.0.hc_head_fn` in the V4-Pro index, not
        `layers.0.hc_head_fn`). hc_head_fn is a 2D linear_param."""
        from backend.rendering._node_build import NodeBuilder
        nb = NodeBuilder(self.resolved, {}, {})
        nodes = nb.build_graph('dsv4_mtp_block')
        ids = {n['id'] for n in nodes}
        consumed = set()
        for n in nodes:
            for inp in n.get('inputs', []):
                head = inp.split('.')[0] if '.' in inp else inp
                if head in ids: consumed.add(head)
                if inp in ids: consumed.add(inp)
        sources = [n['id'] for n in nodes if n.get('is_source')]
        orphans = [s for s in sources if s not in consumed]
        self.assertEqual(
            orphans, [],
            f"dsv4_mtp_block has orphan parameter_vectors: {orphans}",
        )
        # MTP-private hc_head_fn must be 2D (Hc, Hc·H) not the prior
        # 1D parameter_vector form, and land under the `mtp.X.` aux
        # stack prefix.
        mtp_hc_head_fn = next(
            (t for t in self.materialized if t.name == "mtp.X.hc_head_fn"),
            None,
        )
        self.assertIsNotNone(mtp_hc_head_fn, "MTP hc_head_fn missing")
        self.assertEqual(
            len(mtp_hc_head_fn.shape), 2,
            f"MTP hc_head_fn should be 2D linear_param; got "
            f"{mtp_hc_head_fn.shape}",
        )


def _iter_edges(block_graph):
    """Yield every edge in a `build_block_graph_data` result, recursing
    into nested `child_edges` so component-internal edges are covered."""
    for block in block_graph.values():
        nodes = block["nodes"] if isinstance(block, dict) else block
        for e in (block.get("edges") or []) if isinstance(block, dict) else []:
            yield e

        def _walk(ns):
            for n in ns:
                for e in (n.get("child_edges") or []):
                    yield e
                if n.get("children"):
                    yield from _walk(n["children"])
        yield from _walk(nodes)


def _iter_nodes(block_graph):
    for block in block_graph.values():
        nodes = block["nodes"] if isinstance(block, dict) else block

        def _walk(ns):
            for n in ns:
                yield n
                if n.get("children"):
                    yield from _walk(n["children"])
        yield from _walk(nodes)


class TestComputeGraphRendersWithoutShapelessEdges(unittest.TestCase):
    """Structural invariant: the rendered compute graph must keep the
    backend-expressed dataflow in lockstep with the frontend render —
    NO edge may carry a `None` shape, and every secondary-port edge
    (hc_pre `.post` / `.comb`) must thread its true per-port shape into
    the consumer, not fall back to the producer's main output.

    Regression guard for the V4-Pro hc_post rendering bug: the engine's
    component-invocation alias loop ignored secondary ports, so the
    inner `hc_combine` `post_exp` / `comb_exp` broadcasts received no
    shape and the expanded view drew shapeless edges into `hc_post`.
    Plus the two unrenderable ops it surfaced — the stale `activation`
    (split into per-fn ops) and the unregistered `hash_route`."""

    @classmethod
    def setUpClass(cls):
        import json, pathlib
        from backend.arch.loader import SpecLoader
        from backend.rendering.builders import build_block_graph_data
        loader = SpecLoader()
        spec = loader._specs['deepseek_v4']
        config = json.loads(pathlib.Path(
            'backend/arch/snapshots/deepseek-ai__DeepSeek-V4-Pro.json'
        ).read_text())['config']
        resolved = loader.resolve(spec, config)
        # B=8, S=1024 so per-port shapes are concrete integers.
        cls.bg, _ = build_block_graph_data(resolved, batch_size=8, seq_len=1024)

    def test_no_shapeless_edges_anywhere(self):
        """Every edge whose source carries a shape must label that shape.
        A `None` shape == an edge the backend graph expressed but the
        render can't draw."""
        shapeless = [
            (e.get("src", "").split("|")[-1],
             e.get("dst", "").split("|")[-1],
             e.get("src_port"))
            for e in _iter_edges(self.bg)
            if e.get("shape") is None
        ]
        self.assertEqual(
            shapeless, [],
            f"{len(shapeless)} shapeless edge(s) in the rendered compute "
            f"graph — backend dataflow that the frontend can't render: "
            f"{shapeless[:10]}",
        )

    def test_hc_post_secondary_port_edges_carry_per_port_shapes(self):
        """The `hc_attn_pre.post` / `.comb` edges into `hc_post` must
        show the sinkhorn per-port shapes ([B,S,Hc] / [B,S,Hc,Hc]),
        NOT the main reduced [B,S,H]."""
        by_port = {}
        for e in _iter_edges(self.bg):
            if e.get("dst", "").endswith("|hc_attn_post") and e.get("src_port"):
                by_port[e["src_port"]] = e.get("shape")
        # Hc=4, H=7168 for V4-Pro at this config.
        self.assertEqual(by_port.get("post"), "[8,1024,4]")
        self.assertEqual(by_port.get("comb"), "[8,1024,4,4]")

    def test_hc_combine_inner_broadcasts_see_port_shapes(self):
        """Inside the expanded `hc_combine`, the consumers of the
        secondary ports must receive the port tensor — the inner consumer
        shape that was `None` before the alias fix. `post` feeds the
        `post_exp` broadcast ([B,S,Hc]); `comb` feeds the fused `comb_red`
        (`hc_mix`) op as its first input ([B,S,Hc,Hc]) — the eager
        `comb_exp` broadcast was removed with the hc_mix fusion (matches
        vLLM `mhc/torch.py::mhc_post_torch`)."""
        seen = {}
        for n in _iter_nodes(self.bg):
            nid = n.get("id", "")
            if nid.endswith("post_exp__hc_combine"):
                seen.setdefault("post_exp", n.get("tensor_in"))
            elif nid.endswith("comb_red__hc_combine"):
                seen.setdefault("comb_red", n.get("tensor_in"))
        self.assertEqual(seen.get("post_exp"), "[8,1024,4]")
        # Both operands matter: the caller-side actual name `x` collides
        # with hc_combine's first formal, but must still resolve to the
        # rank-4 residual rather than the fresh rank-3 branch.
        self.assertEqual(
            seen.get("comb_red"),
            "[8,1024,4,4], [8,1024,4,7168]",
        )


class TestUnknownOpRejected(unittest.TestCase):
    """The engine fails loudly on an op that resolves to no primitive /
    component / custom_op, instead of emitting a silent shapeless no-op
    node. This is the root-level backstop that keeps a dropped or typo'd
    op from shipping as an unrenderable edge."""

    def _resolved_with_op(self, op_name):
        from backend.arch.schema import ArchSpec, BlockGraph, GraphNode, ModelLayer
        from backend.arch.loader import ResolvedSpec
        block = BlockGraph(
            inputs=["x"],
            graph={"n": GraphNode(op=op_name, inputs=["x"], dim="d_model")},
            outputs=["n"],
        )
        spec = ArchSpec(
            name="t", model_id="t", hf_class="t",
            model_inputs={"input_ids": {"kind": "tokens"}},
            dims={"d_model": "config.hidden_size"},
            model=[
                ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                           boundary_role="input_embedding",
                           dims={"n_vocab": "d_model", "d_model": "d_model"}),
                ModelLayer(name="layers", template="block", repeat="1",
                           inputs=["embed"]),
            ],
            blocks={"block": block},
        )
        return ResolvedSpec(spec, {"d_model": 8},
                            {"hidden_size": 8, "torch_dtype": "bfloat16"}, {})

    def test_unknown_op_raises_in_walk(self):
        from backend.arch.eval.projection import project_block
        resolved = self._resolved_with_op("totally_not_an_op")
        # S5c: an unregistered op is rejected at build time by its declaration —
        # "neither a registered primitive, a custom op of this spec, nor a component".
        with self.assertRaisesRegex(Exception, r"is neither a registered primitive"):
            project_block(resolved, "block", batch=1, input_tokens=1)

    def test_known_primitive_does_not_raise(self):
        # Sanity: a real op (rmsnorm) on the same harness evaluates fine.
        from backend.arch.eval.projection import project_block
        resolved = self._resolved_with_op("rmsnorm")
        project_block(resolved, "block", batch=1, input_tokens=1)


class TestV3MultiOutputStates(unittest.TestCase):
    def test_hc_ports_publish_distinct_states_in_the_single_walk(self):
        import json
        from backend.arch.loader import SpecLoader
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.step_eval import evaluate_step

        config = json.loads((
            PROJECT_ROOT / "backend" / "arch" / "snapshots" /
            "deepseek-ai__DeepSeek-V4-Pro.json"
        ).read_text())["config"]
        resolved = SpecLoader().match(config, model_id="deepseek-ai/DeepSeek-V4-Pro")
        self.assertIsNotNone(resolved)
        step = evaluate_step(
            resolved,
            ParallelConfig(
                tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8),
            phase="prefill", batch=1, input_tokens=128, output_tokens=0,
        )
        analysis = step.analysis
        expected = {
            (uid, port)
            for uid, shape in analysis.shapes.values.items()
            for port in ("out", *shape.ports)
        }
        self.assertEqual(set(analysis.port_states), expected)
        secondary = {
            (analysis.program.by_uid(uid).op, port)
            for uid, port in analysis.port_states if port != "out"
        }
        self.assertEqual(
            secondary,
            {("hc_split_sinkhorn", "post"),
             ("hc_split_sinkhorn", "comb"),
             ("hc_pre", "post"),
             ("hc_pre", "comb")},
        )
        self.assertEqual(len(analysis.port_states) - len(analysis.states.values), 40)


class TestRequiredDimsValidator(unittest.TestCase):
    """OpSpec.required_dims declares the dim keys an op's flops/params
    read. BlockGraph's validator checks call-site-local keys
    (`in`, `out`, `dim`) at load time and rejects with a friendly
    error — turning runtime KeyError into load-time validation."""

    def test_linear_rejects_missing_in_out(self):
        with self.assertRaisesRegex(Exception, r"missing required dim binding\(s\): \['in', 'out'\]"):
            BlockGraph(
                inputs=["x"],
                graph={"lin": GraphNode(op="linear", inputs=["x"])},
                outputs=["lin"],
            )

    def test_sum_reduce_rejects_missing_in(self):
        # The bug surfaced during V4-Pro hc_combine debugging: sum_reduce
        # silently failed at runtime with KeyError 'in'. Validator now
        # rejects upfront.
        with self.assertRaisesRegex(Exception, r"missing required dim binding\(s\): \['in'\]"):
            BlockGraph(
                inputs=["x"],
                graph={"red": GraphNode(op="sum_reduce", inputs=["x"], axis=2)},
                outputs=["red"],
            )

    def test_rmsnorm_accepts_dim_shorthand(self):
        # `dim:` shorthand should satisfy the `dim` required key.
        BlockGraph(
            inputs=["x"],
            graph={"n": GraphNode(op="rmsnorm", inputs=["x"], dim="d_model")},
            outputs=["n"],
        )

    def test_global_dims_not_enforced_at_blockgraph_level(self):
        # `permute` declares required_dims=("d_model",) but H is a spec-global
        # dim — the BlockGraph validator can't see globals, so it skips
        # the check for non-local keys (anything not in {in, out, dim}).
        # Should NOT raise even though `H` isn't bound on the call site.
        BlockGraph(
            inputs=["x"],
            graph={"d": GraphNode(op="permute", inputs=["x"])},
            outputs=["d"],
        )

    def test_linear_param_required_dims_match_linear(self):
        # linear_param: same FLOPs/params signature minus bias → same
        # required_dims as linear.
        from backend.arch.ops import OP_REGISTRY
        self.assertEqual(
            set(OP_REGISTRY["linear_param"].required_dims),
            set(OP_REGISTRY["linear"].required_dims),
        )


class TestCustomOpSpecExtensions(unittest.TestCase):
    """`CustomOpSpec` (spec-scoped, declared in YAML under
    `custom_ops:`) gained the same three knobs as registered
    primitives: `secondary_outputs` (multi-output ports),
    `weight_tensor_suffix` (nn.Parameter-style storage), and
    `required_dims` (load-time call-site checks). Spec writers can
    now declare V4-Pro-class novel kernels without touching ops.py."""

    def test_custom_op_secondary_outputs_field_exists(self):
        from backend.arch.schema import CustomOpSpec
        cop = CustomOpSpec(
            flops="0", params="0",
            signature={"x": ["B", "s_ctx", "Mc"]},
            couples_on=[],
            secondary_outputs={
                "post": ["B", "s_ctx", "n_hidden_copies"],
                "comb": ["B", "s_ctx", "n_hidden_copies", "n_hidden_copies"],
            },
            shape_kind="axis_replace", out_key="n_hidden_copies",
            description="test",
        )
        self.assertEqual(set(cop.secondary_outputs), {"post", "comb"})

    def test_custom_op_weight_tensor_suffix_default_and_bare(self):
        from backend.arch.schema import CustomOpSpec
        sig = dict(signature={"x": ["B", "s_ctx"]}, couples_on=[],
                   shape_kind="preserve")
        # Default `.weight`
        default = CustomOpSpec(flops="0", params="0", description="t", **sig)
        self.assertEqual(default.weight_tensor_suffix, ".weight")
        # Bare nn.Parameter idiom
        bare = CustomOpSpec(
            flops="0", params="0", weight_tensor_suffix="", description="t",
            **sig,
        )
        self.assertEqual(bare.weight_tensor_suffix, "")

    def test_custom_op_port_ref_disambiguation(self):
        """`<node>.<port>` resolves against custom_ops when passed in."""
        from backend.arch.schema import (
            CustomOpSpec, GraphNode, parse_input_ref,
        )
        custom_ops = {
            "my_split": CustomOpSpec(
                flops="0", params="0",
                signature={"x": ["B", "s_ctx"]},
                couples_on=[],
                secondary_outputs={"a": ["B", "s_ctx"], "b": ["B", "s_ctx"]},
                description="t",
                shape_kind="preserve",
            ),
        }
        graph = {
            "splitter": GraphNode(op="my_split", inputs=["x"]),
            "consumer": GraphNode(op="add", inputs=["splitter.a"], dim="d_model"),
        }
        # With custom_ops bound: port reference resolves.
        self.assertEqual(
            parse_input_ref("splitter.a", graph, custom_ops),
            ("splitter", "a"),
        )
        # Without custom_ops: falls back to bare ref.
        self.assertEqual(
            parse_input_ref("splitter.a", graph),
            ("splitter.a", "out"),
        )
        # Unknown port still falls back to bare.
        self.assertEqual(
            parse_input_ref("splitter.nope", graph, custom_ops),
            ("splitter.nope", "out"),
        )

    def test_archspec_rejects_bogus_custom_op_port(self):
        """ArchSpec-level validator catches typo'd custom_op ports."""
        from backend.arch.schema import (
            ArchSpec, BlockGraph, CustomOpSpec, GraphNode,
        )
        graph = {
            "splitter": GraphNode(
                op="my_split",
                inputs=["x"],
                dims={"in": "d_model"},
            ),
            "consumer": GraphNode(
                op="add",
                inputs=["splitter.podt", "x"],   # typo, real port is `post`
                dim="d_model",
            ),
        }
        block = BlockGraph(inputs=["x"], graph=graph, outputs=["consumer"])
        with self.assertRaisesRegex(Exception, r"references port 'splitter\.podt'"):
            ArchSpec(
                name="t",
                model_id="t",
                hf_class="t",
                dims={"d_model": "1"},
                model=[],
                blocks={"block": block},
                custom_ops={
                    "my_split": CustomOpSpec(
                        flops="0", params="0",
                        signature={"x": ["B", "s_ctx", "d_model"]},
                        couples_on=[],
                        secondary_outputs={"post": ["B", "s_ctx", "d_model"]},
                        description="t",
                        shape_kind="preserve",
                    ),
                },
            )


if __name__ == "__main__":
    unittest.main()
