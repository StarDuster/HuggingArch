"""Axis-identity transfer — positional sharding map per op (DimMap analogue).

Falsification: each op kind must move a sharded axis to the correct OUTPUT
position (or drop it on contraction), tracked by POSITION so a concretized head
axis (`64` not `N`) behaves identically. A wrong permutation, split slot, or
contraction handling breaks a specific case.

Engine: backend/inference/axis_propagation.axis_transfer.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.axis_propagation import (
    axis_transfer,
    dim_map,
    edge_consistency_errors,
    rank_out,
    shape_consistency_errors,
)
from backend.arch.eval.dim import Dim, value_of
from backend.arch.shape_node import ShapeProbe, _Ref as _ShapeRef


def _values(shape):
    return tuple(int(value_of(axis)) for axis in shape)


def _evaluated_rows(result):
    """Primitive graph nodes joined to their semantic Shape layer by UID."""
    analysis = result.analysis
    for node in analysis.program.nodes():
        shape = analysis.shapes.values.get(node.uid)
        if shape is None or shape.out is None:
            continue
        out = [int(value_of(axis)) for axis in shape.out]
        ins = [
            [int(value_of(axis)) for axis in dims]
            for dims in shape.ins
        ]
        yield (
            node.aggregation_id,
            node,
            ins,
            out,
            analysis.node_dims.values.get(node.uid) or node.dims or None,
        )


def _Src(rank):
    return ShapeProbe(out_shape=tuple(range(rank)))


def _Ref(rank):
    return _ShapeRef(src=_Src(rank))


def _Node(rank_out, axes=None, axis=None, in_ranks=(), dims_raw=None,
          shape=None):
    """Minimal ShapeNode stand-in: an output rank (or an explicit `shape`
    override), optional permute axes / replace axis, input ranks (for the
    explicit / arity-sensitive kinds), and optional declared factor names."""
    return ShapeProbe(
        out_shape=tuple(range(rank_out)) if shape is None else shape,
        axes=axes, axis=axis, dims_raw=dims_raw or {},
        inputs_obj=tuple(_Ref(r) for r in in_ranks))


class TestAxisTransfer(unittest.TestCase):
    def test_preserve_identity(self):
        n = _Node(4, in_ranks=(4,))
        self.assertEqual(axis_transfer("rmsnorm", n, 1), 1)
        self.assertEqual(axis_transfer("softmax", n, 1), 1)

    def test_permute_follows_axes(self):
        # transpose [0,2,1,3]: out[j] comes from in[axes[j]]; input axis 2 lands
        # at output 1, input 1 at output 2.
        n = _Node(4, axes=[0, 2, 1, 3], in_ranks=(4,))
        self.assertEqual(axis_transfer("transpose", n, 2), 1)
        self.assertEqual(axis_transfer("transpose", n, 1), 2)

    def test_contract_drops_K_keeps_batch(self):
        # attention_apply: [B,N,Sq,Lq]·[B,N,Lq,D] → [B,N,Sq,D]. Head N (1) passes;
        # the last axis of operand 0 (Lq, contracted) is dropped.
        n = _Node(4, in_ranks=(4, 4))
        self.assertEqual(axis_transfer("attention_apply", n, 1), 1)
        self.assertIsNone(axis_transfer("attention_apply", n, 3))

    def test_axis_replace_drops_feature(self):
        # linear [B,S,in] → [B,S,out]: the last (feature) axis is rewritten — an
        # input shard on it (row-parallel) does not survive as the same axis.
        n = _Node(3, in_ranks=(3,))
        self.assertIsNone(axis_transfer("linear", n, 2))
        self.assertEqual(axis_transfer("linear", n, 1), 1)

    def test_view_split_feature_to_head_count(self):
        # [B,S,feat] → [B,S,N,D]: feature (last in) → head count N (rank_out-2).
        n = _Node(4, axis=-1, in_ranks=(3,))
        self.assertEqual(axis_transfer("view_split", n, 2), 2)  # → N slot
        self.assertEqual(axis_transfer("view_split", n, 0), 0)

    def test_view_merge_head_to_feature(self):
        # [B,S,N,D] → [B,S,feat]: head N (rank_in-2) → merged feature (last out).
        n = _Node(3, in_ranks=(4,))
        self.assertEqual(axis_transfer("view_merge", n, 2), 2)

    def test_attention_score_head_passes_D_drops(self):
        # [B,N,Sq,D]·[B,N,Lq,D] → [B,N,Sq,Lq]: head N (1) passes; D (last) gone.
        n = _Node(4, in_ranks=(4, 4))
        self.assertEqual(axis_transfer("attention_score", n, 1), 1)
        self.assertIsNone(axis_transfer("attention_score", n, 3))

    def test_positional_not_symbolic(self):
        # The transfer never reads an axis symbol, so it is identical whether
        # the head axis is rendered N or a concrete 64 — only the rank matters.
        n = _Node(4, axes=[0, 2, 1, 3], in_ranks=(4,))
        # same call, no symbol anywhere — position 2 → 1 regardless of labels.
        self.assertEqual(axis_transfer("transpose", n, 2), 1)


class TestDimMapStructure(unittest.TestCase):
    def test_view_split_map(self):
        n = _Node(4, axis=-1, in_ranks=(3,))
        dm = dim_map("view_split", n)
        self.assertEqual([s.kind for s in dm], ["from", "from", "split", "split"])
        self.assertEqual(dm[2].part, 0)  # N (head count)
        self.assertEqual(dm[3].part, 1)  # D (per-head)

    def test_contract_map_last_from_input1(self):
        n = _Node(4, in_ranks=(4, 4))
        dm = dim_map("attention_apply", n)
        self.assertEqual(dm[-1].refs, ((1, 3),))  # N (out) from V's last dim

    def test_broadcast_inserts_one_new_axis_and_shifts_the_rest(self):
        n = _Node(4, axis=-2, in_ranks=(3,))
        dm = dim_map("broadcast", n)
        self.assertEqual(
            [(source.kind, source.refs) for source in dm],
            [
                ("from", ((0, 0),)),
                ("from", ((0, 1),)),
                ("new", ()),
                ("from", ((0, 2),)),
            ],
        )

    def test_elementwise_unit_axis_is_not_an_index_source(self):
        gate = ShapeProbe(out_shape=(Dim.symbol("gate", 1),))
        activation = ShapeProbe(out_shape=(
            Dim.symbol("B", 2),
            Dim.symbol("s_ctx", 8),
            Dim.symbol("d_model", 16),
        ))
        node = ShapeProbe(
            out_shape=activation.out_shape,
            inputs_obj=(
                _ShapeRef(src=gate, port=""),
                _ShapeRef(src=activation, port=""),
            ),
        )

        dm = dim_map("mul", node)

        self.assertEqual(dm[-1].refs, ((1, 2),))
        self.assertIsNone(axis_transfer("mul", node, 0, input_index=0))
        self.assertEqual(axis_transfer("mul", node, 2, input_index=1), 2)

    def test_dsa_score_last_axis_is_bound_s_kv(self):
        # Score writes a new last axis sized by the walk's `s_kv` (already
        # Min(s_kv, n_topk_keys)). That is not K's stored sequence.
        n = _Node(4, in_ranks=(4, 4))
        dm = dim_map("dsa_attention_score", n)
        self.assertEqual([s.kind for s in dm], ["from", "from", "from", "new"])
        self.assertEqual(dm[3].dim_key, "s_kv")
        self.assertEqual(axis_transfer("dsa_attention_score", n, 1), 1)
        self.assertEqual(dm[1].refs, ((0, 1),))

    def test_dsa_softmax_and_apply_keep_their_kinds(self):
        # Softmax is preserve; apply is contract. Neither invents a last axis.
        n_sm = _Node(4, in_ranks=(4,))
        dm_sm = dim_map("dsa_softmax", n_sm)
        self.assertEqual([s.kind for s in dm_sm], ["from", "from", "from", "from"])
        self.assertEqual(axis_transfer("dsa_softmax", n_sm, 1), 1)

        n_ap = _Node(4, in_ranks=(4, 4))
        dm_ap = dim_map("dsa_attention_apply", n_ap)
        self.assertEqual([s.kind for s in dm_ap], ["from", "from", "from", "from"])
        self.assertEqual(dm_ap[3].refs, ((1, 3),))
        self.assertEqual(axis_transfer("dsa_attention_apply", n_ap, 1), 1)


class TestShapeConsistency(unittest.TestCase):
    """The dim-map's size relations must agree with resolved shapes — the
    cross-check the size-only compute_shape_out lacks."""

    def test_view_split_product_holds(self):
        # [B,S,384] → [B,S,3,128]: split parts 3*128 == 384. Consistent.
        n = _Node(4, axis=-1, in_ranks=(3,))
        self.assertEqual(
            shape_consistency_errors("view_split", n,
                                     [(1, 64, 384)], (1, 64, 3, 128)), [])

    def test_view_split_bad_product_flagged(self):
        # 3*128 = 384 != 999 → inconsistency caught.
        n = _Node(4, axis=-1, in_ranks=(3,))
        errs = shape_consistency_errors("view_split", n,
                                        [(1, 64, 999)], (1, 64, 3, 128))
        self.assertTrue(errs)

    def test_dim_map_consistent_with_engine_shapes(self):
        """End-to-end: every mappable leaf of every shipped model has a DimMap
        whose size relations match the engine's resolved shapes — a complete,
        self-validating shape propagation. Includes Step-3.7 / MiMo whose head
        axis concretizes to 64, the case that defeats symbol matching: the
        positional DimMap validates them identically."""
        import json
        from tests.snapshots import shared_loader
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        root = Path(__file__).resolve().parents[3]
        snaps = [
            "meta-llama__Llama-3.2-3B.json", "deepseek-ai__DeepSeek-V3.json",
            "moonshotai__Kimi-K2.5.json", "XiaomiMiMo__MiMo-V2.5.json",
            "stepfun-ai__Step-3.7-Flash.json",
        ]

        def _ints(shape):
            return shape and all(isinstance(x, int) for x in shape)

        def _norm_in(si):
            if not si:
                return []
            return list(si) if isinstance(si[0], (list, tuple)) else [si]

        seen = False
        for name in snaps:
            snap = root / "backend/arch/snapshots" / name
            if not snap.exists():
                continue
            seen = True
            resolved = shared_loader().match(json.loads(snap.read_text())["config"],
                                             model_id=snap.stem.replace("__", "/", 1))
            r = analyze(resolved_spec=resolved,
                        parallel=ParallelConfig(tp_attention=1),
                        batch_size=1, avg_input_tokens=64, avg_output_tokens=0, phase="prefill", breakdown=False)
            checked = 0
            for sid, ir, in_shapes, out_shape, _dims in _evaluated_rows(r):
                if not _ints(out_shape):
                    continue
                in_shapes = [s for s in in_shapes if _ints(s)]
                if not in_shapes:
                    continue
                node = _Node(len(out_shape),
                             axes=getattr(ir, "axes", None),
                             axis=getattr(ir, "axis", None),
                             in_ranks=tuple(len(s) for s in in_shapes))
                errs = shape_consistency_errors(ir.op, node, in_shapes, out_shape)
                self.assertEqual(errs, [], f"{name} {sid} ({ir.op}): {errs}")
                checked += 1
            self.assertTrue(checked, f"{name}: no mappable leaves checked")
        self.assertTrue(seen, "no snapshots present")


class TestEdgeConsistency(unittest.TestCase):
    """Rank-sensitive ops flag a wrong-rank input (mis-wired edge / missing
    reshape); rank-polymorphic ops don't."""

    def test_attention_score_wrong_rank_flagged(self):
        # expects rank-4 Q/K; a rank-3 input → mis-wired / missing head reshape.
        errs = edge_consistency_errors(
            "attention_score", [(1, 64, 128), (1, 8, 64, 128)])
        self.assertTrue(errs)
        self.assertIn("input 0", errs[0])

    def test_attention_score_right_rank_clean(self):
        self.assertEqual(
            edge_consistency_errors(
                "attention_score", [(1, 8, 64, 128), (1, 8, 64, 128)]), [])

    def test_pointwise_rank_polymorphic_skipped(self):
        # rmsnorm (PRESERVE) runs on any rank — never flag.
        self.assertEqual(edge_consistency_errors("rmsnorm", [(1, 64)]), [])
        self.assertEqual(edge_consistency_errors("linear", [(1, 64, 8, 128)]), [])


class TestDimMapPropagator(unittest.TestCase):
    """The DimMap is a second, independent shape PROPAGATOR (given inputs,
    infer each output shape) — the declarative form of the engine's procedural
    compute_shape_out. The two agree on every mappable leaf of every model
    (zero mismatch), which is the DimMap being the propagator.
    """

    def test_dimmap_propagation_matches_engine(self):
        import json
        from backend.arch.axis_propagation import infer_shape
        from tests.snapshots import shared_loader
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        root = Path(__file__).resolve().parents[3]

        def _ints(sh):
            return bool(sh) and all(isinstance(x, int) for x in sh)

        def _norm_in(si):
            if not si:
                return []
            return list(si) if isinstance(si[0], (list, tuple)) else [si]

        seen = False
        for name in ("meta-llama__Llama-3.2-3B.json",
                     "deepseek-ai__DeepSeek-V3.json",
                     "stepfun-ai__Step-3.7-Flash.json"):
            snap = root / "backend/arch/snapshots" / name
            if not snap.exists():
                continue
            seen = True
            resolved = shared_loader().match(json.loads(snap.read_text())["config"],
                                             model_id=snap.stem.replace("__", "/", 1))
            r = analyze(resolved_spec=resolved,
                        parallel=ParallelConfig(tp_attention=1),
                        batch_size=1, avg_input_tokens=64, avg_output_tokens=0, phase="prefill", breakdown=False)
            checked = 0
            for sid, ir, ins, out, dims in _evaluated_rows(r):
                ins = [s for s in ins if _ints(s)]
                if not _ints(out) or not ins:
                    continue
                node = _Node(len(out), axes=getattr(ir, "axes", None),
                             axis=getattr(ir, "axis", None),
                             in_ranks=tuple(len(s) for s in ins))
                inferred = infer_shape(ir.op, node, ins, dims)
                if inferred is None:
                    continue
                self.assertEqual(
                    _values(inferred), tuple(out),
                    f"{name} {sid} ({ir.op}): DimMap inferred {inferred} "
                    f"!= engine {out}")
                checked += 1
            self.assertTrue(checked, f"{name}: no mappable leaves inferred")
        self.assertTrue(seen, "no snapshots present")


class TestIRCarriesPermutation(unittest.TestCase):
    """`transpose` registers `permute_to=None`, so its DimMap is derivable ONLY
    from the call-site permutation. That list lives on `GraphNode.axes`, and
    until `SpecIRNode.axes` existed the walker dropped it — so `dim_map` /
    `axis_transfer` returned None for every transpose and permute node in every
    model, and both the shape cross-check and the sharded-axis transfer silently
    skipped them. These pin the field through the walk and the resulting
    coverage, so a future walker change can't drop it again unnoticed.
    """

    # Floor, not a target. Remainder is leftover SOURCE / EXPLICIT ops
    # whose correspondence is still unwritten, not a missing field.
    COVERAGE_FLOOR = 0.88

    MODELS = (
        "meta-llama__Llama-3.2-3B", "Qwen__Qwen3-30B-A3B",
        "deepseek-ai__DeepSeek-V3", "deepseek-ai__DeepSeek-V3.2",
        "XiaomiMiMo__MiMo-V2.5", "openai__gpt-oss-20b",
        "google__gemma-4-12B-it", "deepseek-ai__DeepSeek-V4-Pro",
    )

    @staticmethod
    def _Evaluated(ir, out_rank, in_ranks):
        """`dim_map` needs the output RANK, which a `SpecIRNode` does not carry
        (its `out_shape` is the declared shape of source ops only) —
        ranks live in the evaluated shape layer. So the DimMap is a function of
        BOTH representations: the permutation / axis from the IR and the ranks
        from `Analysis.shapes`.
        """
        return ShapeProbe(
            out_shape=tuple(range(out_rank)), axes=ir.axes, axis=ir.axis,
            dims_raw=dict(ir.dims_raw),
            inputs_obj=tuple(_Ref(r) for r in in_ranks))

    def _analyzed(self, name):
        import json
        from tests.snapshots import shared_loader
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        snap = (Path(__file__).resolve().parents[3]
                / "backend/arch/snapshots" / f"{name}.json")
        if not snap.exists():
            return None
        resolved = shared_loader().match(json.loads(snap.read_text())["config"],
                                         model_id=snap.stem.replace("__", "/", 1))
        return analyze(resolved_spec=resolved,
                       parallel=ParallelConfig(tp_attention=1),
                       batch_size=1, avg_input_tokens=64, avg_output_tokens=0, phase="prefill", breakdown=False)

    def test_permute_ops_carry_axes_into_the_ir(self):
        import json
        from tests.snapshots import shared_loader
        from backend.arch.spec_tree import iter_tree
        root = Path(__file__).resolve().parents[3]
        seen_transpose = False
        for name in self.MODELS:
            snap = root / "backend/arch/snapshots" / f"{name}.json"
            if not snap.exists():
                continue
            resolved = shared_loader().match(json.loads(snap.read_text())["config"],
                                             model_id=snap.stem.replace("__", "/", 1))
            tree = iter_tree(resolved)

            def _walk(nodes):
                nonlocal seen_transpose
                for n in nodes:
                    if n.op == "transpose":
                        seen_transpose = True
                        self.assertIsNotNone(
                            n.axes,
                            f"{name} {n.display_path}: transpose lost its "
                            f"`axes` permutation walking into the IR")
                        rank = len(n.axes)
                        self.assertTrue(
                            dim_map(n.op, self._Evaluated(n, rank, (rank,))),
                            f"{name} {n.display_path}: transpose has `axes` "
                            f"but no DimMap")
                    _walk(n.children)

            _walk(list(tree.top_level))
            for _meta, roots in tree.blocks:
                _walk(roots)
        self.assertTrue(seen_transpose, "no transpose node in any model")

    def test_dimmap_coverage_floor(self):
        from backend.arch.axis_propagation import infer_shape
        from backend.arch.ops import is_primitive

        def _ints(sh):
            return bool(sh) and all(isinstance(x, int) for x in sh)

        def _norm_in(si):
            if not si:
                return []
            return list(si) if isinstance(si[0], (list, tuple)) else [si]

        total = mapped = inferred_ok = 0
        seen = False
        for name in self.MODELS:
            r = self._analyzed(name)
            if r is None:
                continue
            seen = True
            for sid, ir, ins, out, dims in _evaluated_rows(r):
                # Components have no OpSpec — their correspondence is the
                # composition of their children's, nothing to declare.
                if not is_primitive(ir.op):
                    continue
                ins = [s for s in ins if _ints(s)]
                if not _ints(out) or not ins:
                    continue
                total += 1
                node = self._Evaluated(ir, len(out),
                                       tuple(len(s) for s in ins))
                if dim_map(ir.op, node) is not None:
                    mapped += 1
                got = infer_shape(ir.op, node, ins, dims)
                if got is None:
                    continue
                inferred_ok += 1
                self.assertEqual(
                    _values(got), tuple(out),
                    f"{name} {sid} ({ir.op}): DimMap inferred {got} "
                    f"!= engine {out}")
        self.assertTrue(seen, "no snapshots present")
        self.assertGreaterEqual(mapped / total, self.COVERAGE_FLOOR,
                                f"DimMap axis coverage fell to "
                                f"{mapped}/{total}")
        self.assertGreaterEqual(inferred_ok / total, self.COVERAGE_FLOOR,
                                f"DimMap shape coverage fell to "
                                f"{inferred_ok}/{total}")


class TestRankOut(unittest.TestCase):
    """Output rank derived FORWARD from the input ranks.

    `dim_map` reads the output rank off the node, which works for checking a
    shape the engine already computed but is circular for producing one. These
    pin the forward rule, whose falsification is sharp: a wrong rank
    reinterprets every axis position after it, so the rule must either be right
    or say None. Reporting a plausible-but-wrong rank is the one unacceptable
    outcome, which is why the model-wide test asserts zero wrong answers rather
    than a mismatch budget.
    """

    def test_rank_by_shape_kind(self):
        n4 = _Node(4, in_ranks=(4,))
        for op in ("rmsnorm", "silu", "softmax"):        # PRESERVE
            self.assertEqual(rank_out(op, n4, (4,)), 4, op)
        self.assertEqual(rank_out("linear", n4, (3,)), 3)        # AXIS_REPLACE
        self.assertEqual(rank_out("concat", n4, (3, 3)), 3)      # AXIS_CONCAT
        # CONTRACT: A[B,N,Sq,Lq] · V[B,N,Lq,D] → [B,N,Sq,D]; input0's rank
        # survives with its last axis swapped for input1's.
        self.assertEqual(rank_out("attention_apply", n4, (4, 4)), 4)

    def test_unregistered_op_reports_none(self):
        self.assertIsNone(rank_out("no_such_op", _Node(3), (3,)))

    def test_rank_of_view_ops(self):
        self.assertEqual(rank_out("view_split", _Node(4), (3,)), 4)
        self.assertEqual(rank_out("view_merge", _Node(3), (4,)), 3)

    def test_permute_rank_is_the_permutation_length(self):
        n = _Node(4, axes=[0, 2, 1, 3], in_ranks=(4,))
        self.assertEqual(rank_out("transpose", n, (4,)), 4)

    def test_permute_disagreeing_with_input_rank_reports_none(self):
        """A 4-entry permutation on a rank-3 input is a mis-wired call site.
        Reporting 4 would let the caller build a shape of the wrong rank."""
        n = _Node(4, axes=[0, 2, 1, 3], in_ranks=(3,))
        self.assertIsNone(rank_out("transpose", n, (3,)))

    def test_no_input_ranks_reports_none(self):
        """Nothing to derive from — must not fall back to a default rank."""
        self.assertIsNone(rank_out("rmsnorm", _Node(3), ()))
        self.assertIsNone(rank_out("linear", _Node(3), ()))

    def test_authored_explicit_ranks(self):
        """Ranks written next to the DimMap, not guessed from a leftover op."""
        self.assertEqual(rank_out("combine", _Node(3, in_ranks=(3,)), (3,)), 3)
        self.assertEqual(rank_out("sum_reduce", _Node(3, in_ranks=(3,)), (3,)), 2)
        self.assertEqual(rank_out("head_repeat", _Node(4, in_ranks=(4,)), (4,)), 4)
        self.assertEqual(
            rank_out("gated_delta_rule", _Node(4, in_ranks=(4, 4, 4, 4, 3)),
                     (4, 4, 4, 4, 3)),
            4)

    def test_unwritten_explicit_rank_reports_none(self):
        """SOURCE / leftover EXPLICIT with no rank rule return None, never a guess."""
        n = _Node(4, in_ranks=(4,), shape=())
        self.assertIsNone(rank_out("parameter_tensor", n, ()))

    def test_broadcast_rank_inserts_one_axis(self):
        self.assertEqual(rank_out(
            "broadcast", _Node(4, axis=-2, in_ranks=(3,)), (3,)), 4)

    def test_forward_rank_agrees_with_engine_on_shipped_models(self):
        from backend.arch.ops import is_primitive

        def _ints(sh):
            return bool(sh) and all(isinstance(x, int) for x in sh)

        def _norm_in(si):
            if not si:
                return []
            return list(si) if isinstance(si[0], (list, tuple)) else [si]

        total = ok = none = 0
        seen = False
        for name in TestIRCarriesPermutation.MODELS:
            r = TestIRCarriesPermutation._analyzed(self, name)
            if r is None:
                continue
            seen = True
            for sid, ir, ins, out, _dims in _evaluated_rows(r):
                if not is_primitive(ir.op):
                    continue
                if not _ints(out):
                    continue
                ins = [s for s in ins if _ints(s)]
                total += 1
                # `ir` is the real IR node, which carries no output shape —
                # so this cannot leak the rank it is being asked to predict.
                got = rank_out(ir.op, ir, tuple(len(s) for s in ins))
                if got is None:
                    none += 1
                    continue
                self.assertEqual(
                    got, len(out),
                    f"{name} {sid} ({ir.op}): forward rank {got} != "
                    f"engine rank {len(out)} for shape {out}")
                ok += 1
        self.assertTrue(seen, "no snapshots present")
        # 90.5% derived at the time of writing; the remainder is the same known
        # op list the DimMap is missing, plus a handful of linears whose input
        # shape never resolves to ints.
        self.assertGreaterEqual(ok / total, 0.88,
                                f"forward rank coverage fell to {ok}/{total} "
                                f"({none} None)")


class TestDimMapValidator(unittest.TestCase):
    """The inspector's per-op DimMap consistency check runs clean on the
    shipped models (their dataflow geometry is dim-correspondence-consistent),
    and is falsifiable (the bad-product unit test above)."""

    def test_validator_clean_on_shipped_models(self):
        import json
        from tests.snapshots import shared_loader
        from backend.arch.validate._shapes import (
            run_dim_map_consistency_check,
        )
        root = Path(__file__).resolve().parents[3]
        seen = False
        for name in ("meta-llama__Llama-3.2-3B.json",
                     "deepseek-ai__DeepSeek-V3.json",
                     "stepfun-ai__Step-3.7-Flash.json"):
            snap = root / "backend/arch/snapshots" / name
            if not snap.exists():
                continue
            seen = True
            resolved = shared_loader().match(json.loads(snap.read_text())["config"],
                                             model_id=snap.stem.replace("__", "/", 1))
            from backend.arch.eval.projection import project_model
            issues = run_dim_map_consistency_check(
                resolved, project_model(resolved).block_results)
            self.assertEqual(
                issues, [], f"{name}: dim-map mismatches {issues[:4]}")
        self.assertTrue(seen, "no snapshots present")


class TestDimThroughAuthoredShape(unittest.TestCase):
    """An authored rule receives Dims and hands Dims back.

    The mapping itself is written (DimMap). Dim is only the carrier: a
    copied axis keeps the name it arrived with, a named new axis keeps
    the binding's name. Nothing here invents a correspondence.
    """

    def test_infer_shape_copies_dim_and_reads_named_new(self):
        from backend.arch.axis_propagation import infer_shape

        node = _Node(4, axis=-1, in_ranks=(3,),
                     dims_raw={"n_heads": "n_heads", "d_head": "d_head"})
        ins = [(
            Dim.symbol("B", 1),
            Dim.symbol("s_ctx", 64),
            Dim.symbol("d_model", 4096),
        )]
        dims = {
            "n_heads": Dim.symbol("n_heads", 32),
            "d_head": Dim.symbol("d_head", 128),
        }
        out = infer_shape("view_split", node, ins, dims)
        self.assertIsNotNone(out)
        self.assertTrue(out[0].is_("B"))
        self.assertTrue(out[1].is_("s_ctx"))
        self.assertTrue(out[2].is_("n_heads"))
        self.assertTrue(out[3].is_("d_head"))

    def test_gated_delta_restates_v_not_a_guess(self):
        from backend.arch.axis_propagation import infer_shape

        node = _Node(4, in_ranks=(4, 4, 4, 4, 3))
        v = (
            Dim.symbol("B", 1),
            Dim.symbol("n_lin_v_heads", 32),
            Dim.symbol("s_ctx", 64),
            Dim.symbol("d_lin_v_head", 128),
        )
        other = (
            Dim.symbol("B", 1),
            Dim.symbol("n_lin_k_heads", 16),
            Dim.symbol("s_ctx", 64),
            Dim.symbol("d_lin_k_head", 128),
        )
        ins = [other, other, v, v, v[:3]]
        out = infer_shape("gated_delta_rule", node, ins, {})
        self.assertEqual(out, v)
        self.assertTrue(out[1].is_("n_lin_v_heads"))

    def test_a_lookup_keeps_the_ids_axes_and_adds_the_row_width(self):
        """The ids' two axes ARE the output's first two — the same Dim
        objects, not a second reading of `B` / `s_ctx` from the frame."""
        from backend.arch.axis_propagation import infer_shape

        ids = (Dim.symbol("B", 2), Dim.symbol("s_ctx", 64))
        dims = {
            "B": Dim.symbol("B", 7),
            "s_ctx": Dim.symbol("s_ctx", 9),
            "d_model": Dim.symbol("d_model", 4096),
            "n_vocab": Dim.symbol("n_vocab", 32000),
        }
        out = infer_shape("embedding", _Node(3, in_ranks=(2,)), [ids], dims)
        self.assertEqual([int(x) for x in out], [2, 64, 4096])
        self.assertIs(out[0], ids[0])
        self.assertIs(out[1], ids[1])
        self.assertTrue(out[2].is_("d_model"))

    def test_permute_writes_the_expert_buffer(self):
        """`[n_expert_copies, M, d_model]`: the experts outermost, `M` the
        mean rows one expert receives. Both come out of the op's own
        template through the general signature path, bound the way the walk
        binds them on the dispatch node."""
        from backend.arch.axis_propagation import infer_shape

        ins = [(
            Dim.symbol("B", 2),
            Dim.symbol("s_ctx", 64),
            Dim.symbol("d_model", 4096),
        )]
        dims = {"n_expert_copies": Dim.symbol("n_expert_copies", 128),
                "M": Dim.symbol("M", 8)}
        out = infer_shape("permute", _Node(3, in_ranks=(3,)), ins, dims)
        self.assertTrue(out[0].is_("n_expert_copies"))
        self.assertEqual(int(out[0]), 128)
        self.assertTrue(out[1].is_("M"))
        self.assertEqual(int(out[1]), 8)
        self.assertTrue(out[2].is_("d_model"))

    def test_combine_takes_its_token_axes_from_the_routing_operand(self):
        """The inverse: the expert axis and `M` go, and `B` / `s_ctx` come
        from the second operand, which is the only tensor here that still
        carries them. Positional, out of the templates."""
        from backend.arch.axis_propagation import infer_shape

        ins = [
            (Dim.symbol("n_expert_copies", 128), Dim.symbol("M", 8),
             Dim.symbol("d_model", 4096)),
            (Dim.symbol("B", 2), Dim.symbol("s_ctx", 64),
             Dim.symbol("n_experts_active", 8)),
        ]
        out = infer_shape("combine", _Node(3, in_ranks=(3, 3)), ins, {})
        self.assertTrue(out[0].is_("B"))
        self.assertTrue(out[1].is_("s_ctx"))
        self.assertTrue(out[2].is_("d_model"))
        self.assertEqual([int(x) for x in out], [2, 64, 4096])

    def test_axis_view_split_folds_and_marker_preserves(self):
        from backend.arch.axis_propagation import infer_shape

        seq = Dim.symbol("s_ctx", 64)
        dim = Dim.symbol("dim", 128)
        ins = [(Dim.symbol("B", 1), seq, dim)]
        # The collapse moved upstream: an axis-named view_split factors the
        # sequence into (slots, window) — exact product — and
        # token_pool_compressor is the shape-preserving KV-law marker on the
        # already-pooled latent.
        folded = infer_shape(
            "view_split", _Node(4, axis=1, in_ranks=(3,),
                                dims_raw={"s_slots": "", "window": ""}), ins,
            {"s_slots": Dim.symbol("s_slots", 16),
             "window": Dim.symbol("window", 4)})
        self.assertTrue(folded[0].is_("B"))
        self.assertEqual(int(folded[1]), 16)
        self.assertEqual(int(folded[2]), 4)
        self.assertTrue(folded[3].is_("dim"))

        pooled_in = [(Dim.symbol("B", 1), Dim.symbol("s_ctx", 16), dim)]
        pooled = infer_shape(
            "token_pool_compressor", _Node(3, in_ranks=(3,)), pooled_in,
            {"compress_ratio": Dim.symbol("compress_ratio", 4)})
        self.assertTrue(pooled[0].is_("B"))
        self.assertEqual(int(pooled[1]), 16)
        self.assertTrue(pooled[2].is_("dim"))

        # Exact product: a fold whose factors do not multiply back to the
        # split axis is a shape error, never a rounded tail.
        node = _Node(4, axis=1, in_ranks=(3,),
                     dims_raw={"s_slots": "", "window": ""})
        self.assertEqual(shape_consistency_errors(
            "view_split", node, [(1, 64, 128)], (1, 16, 4, 128)), [])
        self.assertTrue(shape_consistency_errors(
            "view_split", node, [(1, 66, 128)], (1, 17, 4, 128)))
        # The split axis must exist on the input.
        with self.assertRaisesRegex(ValueError, "outside input rank"):
            infer_shape(
                "view_split", _Node(4, axis=3, in_ranks=(3,),
                                    dims_raw={"s_slots": "", "window": ""}),
                ins, {"s_slots": Dim.symbol("s_slots", 16),
                      "window": Dim.symbol("window", 4)})

        merged = infer_shape(
            "spatial_merge", _Node(3, in_ranks=(3,)), ins,
            {"n_spatial_merge": Dim.symbol("n_spatial_merge", 2)})
        self.assertEqual(int(merged[1]), 16)
        self.assertEqual(int(merged[2]), 512)
        self.assertIn("s_ctx", merged[1].names)
        self.assertIn("dim", merged[2].names)


if __name__ == "__main__":
    unittest.main()
