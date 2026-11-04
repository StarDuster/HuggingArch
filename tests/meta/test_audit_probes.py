"""Probes from the 2026-09 audit.

Each test states a fact the fix plan wants to be true. The ones marked
`expectedFailure` describe holes that are open at the time of writing: a
missing binding or a malformed declaration is answered with a plausible
number instead of an error. When the corresponding fix lands, the marker
comes off in the same commit. The unmarked ones already hold on the corpus
and stay as the fact they anchor.

Every corpus test here walks all snapshots once, so the file is slow.
"""
from __future__ import annotations

import collections
import json
import sys
import unittest
import pytest
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import ops                                 
from backend.arch.eval import evaluator as _evaluator        
from backend.arch.eval import flops as _flops                
from backend.arch.eval.program import Program                
from backend.arch.eval.transforms import DECODE, Deployment, Phase
from tests.snapshots import shared_loader

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _corpus():
    loader = shared_loader()
    for snap in sorted(SNAPSHOTS.glob("*.json")):
        config = json.loads(snap.read_text(encoding="utf-8"))["config"]
        resolved = loader.match(config, model_id=snap.stem.replace("__", "/", 1))
        assert resolved is not None, f"{snap.name} no longer resolves"
        yield snap.stem, resolved


class TestPhaseTripleIsRequired(unittest.TestCase):
    """C1: attention FLOPs must not invent `s_pairs` when the phase
    binding did not supply it. `bind_labels` fills `s_query`/`s_kv` from
    the shapes, so the live pairs are the one key that can only come
    from `Phase.bind`."""

    def test_missing_pairs_raises(self) -> None:
        spec = ops.OP_REGISTRY["attention_score"]
        dims = {"B": 1, "n_heads": 8, "n_kv_heads": 8, "d_head": 64,
                "s_query": 128, "s_kv": 128, "s_ctx": 128}
        with self.assertRaises(KeyError):
            spec.flops_fn(dims)


class TestShapeTemplateRejectsUnknownNames(unittest.TestCase):
    """C2: a weight_shape entry that is neither a bound slot nor an
    expression over bound dims is a spelling error, not a free symbol. The
    resolver answers it the way a missing dim is answered: `KeyError`."""

    def test_unknown_bare_name_raises(self) -> None:
        with self.assertRaises(KeyError):
            ops.resolve_shape_template(("in", "nonexistent_dim"), {"in": 8})


class TestWalkIsStrictAboutShapeLabels(unittest.TestCase):
    """C4: the evaluation walk and the per-rank projection apply the same
    rule to conflicting shape labels. The walk now raises like the
    projection; no corpus model has a conflict, which this test pins."""

    def test_corpus_walks_under_strict_labels(self) -> None:
        warnings.filterwarnings("ignore")
        original = _flops.flops_from_shape

        def strict(op, dims, ins, evaluate, **kw):
            kw["require_consistent_labels"] = True
            return original(op, dims, ins, evaluate, **kw)

        _evaluator.flops_from_shape = strict
        try:
            for name, resolved in _corpus():
                with self.subTest(model=name):
                    prefill = Deployment(Program.of(resolved)).at(
                        batch=1, input_tokens=4096)
                    prefill.analyze()
                    decode = Deployment(
                        Program.of(resolved),
                        phase=Phase(kind=DECODE, cache_len=4096),
                    ).at(batch=8, input_tokens=4096, output_tokens=512)
                    decode.analyze()
        finally:
            _evaluator.flops_from_shape = original


class TestKVIsolationFailureIsAnError(unittest.TestCase):
    """C5: asking the cache layer about a component the spec does not have
    raises; it used to answer an empty tuple, which capacity read as a
    model with no KV cache."""

    def test_unknown_component_raises(self) -> None:
        from backend.arch.eval.cache import of_component
        from backend.arch.eval.transforms import analyze_isolated
        from backend.inference.kv_topology import kv_cache_topology
        resolved = shared_loader().match(json.loads(
            (SNAPSHOTS / "meta-llama__Llama-3.2-3B.json").read_text())["config"],
            model_id="meta-llama/Llama-3.2-3B")
        with self.assertRaises(KeyError):
            of_component(analyze_isolated(resolved, "no_such_component"),
                         "no_such_component")
        with self.assertRaises(KeyError):
            kv_cache_topology(resolved, "no_such_component")


class TestKernelBenchReadsSpecDtype(unittest.TestCase):
    """G1: the planner reads stored-tensor dtypes, not a last-wins role map.

    An empty *node* map would still price every GEMM as bf16. An empty *role*
    map means every role is mixed — speak-up, not a missing spec (MiMo-V2.5
    is fp8 with per-layer ignores, so all four roles disagree).
    """

    def test_every_corpus_model_yields_node_dtypes(self) -> None:
        """Every model's GEMM rows name a weight dtype.

        This used to call `plan._weight_dtypes`, which walked the
        materialized tensors a second time to rebuild per-node weight
        dtypes. That walk is gone: the dtype is on the row, where the walk
        put it (`access.LeafAccess.weight_dtype`), and the plan reads it.
        So the question is the same one and it is asked of the row — an
        empty answer still means every GEMM would be priced bf16.
        """
        from backend.arch.ops import is_gemm_op
        from backend.inference.framework_bindings import (DEFAULT_FRAMEWORK,
                                                          get_binding)
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        binding = get_binding(DEFAULT_FRAMEWORK)
        for name, resolved in _corpus():
            with self.subTest(model=name):
                ar = analyze(resolved, parallel=ParallelConfig(),
                             framework_binding=binding,
                             batch_size=1, avg_input_tokens=8, avg_output_tokens=0, breakdown=False)
                named = [row for row in ar.per_op_timing
                         if is_gemm_op(str(row.get("op") or ""))
                         and row.get("weight_dtype")]
                self.assertTrue(
                    named,
                    f"{name}: no GEMM row carries a weight dtype, so every "
                    f"one of them would be measured as bf16")


class TestRenderedWeightBytesMatchTensorRows(unittest.TestCase):
    """D3: the inspector's per-leaf `weight_bytes` prices each parameter at
    the width `Analysis.tensors()` holds for that node, with the scales its
    format stores beside it, and a leaf that stores parameters has a row.
    Rendering used to multiply `params` by a dtype-label lookup of its own,
    which had no entry for `FP4` and priced the expert banks of gpt-oss,
    Kimi-K3 and DeepSeek-V4-Pro at zero."""

    @staticmethod
    def _leaves(node):
        if isinstance(node, dict):
            if "uid" in node and "weight_bytes" in node and not node.get("children"):
                yield node
            for value in node.values():
                yield from TestRenderedWeightBytesMatchTensorRows._leaves(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                yield from TestRenderedWeightBytesMatchTensorRows._leaves(value)

    def test_leaf_bytes_are_params_at_the_row_width(self) -> None:
        warnings.filterwarnings("ignore")
        from backend.rendering.builders import build_block_graph_data
        mismatched = []
        for name, resolved in _corpus():
            view = build_block_graph_data(resolved)
            # The drawn blocks are the whole checkpoint's, the encoders'
            # included (`Deployment.checkpoint`), as the view's own walk.
            analysis = Deployment(Program.of(resolved)).at(
                batch=1, input_tokens=1).checkpoint()
            stored: dict = collections.defaultdict(lambda: [0.0, 0.0])
            for row in analysis.tensors():
                if row.is_buffer or row.is_metadata:
                    continue
                # A row is one (tensor × layer segment) at PER-COPY bytes and
                # elements; `count` is how many of the node's copies the
                # segment covers. Weighting by it is what makes a node whose
                # layers mix formats (Step-5's expert bank: FP8 in 63 of 66
                # layers, BF16 in 3) average to the copy-weighted width the
                # render prices it at, instead of the mean of its formats.
                stored[row.uid][0] += row.count * (float(row.storage_bytes)
                                                   + float(row.format.metadata_bytes))
                stored[row.uid][1] += row.count * float(row.elements)
            for leaf in self._leaves(view):
                params = leaf.get("params", 0)
                if not params:
                    continue
                rendered = int(leaf["weight_bytes"] or 0)
                if leaf["uid"] not in stored:
                    mismatched.append((name, leaf.get("id"), rendered, "no row"))
                    continue
                total_bytes, elements = stored[leaf["uid"]]
                expected = round(params * total_bytes / elements)
                if rendered != expected:
                    mismatched.append((name, leaf.get("id"), rendered, expected))
        self.assertEqual(mismatched, [], f"{len(mismatched)} leaves disagree: {mismatched[:5]}")


if __name__ == "__main__":
    unittest.main()
