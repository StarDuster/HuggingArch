"""A view's reader is charged the tensor behind the view, not the view.

A view moves no bytes (`OpSpec.is_view_op`); the kernel that reads it loads
the stored tensor underneath. Where the view REPEATS its input — a stride-0
`.expand()`, `broadcast` declaring `repeats_input`, or a view-only component
such as `head_broadcast` — the reader's input shape is larger than what is
stored, and charging that shape bills bytes nothing reads. DeepSeek-V4's
indexer broadcasts its [B, S, H] head weights to the [B, H, S, T] score for
the `mul` the fused MQA-logits kernel absorbs; billed at the view, that one
operand was ~35 TB per layer at a 1M-token prefill.

The check, for every fused node a framework's rewrite builds (whose external
inputs are where views most often end): an external input reached through a
chain of views that only repeat or reorder is charged, once the chain's
fan-out is divided out, no more elements than the stored tensor behind the
chain holds. (It may be charged fewer — a sparse or windowed reader reads a
part of what is stored.)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import traffic
from backend.arch.eval.cache import Edges, is_view
from backend.arch.eval.dim import value_of
from backend.arch.eval.fusion import Framework
from backend.arch.eval.launch import FusedNode
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.eval.values import product
from backend.inference.framework_bindings import get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from tests.snapshots import resolve_snapshot, shared_loader

pytestmark = pytest.mark.regression  # four models x two frameworks, full walks

#: Fused indexers fed by views (a broadcast head-weight operand, a
#: head-broadcast key), pooled-key indexers, and hyper-connection blocks.
MODELS = ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4.1-Flash",
          "zai-org/GLM-5.3-Flash", "Qwen/Qwen3.8-Flash-Next")


def _elements(dims) -> Any:
    return product(dims) if dims else 1


def _behind(src: Any, edges: Edges, shapes: Any) -> Optional[Any]:
    """The stored tensor behind `src`'s chain of views, when no view on the
    chain narrows what it is handed (a `slice` reads less than is stored, and
    that is the edge's own extent); None otherwise."""
    node = src
    seen = set()
    while node is not None and is_view(node) and node.uid not in seen:
        seen.add(node.uid)
        if not node.inputs_obj:
            return None
        inner = edges.source(node, 0)
        own, under = shapes.get(node.uid), shapes.get(inner.uid) if inner else None
        if own is None or under is None or own.out is None or under.out is None:
            return None
        if value_of(_elements(own.out)) < value_of(_elements(under.out)):
            return None
        node = inner
    return node


@pytest.mark.parametrize("framework", ("vllm", "sglang"))
@pytest.mark.parametrize("model_id", MODELS)
def test_a_fused_input_is_charged_the_tensor_behind_its_views(model_id, framework):
    resolved = resolve_snapshot(model_id, shared_loader())
    analysis = (Deployment(Program.of(resolved)).at(batch=1, input_tokens=1024)
                .fused(Framework.of(get_binding(framework), FUSION_SCHEMES))
                .analyze())
    program = analysis.program
    edges = Edges(analysis)
    shapes = analysis.shapes.values
    checked = 0
    wrong = []
    for node in program.walked_nodes():
        if not isinstance(node, FusedNode):
            continue
        shape = shapes.get(node.uid)
        if shape is None:
            continue
        for index, dims in enumerate(shape.ins):
            if index >= len(node.inputs_obj or ()):
                continue
            src = edges.source(node, index)
            if src is None or not is_view(src):
                continue
            stored = _behind(src, edges, shapes)
            if stored is None or stored is src:
                continue
            charged = value_of(_elements(dims)) / value_of(
                traffic._fanout(src, shapes, edges=edges))
            held = value_of(_elements(shapes[stored.uid].out))
            checked += 1
            if charged > held:
                wrong.append((node.op, node.display_path, src.display_path,
                              charged, held))
    assert checked, model_id
    assert wrong == []
