"""Convert a spec block graph into the canonical StructOp list.

Duck-types the block graph object: expects `graph: Dict[str, Node]`
where each Node has `.op` (str) and `.inputs` (list[str]), plus
`outputs: list[str]` and `dependency_order()` at the block level — the
order the evaluator walks, so this comparison and the evaluation agree on
which node comes first. Matches `backend.arch.schema.BlockGraph` but the
module doesn't import that — keeps the structural package decoupled from
arch internals.
"""
from __future__ import annotations

from typing import Any, List

from .types import StructOp
from backend.arch.op_names import op_names
from backend.arch.ops import OP_REGISTRY, rotary_ops


# Ops that are pure tensor manipulation (never written as `self.xxx(...)`
# in forward code — they're inline expressions like `a + b`, `a * b`,
# `x[:, :limit]`, `F.silu(x)`). Exclude from "call" classification so the
# comparator doesn't hunt for non-existent submodule calls in forward.
# Ops that never appear as a named `self.xxx(...)` submodule in a forward, so
# the structural comparator emits nothing for them.
#
# Hand-listed, and it should not be: this is a property of an op. The obvious
# derivation — "owns no weight" — over-covers, naming 57 ops against these 28
# by sweeping in collectives, `gather`, `topk` and `combine`, so the criterion
# is the comparator's and has to come from it rather than from a proxy.
_TENSOR_MATH_OPS = op_names(
    # primitive tensor ops
    "add", "mul",
    # row scatters: `x.masked_scatter(mask, v)` / `x[mask, :] += v`
    "scatter_rows_replace", "scatter_rows_add",
    # view ops
    "slice", "concat", "view_split", "view_merge", "transpose",
    # attention-internal primitives
    "attention_score", "attention_apply", "softmax", "softmax_with_sink",
    "dsa_attention_score", "dsa_attention_apply", "dsa_softmax", "dsa_topk_mask",
    "indexer_logits",
    "swa_attention_score", "swa_attention_apply", "swa_softmax",
    # activations: every op that declares itself one (`is_activation`) —
    # written `F.silu(x)` / `self.act_fn(x)` / `torch.sigmoid(x)`, never as
    # a submodule named after the spec node. Asked of the registry, not
    # listed: a hand list of them covered six and missed the rest.
    *(name for name, op in OP_REGISTRY.items() if op.is_activation),
    # rotations: `apply_rotary_pos_emb(q, k, cos, sin)` is a function, not a
    # submodule — every op that declares one (`rotary_position_axes`)
    *rotary_ops(),
    # bare-parameter ops
    "parameter_vector",
    # compute kernels that aren't named submodules
    "gated_delta_rule", "conv1d",
)


def extract_spec_ops(block: Any) -> List[StructOp]:
    """Walk `block.graph` in dependency order and emit canonical StructOps.

    - `op == "add"` → StructOp(kind="add", inputs=node.inputs, output=name)
    - anything else → StructOp(kind="call", name=name, inputs=node.inputs,
                                 output=name) — the node name IS the
      submodule name per the "mirror named forward submodules" rule.
    A synthetic "return" is appended pointing at `block.outputs` so the
    tail matches the forward-side shape.
    """
    if block is None or not getattr(block, "graph", None):
        return []

    graph = dict(block.graph)
    # The order the evaluator walks: one answer to "which node comes first".
    order = block.dependency_order()
    ops: List[StructOp] = []

    for name in order:
        node = graph[name]
        inputs = list(getattr(node, "inputs", []) or [])
        op = getattr(node, "op", "")
        if op == "add":
            ops.append(StructOp(kind="add", inputs=inputs, output=name))
        elif op in _TENSOR_MATH_OPS:
            # Pure tensor manipulation — never appears as a `self.xxx(...)`
            # call in forward, so emit nothing. The comparator only
            # inspects structural calls + adds.
            continue
        else:
            # Named submodule call (rmsnorm / linear / embedding /
            # component invocation). Spec node name matches the forward
            # submodule name per the "mirror named forward submodules"
            # authoring convention.
            ops.append(StructOp(kind="call", name=name, inputs=inputs, output=name))

    ops.append(StructOp(kind="return",
                        inputs=list(getattr(block, "outputs", []) or [])))
    return ops
