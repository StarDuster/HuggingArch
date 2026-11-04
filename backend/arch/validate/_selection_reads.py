"""A score that reads through a selection reads a bounded set of positions.

An attention score wired with a `select` operand (`OpSpec.input_roles`: the
top-k a sparse indexer hands it) reads, per query, the positions the
selection names — a fixed count however long the cache grows. Every cached
tensor it reads must say so in its read law (`CachedTensor.read`): a window,
or a top-k cap (`read_topk`). A tensor whose read law still grows with the
cache length while the only reader scoring it is selected states two reads
for one tensor, and every figure priced off the law — the KV bytes a step
reads, the decode roofline — takes the larger one.

DeepSeek-V4.1-Flash's generated spec had it: its layer-20 attention scored
under the indexer's top-k while its compressed cache kept the law of a
dense read, so the decode KV read at 64K came out 2.35x what the kernel
loads. The weight bytes matched the checkpoint exactly; no other check
looks at a read law against the score that reads it.

Internal consistency of the graph alone, nothing read from a source: it
gates.
"""
from __future__ import annotations

from typing import Any, List

from backend.arch import roles
from backend.arch.eval.cache import Edges, kv_access, law_grows, resident

from ._finding import Finding

KIND = "selected_read_unbounded"


def _attention_call(node: Any) -> Any:
    """The outermost enclosing call whose role is attention's."""
    found, cursor = None, node.parent
    while cursor is not None:
        if roles.kind_of(cursor.role or "") == "attention":
            found = cursor
        cursor = cursor.parent
    return found


def _members(call: Any) -> set:
    out, pending = set(), [call]
    while pending:
        node = pending.pop()
        out.add(node.uid)
        pending.extend(node.children or ())
    return out


def _keys_read(node: Any, edges: Edges, access: Any, inside: set) -> set:
    """The cached tensors read on the way to the score's keys: the cache
    reads of every node its `k` operand comes from, inside its call, and the
    score's own — an absorbed MLA scores the latent through views alone, so
    the score itself is the latent's reader. Its other operands, the query
    and the selection, are made this step and read no cache."""
    reads: set = set(access[node.uid].reads) if node.uid in access else set()
    seen: set = set()
    pending = list(edges.sources(node, 1))
    while pending:
        source = pending.pop()
        if source.uid in seen or source.uid not in inside:
            continue
        seen.add(source.uid)
        if source.uid in access:
            reads |= access[source.uid].reads
        for index in range(len(source.inputs_obj or ())):
            pending.extend(edges.sources(source, index))
    return reads


def run_selection_read_check(analysis: Any) -> List[Finding]:
    program = analysis.program
    held = resident(analysis)
    access = kv_access(analysis).values
    edges = Edges(analysis)
    out: List[Finding] = []
    for node in program.nodes():
        declaration = node.declaration
        if node.children or declaration is None:
            continue
        # `input_roles` is the op view both a registered primitive and a
        # spec-declared custom op present (`ops_base.OpView`).
        wired = tuple(declaration.input_roles or ())[:len(node.inputs_obj or ())]
        if "select" not in wired:
            continue
        call = _attention_call(node)
        if call is None:
            continue
        for ref in sorted(_keys_read(node, edges, access, _members(call)),
                          key=str):
            _holder, _cache, tensor = held[ref]
            if tensor.recurrent or not law_grows(tensor.read):
                continue
            out.append(Finding(
                kind=KIND,
                node=node.display_path,
                op=node.op,
                detail=(
                    f"{node.display_path} ({node.op}) scores under a "
                    f"selection (its `select` operand), so it reads a fixed "
                    f"number of positions per query, but the cached tensor "
                    f"`{tensor.name}` its keys come from keeps the read law "
                    f"{tensor.read} of a dense read. State the selection's "
                    f"cap on the cached node (e.g. `read_topk:` on a "
                    f"`token_pool_compressor`), or wire the score without "
                    f"the selection if it reads every position."),
            ))
    return out
