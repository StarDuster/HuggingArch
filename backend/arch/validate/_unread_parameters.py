"""A stored parameter that no node of the modelled graph reads.

A parameter source — a lookup's own table (`OpSpec.lookup`) or a tensor
whose output is the tensor itself (`OpSpec.is_identity_source`:
`parameter_tensor`, `parameter_vector`) — stores bytes the checkpoint
holds. When no edge of the program reads it, the byte anchor cannot tell:
the bytes are there either way. Three different things look like that:

* a dropped edge, which is a bug. The corpus has had it four times: a
  position table the model samples by interpolation, written as a lookup
  the blocks never read (Qwen3.8-27B, Qwen3.5-35B-A3B — vLLM qwen3_vl.py
  `fast_pos_embed_interpolate` reads `pos_embed.weight` itself);
  speech-code banks the audio encoder sums, left at the model root while
  the encoder started from an unwired entry (MiMo-V2.5); an MTP head's own
  token embedding, while its `enorm` read the hidden state instead
  (LongCat-Flash);
* a parameter the forward never reads, which the framework loads and
  ignores (Kimi-K3's layer-0 `self_attention_res_*`: the AttnRes history is
  empty there) — `kind: forward_unread`;
* a parameter the forward reads in a step the graph does not model
  (Qwen3.8-Flash-Next's PLE index tables, read by the n-gram id hash) —
  `kind: unmodelled_reader`.

The last two are the spec's declaration, `stored_without_graph_reader:
{kind, reason, source_anchor}` (`schema.weights.StoredWithoutGraphReaderSpec`):
the tensor is stored and counted, and no graph node is billed a read of it.

So the check holds the graph and the declaration to each other:

* a parameter source with no graph reader and no declaration is an error
  (`unread_lookup_table` for a table, `unread_stored_parameter` for an
  identity source): wire its reader, or declare it;
* a node that declares `stored_without_graph_reader` and that a graph node
  reads is an error (`stored_without_graph_reader_is_read`);
* the declaration on anything but a stored identity source is an error
  (`stored_without_graph_reader_not_a_parameter`). A lookup bills its own
  gather whether or not anything reads its rows, so a stored table with no
  graph reader is written as `parameter_tensor` with the declaration.

A read is any edge the walk bound (`Analysis.input_references`): a node's
`in:` inside a graph, a model layer's or a group's `inputs:`, a stream read —
and a `tied_weight:` (which reads the table it ties to). The model's input embedding
(`boundary_role: input_embedding`) and a tied lookup, which stores no table
of its own, are not asked.
"""
from __future__ import annotations

from typing import Iterable, List, Set

from backend.arch.eval.program import Program
from backend.arch.ops import OP_REGISTRY

from ._finding import Finding


class _Reads:
    """The nodes the walk's edges read, by uid."""

    def __init__(self, analysis):
        self.uids: Set[int] = {
            found.occurrence.uid
            for reference in analysis.reference_table.references.values()
            for _iteration, sources in reference.table
            for found in sources}
        self.uids.update(node.tied_to_uid for node in analysis.program.nodes()
                         if node.tied_to_uid is not None)

    def reads(self, node) -> bool:
        """Whether an edge reads `node`."""
        return node.uid in self.uids


def _is_lookup(program: Program, node) -> bool:
    """A lookup into a table it stores itself (`Program.lookups`)."""
    site = program.lookups.get(node.uid)
    return bool(site is not None and site.table == node.uid
                and node.boundary_role != "input_embedding"
                and node.tied_to is None)


def _is_parameter(node) -> bool:
    """A stored identity source: its output is the tensor it stores."""
    spec = OP_REGISTRY.get(node.op)
    return bool(spec is not None and spec.is_identity_source())


def _sources(program: Program) -> Iterable:
    for node in program.nodes():
        if node.children:
            continue
        if _is_lookup(program, node) or _is_parameter(node) \
                or node.stored_without_graph_reader is not None:
            yield node


_DECLARE = ("declare `stored_without_graph_reader: {{ kind: forward_unread "
            "| unmodelled_reader, reason: <why no graph node reads it>, "
            "source_anchor: <file:line> }}` on {where} — `forward_unread` "
            "when the forward never reads it (the framework loads it and "
            "ignores it), `unmodelled_reader` when the forward reads it in a "
            "step the graph does not model")


def _finding(kind: str, node, detail: str) -> Finding:
    return Finding(kind=kind, template=node.template or None,
                   node=node.display_path, op=node.op, detail=detail)


def run_unread_parameter_check(analysis) -> List[Finding]:
    """One finding per stored parameter whose readers disagree with its
    `stored_without_graph_reader` declaration."""
    program = analysis.program
    reads = _Reads(analysis)
    findings: List[Finding] = []
    for node in _sources(program):
        where = node.display_path
        declared = node.stored_without_graph_reader
        read = reads.reads(node)
        if declared is not None and not _is_parameter(node):
            findings.append(_finding(
                "stored_without_graph_reader_not_a_parameter", node,
                f"{where} (`{node.op}`) declares "
                f"`stored_without_graph_reader`, which only a stored "
                f"parameter source declares (an op whose output is its "
                f"stored tensor). Drop the declaration, or, for a table no "
                f"graph node reads, write it as `op: parameter_tensor` with "
                f"`shape: [rows, width]` and keep the declaration."))
        elif declared is not None and read:
            findings.append(_finding(
                "stored_without_graph_reader_is_read", node,
                f"{where} declares `stored_without_graph_reader` "
                f"({declared.kind}: {declared.reason}), but a graph node "
                f"reads it. Drop the declaration: it says no graph node "
                f"does."))
        elif declared is None and not read and _is_lookup(program, node):
            findings.append(_finding(
                "unread_lookup_table", node,
                f"{where} (`{node.op}`) is a lookup table that no node "
                f"reads. Wire its reader: name it in the `inputs:` of the "
                f"model layer or group that consumes it (or the `in:` of the "
                f"graph node), as the forward source does. If a later op "
                f"samples the table itself rather than gathering rows by "
                f"token id (an interpolated position table), write it as "
                f"`op: parameter_tensor` with `shape: [rows, width]` and "
                f"have that op read it. If no graph node is to read it, "
                f"write it as `op: parameter_tensor` and "
                + _DECLARE.format(where="it") + "."))
        elif declared is None and not read:
            findings.append(_finding(
                "unread_stored_parameter", node,
                f"{where} (`{node.op}`) is a stored parameter that no node "
                f"reads. Wire its reader as the forward source does (the "
                f"`in:` of the node that consumes it), or "
                + _DECLARE.format(where="the node") + "."))
    return findings
