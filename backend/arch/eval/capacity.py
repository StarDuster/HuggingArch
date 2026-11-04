"""What one rank holds: its resident weights.

The per-request cache beside them is `eval.cache_view.CacheView`'s, and how
many requests fit is the serving capacity estimate's
(`inference/capacity.py`); neither is restated here.

Why there is no divisor
-----------------------
The weights are read off a walk that carries layout. Extents stay global;
per-rank bytes come from the stored artifacts the layout projects, not from
dividing a cluster total by a class table.

The alternative divides a full-cluster total by a per-class summary of which
axes the deployment splits — a table maintained beside the walk that already
knows. Every such summary loses the same three cases: a degree that does not
divide its axis (4 KV heads over 8 ranks shard by 4, and a scalar divisor says
8), a model whose classes saturate at different degrees (no single ratio is
right for gemma-4), and a mixed-precision model where the class is not the unit
the dtype varies over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .values import Extent


#: The `Weights.by_dtype` key quantisation metadata (scales, zero points) is
#: filed under: it rides beside the weight it describes, at its own width.
SCALES = "scales"


@dataclass(frozen=True)
class Weights:
    """What a rank stores, in the three kinds a checkpoint actually holds.

    Three and not two, and the third is the one that gets lost. A checkpoint
    contains learnable parameters, the quantisation metadata that comes with
    them, and non-learnable BUFFERS — `rotary_emb.inv_freq` and its kind — which
    are stored, loaded and resident, and are not parameters. Every one of the
    three is on the device; only the first is comparable against a model card.

    Collapsing buffers into either neighbour is how they go missing. A
    breakdown that reports parameters and calls the result the per-rank
    weight bytes is short by every buffer in the model: 8,192
    bytes on Llama-2-7B, 49,152 on MiMo-V2-Flash. Small, and small in the
    direction that never gets investigated — they are also exactly the tensors
    the header-side `_is_quant_meta` predicate misfiles as quantisation scales,
    which is the same conflation from the other end.

    `by_stack` because the backbone and an auxiliary head are loaded together
    but budgeted apart: a speculative head is weights you may choose not to
    load, and a vision tower is weights that are resident whether or not any
    request has an image in it.

    `by_uid` is the same three kinds for each program node that stores
    something, over all of its copies. A quantisation scale is on the row of
    the weight it scales, so it sits under that weight's node; a buffer sits
    under the node whose op declares it. The drawn graph, the structure view
    and the editor's per-group totals read their weights from here rather
    than pricing the rows again, so they add up to `resident` by construction.
    """

    parameters: Extent = 0
    metadata: Extent = 0
    buffers: Extent = 0
    by_stack: Mapping[str, Extent] = field(default_factory=dict)
    by_uid: Mapping[int, "Weights"] = field(default_factory=dict)
    #: `resident`, split by the width it is stored at: parameters and buffers
    #: under their tensor's logical dtype name, quantisation metadata under
    #: `SCALES`. Kept on the per-node entries in `by_uid` (and on what
    #: `by_root` folds from them); the ledger's own entry leaves it empty.
    by_dtype: Mapping[str, Extent] = field(default_factory=dict)

    @property
    def resident(self) -> Extent:
        """Everything on the device — what a capacity subtraction uses."""
        return self.parameters + self.metadata + self.buffers

    def without_aux(self, program: Any) -> Extent:
        """The backbone alone."""
        declared = program.included_stack_names(include_aux=True)
        unknown = set(self.by_stack) - declared
        if unknown:
            raise ValueError(
                f"weight ledger names undeclared stack(s) {sorted(unknown)!r}"
            )
        included = program.included_stack_names()
        return sum(v for k, v in self.by_stack.items() if k in included)

    def by_root(self, program: Any) -> Dict[int, "Weights"]:
        """The per-node entries, folded onto the accounting root above each
        (`Program.contributions`) — keyed by the root's uid. The values sum
        to `resident`: the same bytes, grouped."""
        roots = {c.node.uid for c in program.contributions}
        out: Dict[int, Weights] = {}
        for uid, held in self.by_uid.items():
            node = program.by_uid(uid)
            while node is not None and node.uid not in roots:
                node = node.parent
            if node is None:
                raise ValueError(
                    f"node {uid} stores weights but sits under no root the "
                    f"model total counts")
            out[node.uid] = out.get(node.uid, Weights()).merged(held)
        return out

    def by_block(self, program: Any) -> Dict[Optional[int], Extent]:
        """`resident`, split by the block invocation that holds it — keyed by
        `block_index` (the ordinal `Program.blocks` and `iter_blocks` share;
        None for a top-level node such as the embedding): `by_root`, keyed
        by the block its root belongs to."""
        blocks = {c.node.uid: c.block_index for c in program.contributions}
        out: Dict[Optional[int], Extent] = {}
        for root, held in self.by_root(program).items():
            key = blocks[root]
            out[key] = out.get(key, 0) + held.resident
        return out

    def add(self, parameters: Extent = 0, metadata: Extent = 0,
            buffers: Extent = 0, *, dtype: Optional[str] = None) -> "Weights":
        """These three kinds with more of each — the fold one node's entry
        in `by_uid` is built by. `dtype` names the width the parameters or
        buffers added here are stored at; the metadata goes under `SCALES`.
        The result has no `by_stack` or `by_uid`: a node's entry is these
        sums, and a whole ledger is built by its rule, not by adding
        entries."""
        by_dtype = dict(self.by_dtype)
        own = parameters + buffers
        if own:
            if not dtype:
                raise UnpricedTensor(
                    "stored bytes added with no dtype to file them under")
            by_dtype[dtype] = by_dtype.get(dtype, 0) + own
        if metadata:
            by_dtype[SCALES] = by_dtype.get(SCALES, 0) + metadata
        return Weights(parameters=self.parameters + parameters,
                       metadata=self.metadata + metadata,
                       buffers=self.buffers + buffers,
                       by_dtype=by_dtype)

    def merged(self, other: "Weights") -> "Weights":
        """Two nodes' entries as one — the fold `by_root` groups them by."""
        by_dtype = dict(self.by_dtype)
        for name, amount in other.by_dtype.items():
            by_dtype[name] = by_dtype.get(name, 0) + amount
        return Weights(parameters=self.parameters + other.parameters,
                       metadata=self.metadata + other.metadata,
                       buffers=self.buffers + other.buffers,
                       by_dtype=by_dtype)


def weights(analysis: Any) -> Weights:
    """Resident weight bytes, from the one join that knows both halves.

    The walk's rule answers what a rank holds: a layout rule prices its
    per-rank artifacts, and the identity `Rule` prices the global join.
    """
    return analysis.rule.resident_weights(analysis)


class UnpricedTensor(ValueError):
    """A stored tensor with no format to price it at."""
