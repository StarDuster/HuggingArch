"""Shared constant tables for rendering.

These were inline dicts scattered across builders.py; collected here so the
quant / dtype / styling vocabularies live in one place. No logic — just data.
"""
from __future__ import annotations

from typing import Dict


from backend.arch import roles
from backend.arch.op_names import op_keyed
from backend.arch.ops import op_labels


# -- Op-name → CSS style class (for the block DAG renderer) ----------------
# Used as fallback when a component's role is missing from _ROLE_STYLE.
# `op_keyed` registers the keys for verification — the pre-registration
# table carried `activation` and `expand`, which no op answers to.
STYLE_BY_OP: Dict[str, str] = op_keyed({
    "rmsnorm": "norm", "layernorm": "norm",
    "linear": "op", "embedding": "embed", "add": "res",
    "mul": "op", "rope": "op",
    "softmax": "attn", "attention_score": "attn", "attention_apply": "attn",
    "slice": "view", "concat": "view",
    "view_split": "view", "view_merge": "view", "transpose": "view",
    # Broadcast / expand (e.g. GQA-MQA KV-head expansion) — pure stride
    # view, zero FLOPs/bytes (real kernels broadcast in-register, never
    # materialize). Same `view` style as the other layout ops.
    "broadcast": "view",
    # MoE boundary ops — index-permute / weighted-reduce. Conceptually
    # view-like (permute does no arithmetic, combine does a weighted
    # reduce but the shape transition is the visible effect) and the
    # `(B,S,H) → (M,H)` / `(M,H) → (B,S,H)` shape transition is the
    # whole point of the node. Treating them as `view` style gets
    # the shape-arrow display the user reads as "this is where M
    # comes from / goes back".
    "permute": "view", "combine": "view",
    # sum_reduce is real (billed) compute, but the shape story — where a
    # window axis collapses — is what the card exists to say (same
    # reasoning as combine).
    "sum_reduce": "view",
    # DSv4's compressed-KV marker is a zero-cost identity on the pooled
    # latent (the pooling is the explicit chain above it), so it reads as a
    # view.
    "token_pool_compressor": "view",
    # Collectives are not listed: `_node_build` styles them "comm" from
    # their declaration (`OpSpec.is_comm_op`).
})


# -- Component role → CSS style class --------------------------------------
# The style class is the renderer's own word and is declared nowhere else;
# the KEYS are `roles.yaml`'s, so each is checked against it at import
# (`roles.require_registered`, the same gate the schema puts on a spec
# writing `role:`). It is the failure `op_keyed` exists for: a key nobody
# registered styles nothing, forever, while reading like a live rule —
# `ffn` and `norm` rode an earlier version of this table after no role
# answered to them. A role not named here is styled by the broadest role it
# refines (`_node_build` walks `roles.lineage`), which is why the variant
# slots — `attention_swa`, `ffn_moe_shared` — need no row of their own.
STYLE_BY_ROLE: Dict[str, str] = {
    roles.require_registered(role): style for role, style in (
        ("attention", "attn"), ("ffn_dense", "ffn"),
        ("ffn_moe", "moe"), ("block", "op"),
    )
}


# -- Op-name → human-readable label ----------------------------------------
# The ONE display-label table (the architecture view, the Mermaid
# diagrams, and the model-config panel all read it). `op_keyed`
# registers the keys so a label for an op that does not exist cannot
# hide. Compound attention components bypass it for their canonical
# `attn_display` (see diagrams.py); a miss falls back to the op name.
OP_LABEL: Dict[str, str] = op_keyed(op_labels())



# Byte widths come from `backend.arch.dtypes` — one table, and a miss
# raises there rather than defaulting.
from backend.arch.dtypes import DTYPE_BYTES  # noqa: F401

