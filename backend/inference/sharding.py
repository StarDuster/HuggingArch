"""Per-leaf weight CLASS — which part of the model a weight belongs to.

Read off declarations only: the IR role of each call down to the leaf, asked
of `roles.yaml` (`role_class`); whether the weight runs on dispatched tokens
(`Program.routed_uids`); the leaf's declared `boundary_role` for the vocab
endpoints; and, for the `replicated` demotion, which axes a TP knob divides
(`dims.yaml`). No name is read.

A class, and not a factor. How much of a tensor a rank holds is a ratio of
shapes — the per-rank shape the forward walk derives over the global one —
and a class cannot produce it: a fused QKV's axes enter additively, so a
degree that splits 32 query heads eight ways beside 2 KV heads twice leaves
6/36 on a rank, which is neither split; and an extent below the degree does
not split at all (gemma-4's single KV head). The class table that used to
live here answered `tp_attention` to both. It survived because nothing read
it — `weight_breakdown` takes its bytes from `_per_rank_share` and only ever
used the class — so the two mechanisms never had to disagree out loud.

Class taxonomy:
- `attention` / `ffn_dense` / `ffn_moe` — the root of a weight-class role's
  lineage (`roles.yaml`): every `attention_*` variant refines `attention`, a
  shared or latent expert refines `ffn_moe`.
- `moe_router` / `moe_routed` / `moe_shared` — the part of an MoE layer a
  weight is (`moe_child_class`): the gate refines `router`, the bank runs
  in the `permute` → `combine` zone, and what is neither is the shared
  expert.
- `embedding` / `lm_head` — model-level vocab endpoints, by declared
  `boundary_role` (`vocab_kind_of_leaf`).
- `aux` — leaves no weight-class role encloses (Llama's `input_layernorm`).
- `replicated` — leaves carrying no axis that lives in a tensor-parallel
  world some deployment field can widen: MLA's latent projections
  (d_kv_lora / d_kv_lora+d_qk_rope with no `n_heads` axis), LoRA, DSA's
  indexer, etc.

Consumer: `backend/inference/weight_breakdown.py`, which groups the tree by
class and labels each row with it.
"""
from __future__ import annotations

from typing import Dict, Optional

from backend.arch.ops import OpSpec
from backend.arch import dims, roles
from backend.arch.spec_tree import LeafTensor


# Atomic dim → the ParallelConfig field giving its per-rank degree, for the
# two meshes that divide a WEIGHT's own axes: `tp` (a matrix axis) and `ep`
# (an expert bank's leading axis). Sequence axes are cp's and applied by the
# policy, not this map.
#
# Read out of `dims.yaml`, where each axis names the WORLD that splits it and
# each world names its field, rather than re-derived here from kinds. Deriving
# it needed a rule per kind and a name check beside it (`d_ffn_expert` is the
# expert intermediate, `d_ffn` the dense one), which is a second place to say
# which knob owns which axis and was wrong in the ways a second place is: it
# named the GQA head symbols and not the linear-attention ones, so the
# per-rank walk left every KDA/GDN gate and SSM head whole while the direct
# walk sharded them. An axis whose world has no field is absent, which is the
# same answer as before for a different and honest reason: no deployment knob
# names DSA's indexer, so it runs at 1.
#
# `n_expert_copies` is here and the routing counts are not: EP divides the
# expert AXIS, and dividing the dispatch denominator `n_route_targets` in
# `M = B·s_ctx·n_experts_active·dp/n_route_targets` would shrink `M` — and
# the router it sizes, below its full set of destinations. `n_experts` — how many experts the model has — is not
# here either: it is a declaration, never an extent. They are separate symbols
# for exactly this reason.
#
# It is imported from here, with the rest of "which knob shards which leaf",
# because both the per-rank shape walk and the weight breakdown ask. Read when
# asked: a copy taken at import misses a symbol a spec declares afterwards.
def dim_to_tp_field() -> Dict[str, str]:
    return dict(dims.degree_fields("tp", "ep"))


def is_attention_role(role: Optional[str]) -> bool:
    """True for the attention weight-class family: the canonical `attention`
    role and variant slot names like `attention_swa` (gpt-oss / mimo / DSv4-MTP
    sliding-window attention). These share attention's sharding semantics
    (TP-attention head split, CP Ulysses head-shard / a2a), so every consumer
    — the weight breakdown and the comm policies — keys off this same
    predicate rather than an exact `== "attention"`. That
    way a new attention variant is recognized everywhere at once instead of
    silently falling through one call site (which left SWA attention untagged
    by the CP policy while the shape layer head-sharded it).

    Membership is `roles.yaml`'s to state. It used to be a `startswith`
    test, which is the same guess by a different spelling: it answered for
    names nobody had classified, and answered `False` for any future variant
    not spelled `attention_*`. The registry answers only for roles someone
    declared, and `SpecLoader` refuses to load an undeclared one."""
    return bool(role) and roles.kind_of(role) == "attention"


def role_class(role: Optional[str]) -> Optional[str]:
    """The weight class a role names, from `roles.yaml`: the root of the
    role's lineage (every attention variant refines `attention`, a shared or
    latent expert refines `ffn_moe`) when the registry marks the role a
    weight class; None for a role that names none — a wrapper, a norm, the
    router — whose weights take the class of what encloses them."""
    if role in roles.weight_classes():
        return roles.lineage(role)[-1]
    return None


def moe_child_class(role: Optional[str], routed: bool) -> str:
    """Which part of an MoE layer a weight is, from two declared facts: the
    gate refines `router` (roles.yaml), and the expert bank runs on
    dispatched tokens (`Program.routed_uids`, the `permute` → `combine`
    zone). What is neither is the always-on shared expert — told from the
    bank by where it runs, not by its role."""
    if "router" in roles.lineage(role):
        return "moe_router"
    if routed:
        return "moe_routed"
    return "moe_shared"


def step_class(prev_class: Optional[str], role: Optional[str]) -> Optional[str]:
    """The class one level down the IR tree — a component invocation or a
    leaf — whose IR `role` is `role` (the invocation's, not the component's
    by name: DeepSeek-V3's `swiglu_ffn` is the dense FFN of its first layers
    and the shared expert of the rest).

    Inside an MoE layer everything stays the MoE layer's, however deep and
    whatever a nested call declares: which PART of it a weight is, is asked
    per weight (`leaf_class`). Elsewhere it is the node's own role class, or
    the enclosing one when its role names none.
    """
    if prev_class == "ffn_moe":
        return prev_class
    own = role_class(role)
    return own if own is not None else prev_class


def _tp_axis_placement(lt: LeafTensor) -> bool:
    """Whether the leaf's op declares that its weight's placement under
    tensor parallelism follows the axes a TP knob divides (`OpSpec.
    tp_axis_placement`); a custom-op leaf carries no such declaration."""
    decl = lt.ir_node.declaration
    return isinstance(decl, OpSpec) and decl.tp_axis_placement


def leaf_class(lt: LeafTensor, prev_class: Optional[str],
               *, routed: bool) -> str:
    """A weight leaf's class: `step_class` taken one level further, at the
    leaf itself. A weight of an MoE layer is the part of it that it is
    (`moe_child_class`, from its own role and whether it runs on dispatched
    tokens) — also when no MoE component encloses it and its own role says
    MoE work: an expert bank wired as inline linears carrying `role: ffn_moe`
    (MiniMax-M2's `block_sparse_moe.experts.w1/w2/w3`). A leaf that inherits
    no class is `aux` (Llama's `input_layernorm`).

    Then the structural demotion: a weight whose placement follows the axes
    TP divides (`OpSpec.tp_axis_placement`) but carries none of them is
    `replicated`, whatever class it would wear — the MLA latent projections
    (latent dims, no head axis) are the canonical case. The per-rank walk
    reaches the same verdict from the same table (`tp_divides`).

    Takes no `ParallelConfig`, and that is the contract rather than an
    omission: classification asks which knob OWNS a weight, and the answer
    must not change when that knob happens to be set to 1.
    """
    cls = step_class(prev_class, lt.role)
    if cls == "ffn_moe":
        cls = moe_child_class(lt.role, routed)
    if _tp_axis_placement(lt):
        from backend.arch.eval.transforms import tp_divides
        if not (lt.dim_symbols & tp_divides()):
            return "replicated"
    return cls or "aux"


def vocab_kind_of_leaf(lt: LeafTensor) -> Optional[str]:
    """Whether a top-level leaf is one of the model's vocab endpoints, by the
    `boundary_role` its IR node declares: `"embed"` for the token embedding
    (`input_embedding`), `"lm_head"` for the terminal vocab projection
    (`output_head`), else None.

    Only the weight breakdown's DISPLAY class reads this: it picks the card a
    top-level parameter shows under. Placement is independent (it reads
    `weight_axes`). Asking the declaration rather than the name or the size
    means a leaf that merely shares the vocab size — a nested MTP head's own
    embedding table — shows as `aux`, which is what it is, and a model that
    spells `embed_tokens` / `lm_head` otherwise is still found."""
    boundary_role = lt.ir_node.boundary_role
    if boundary_role == "input_embedding":
        return "embed"
    if boundary_role == "output_head":
        return "lm_head"
    return None
