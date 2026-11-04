"""ParallelConfig — user knobs for parallel deployment.

The config only states degrees and schemes. Which axes actually shard each
tensor is decided by the layout walk (`propagation.LayoutPropagation`), and
per-rank weight/KV bytes are read off that walk, not divided from these
numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Dict, List, Mapping

from backend.arch import dims
from backend.arch.eval.refusal import DeploymentRefusal
from backend.arch.sharding import degree_in_phase

from . import mesh


class CpScheme(StrEnum):
    """Prefill context-parallel attention scheme. Decode is always DCP."""

    ULYSSES = "ulysses"
    """all_to_all seq-sharded ↔ head-sharded on the attention boundary.
    Modeled for comparison; not the default, as no LLM serving framework
    runs it (vLLM's prefill CP is PCP — config/parallel.py
    `prefill_context_parallel_size`, no head all-to-all in vllm/ — and
    sglang's srt/layers/cp zigzag / interleave cut rows too)."""

    PCP = "pcp"
    """Queries stay sequence-cut; each rank all_gathers the cached
    tensors and attends the full context from local cache (vLLM PCP,
    sglang zigzag/interleave). Does not shard KV storage. The default:
    the scheme the serving frameworks run."""

    LAYERSPLIT = "layersplit"
    """Same attention geometry as PCP; KV occupancy is striped across
    the CP group at min(L, ceil(L/cp)+1) layers per rank. Decode
    broadcast of a remote layer is not priced."""


def stage_layer_ranges(n_layers: int, pp: int) -> List[range]:
    """The contiguous layer range each pipeline stage holds.

    vLLM's partition, from `vllm/distributed/utils.py::get_pp_indices`: an even
    split, with the remainder handed out from the second-to-last stage
    backwards, so the first and last stages — which also carry the embeddings
    and the LM head — are the last to be given an extra layer. The rule matters
    because it is what decides the HEAVIEST stage, and the heaviest stage is
    what has to fit.
    """
    base, rem = divmod(n_layers, pp)
    sizes = [base] * pp
    for i in range(2, rem + 2):
        sizes[-i] += 1
    out: List[range] = []
    start = 0
    for size in sizes:
        out.append(range(start, start + size))
        start += size
    return out


@dataclass
class ParallelConfig:
    pp: int = 1                # pipeline stages (layers split across)
    tp_attention: int = 1      # shard attention heads
    tp_dense_ffn: int = 1      # shard dense FFN intermediate
    tp_moe_expert: int = 1     # shard each MoE expert's intermediate
    ep_moe: int = 1            # distribute MoE experts across ranks
    # The two context parallelisms, which are different deployments and not
    # one knob read in two phases — vLLM carries them as two fields for the
    # same reason:
    #   cp  — PREFILL context parallel (`prefill_context_parallel_size`).
    #     SPANS RANKS OF ITS OWN: `world_size = pp * tp * pcp`. Shards the
    #     token stream, replicates every weight, and crosses the attention
    #     boundary with `cp_scheme` below.
    #   dcp — DECODE context parallel (`decode_context_parallel_size`).
    #     ADDS NO RANKS: it partitions `tp_attention`'s, so `tp_attention %
    #     dcp == 0`, and vLLM states it outright ("DCP does not expand the
    #     process world size. Without PCP, DCP reuses TP ranks"). It shards
    #     the KV cache and nothing else — the query and output projections
    #     stay `n_heads / tp_attention` — and the ranks LSE-combine.
    # Both may be > 1 at once; the cache is then spread over `cp * dcp` ranks.
    cp: int = 1
    dcp: int = 1
    dp_attention: int = 1      # DP-attention: each rank has full heads
    dp_replicas: int = 1       # full-model data-parallel replicas
    # Megatron sequence parallelism: norm/residual region runs Sharded(s_ctx).
    # Pure deployment choice (the layout walk puts the residual sequence
    # axis on the tp_attention family) — same TP world sizes, different comm shape
    # (2 all_gather + 2 reduce_scatter per block instead of 2 all_reduce).
    sequence_parallel: bool = False
    # Prefill attention scheme when cp > 1. Decode is always DCP when
    # dcp > 1 — not a value of this knob.
    cp_scheme: CpScheme = CpScheme.PCP
    # Degrees for rank groups that have no knob of their own above, keyed by
    # `World.label` like every other world-keyed table here. A world is
    # declared in `dims.yaml` (which axes live in it, which mesh it is a
    # subgroup of) and runs at 1 until a deployment names it here, so a new
    # kind of splitting becomes expressible without a new field and without
    # moving any number that exists today. DSA's indexer is the standing
    # example: vLLM and sglang both build it whole on every rank.
    world_degrees: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.cp_scheme, CpScheme):
            return
        try:
            self.cp_scheme = CpScheme(self.cp_scheme)
        except (ValueError, TypeError):
            # Leave the raw value; validate() reports it. Construction
            # stays the same shape as the other knobs.
            pass

    def for_program(self, program) -> "ParallelConfig":
        """This deployment as `program` runs it.

        A mesh region the model has no node in spans no ranks: a model that
        routes no tokens to experts (`Program.has_dispatch_zone` false) has
        no MoE region, so the axes only that region spans run at 1. Without
        this a dense model asked for `tp_moe_expert = 2` beside `cp = 2`
        was refused for an expert mesh it does not have, and an expert
        degree larger than the dense stage inflated the cluster it priced.
        """
        if program.has_dispatch_zone:
            return self
        kept = {axis for region, axes in mesh.REGION_AXES.items()
                if region != "moe" for axis in axes}
        return replace(self, **{axis: 1 for axis in mesh.REGION_AXES["moe"]
                                if axis not in kept})

    @property
    def layer_split_over(self) -> int:
        """CP ranks that stripe stored KV. 1 unless scheme is layersplit.

        Occupancy is a property of this deployment's `cp_scheme` and `cp`,
        not of the attention form of the current phase. `degree_of_world`
        is read without `phase` so decode still sees the declared cp.
        """
        if self.cp_scheme != CpScheme.LAYERSPLIT:
            return 1
        return max(1, self.degree_of_world("cp"))

    def degree_of_world(self, world: str, *, phase: str | None = None) -> int:
        """What degree the rank group LABELLED `world` runs at here.

        The one place a world's degree is read. A world with a `degree:`
        field takes it from that field; one without takes it from
        `world_degrees`; anything nobody has named runs at 1 — which is what
        makes "replicated" arithmetic rather than a declaration.

        `phase` applies `dims.yaml` `phases:` — a world that splits
        nothing in this phase answers 1. Omit it to read the knob as
        declared, independent of serving phase.
        """
        spec = dims.world_by_label().get(world)
        if spec is None:
            raise KeyError(
                f"no world labelled {world!r} in dims.yaml — a rank group is "
                f"declared there before a deployment can set its degree. "
                f"Known: {sorted(dims.world_by_label())}"
            )
        if spec.degree:
            # No getattr default: a misspelled field would otherwise read as
            # degree 1 — every axis in the world silently whole, the knob
            # silently inert.
            degree = max(1, int(getattr(self, spec.degree) or 1))
        else:
            degree = max(1, int(self.world_degrees.get(world, 1) or 1))
        if phase is None:
            return degree
        return degree_in_phase(world, degree, phase)

    def degree_of_symbol(self, symbol: str, *, phase: str) -> int:
        """What this axis is divided by, per rank, in this phase."""
        world = dims.world_of(symbol)
        if world is None:
            return 1
        return self.degree_of_world(world.label, phase=phase)

    def layout_binds(self) -> Dict[str, int]:
        """Scalars the layout walk reads that are not cuts.

        `dp_attention` multiplies MoE token accounting; the request
        family recovers cluster B from the same number. A cut would
        divide an axis; this does not.
        """
        return {"dp_attention": max(1, int(self.dp_attention or 1))}

    def axis_degrees(self, *, phase: str) -> Dict[str, int]:
        """Symbol → divisor in this phase, omitting bound-per-rank worlds."""
        by_rank = {
            label for label, world in dims.world_by_label().items()
            if world.bound_per_rank
        }
        out: Dict[str, int] = {}
        for symbol, world in dims.world_labels().items():
            if world in by_rank:
                continue
            live = any(
                self.degree_of_symbol(symbol, phase=p) > 1
                for p in ("prefill", "decode")
            )
            if live:
                out[symbol] = self.degree_of_symbol(symbol, phase=phase)
        return out

    #: The world Megatron sequence parallelism cuts the residual stream's
    #: sequence axis over: the attention tensor-parallel group, whose
    #: all-reduces it splits into a reduce-scatter and an all-gather.
    SEQUENCE_PARALLEL_WORLD = "tp_attention"

    def refusals(self, phase: str) -> List[DeploymentRefusal]:
        """The deployment rules the layout walk of `phase` refuses this
        config by, whatever the model: the one statement of each. The walk
        (`LayoutPropagation.of`) raises the first; `validate` reports them
        for every phase the deployment serves. A rule that needs the model
        to decide is the walk's alone, and is not stated here."""
        if not self.sequence_parallel:
            return []
        out: List[DeploymentRefusal] = []
        world = self.SEQUENCE_PARALLEL_WORLD
        if self.degree_of_world("cp", phase=phase) > 1:
            # One residual stream, one sequence cut: the walk carries a
            # single residual sequence family, and composing the two cuts
            # (`tp_attention` inside each `cp` shard) is not modelled.
            out.append(DeploymentRefusal(
                f"sequence_parallel and prefill context parallelism (cp = "
                f"{self.cp}) both cut the residual stream's sequence axis "
                f"in {phase}; run one of them",
                knobs=("sequence_parallel", "cp"), phase=phase))
        if self.degree_of_world(world) <= 1:
            out.append(DeploymentRefusal(
                f"sequence_parallel cuts the residual sequence over the "
                f"attention tensor-parallel group ({world}), so it requires "
                f"{world} > 1",
                knobs=("sequence_parallel", world), phase=phase))
        return out

    def validate(self) -> list[str]:
        """Return human-readable error strings (empty if valid)."""
        errs: list[str] = []
        for name in ("pp", "tp_attention", "tp_dense_ffn", "tp_moe_expert",
                     "ep_moe", "cp", "dcp", "dp_attention", "dp_replicas"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                errs.append(f"{name} must be a positive integer, got {v!r}")
        if not isinstance(self.cp_scheme, CpScheme):
            names = ", ".join(repr(s.value) for s in CpScheme)
            errs.append(
                f"cp_scheme must be one of {names}, got "
                f"{self.cp_scheme!r}")
        for world, degree in (self.world_degrees or {}).items():
            spec = dims.world_by_label().get(world)
            if spec is None:
                errs.append(
                    f"world_degrees names {world!r}, which labels no `worlds:` "
                    f"entry in dims.yaml — a rank group nothing knows the axes "
                    f"of has nothing to split (known: "
                    f"{sorted(dims.world_by_label())})")
            elif spec.degree:
                errs.append(
                    f"world_degrees sets {world!r}, which already has the "
                    f"field {spec.degree!r} — set that instead, so one world "
                    f"has one degree")
            elif not isinstance(degree, int) or degree < 1:
                errs.append(
                    f"world_degrees[{world!r}] must be a positive integer, "
                    f"got {degree!r}")
        # A subgroup world partitions ranks its parent region already spans,
        # so its degree comes from the mesh's own answer to "which groups can
        # this fill order build" (`mesh.subgroup_degrees`) — asking for
        # anything else is asking for ranks nobody allocated, or for a group
        # the runtime will not form. Derived from `subgroup_of` rather than
        # written as a `dcp` rule, so the next subgroup is a registry entry;
        # and read off the mesh rather than as `parent % degree`, because the
        # parent is a REGION and not one axis — with prefill CP running, `dcp`
        # may span the CP column instead of dividing `tp_attention`, which the
        # divisibility test refused and vLLM allows.
        for slug, spec in dims.worlds().items():
            if not spec.subgroup_of or spec.degree not in mesh.SUBGROUP_AXES:
                continue
            degree = self.degree_of_world(spec.label)
            legal = mesh.subgroup_degrees(self, spec.degree)
            if degree not in legal:
                errs.append(
                    f"{spec.degree or slug} = {degree} is not a group this "
                    f"mesh can form: it partitions ranks "
                    f"{spec.subgroup_of} already spans rather than adding "
                    f"any, so it must span whole columns of "
                    f"{mesh.SUBGROUP_AXES[spec.degree][1]} (legal here: "
                    f"{list(legal)})")

        if errs:
            return errs
        for phase in dims.PHASES:
            for refusal in self.refusals(phase):
                if str(refusal) not in errs:
                    errs.append(str(refusal))

        # Within a pipeline stage, each phase organizes ranks differently:
        #   attention — tp_attn × cp × dp_attention.
        #   dense FFN — tp_dense_ffn × cp × dp_attention. cp shards the
        #     SEQUENCE and replicates the FFN weight, so it spans the
        #     dense ranks too (token-local FFN: each rank does s_ctx/cp
        #     tokens) but never divides the weight. Dense has no dedicated
        #     dp knob; batches inherited from attention drive its DP
        #     dimension, so dp_dense == dp_attention.
        #   MoE — ep_moe × tp_moe_expert. EP + expert-TP alone span the
        #     whole stage; all ranks see every batch via all-to-all
        #     routing (which also redistributes the cp-sharded tokens, so
        #     cp aliases ep here — not a separate factor), no dp_moe knob.
        # All three must equal per_stage (otherwise ranks idle). We
        # only enforce this when the group > 1 to keep the all-default
        # (per_stage=1) case trivially valid.
        per_stage = self.per_stage_size
        attn_group  = mesh.group_size(self, "attention")
        dense_group = mesh.group_size(self, "dense")
        moe_group   = mesh.group_size(self, "moe")

        if attn_group > 1 and attn_group != per_stage:
            errs.append(
                f"attention parallelism (tp_attention × cp × dp_attention "
                f"= {self.tp_attention} × {self.cp} × {self.dp_attention} "
                f"= {attn_group}) must equal per-stage size ({per_stage}); "
                f"{per_stage - attn_group} rank(s) would idle during "
                f"attention — raise dp_attention to "
                f"{per_stage // (self.tp_attention * self.cp)} to fill"
            )

        if dense_group > 1 and dense_group != per_stage:
            errs.append(
                f"dense FFN parallelism (tp_dense_ffn × cp × dp_attention "
                f"= {self.tp_dense_ffn} × {self.cp} × {self.dp_attention} "
                f"= {dense_group}) must equal per-stage size ({per_stage}); "
                f"cp shards tokens and replicates the FFN weight (it spans "
                f"the dense ranks but does NOT divide the weight), and dense "
                f"inherits dp_attention — keep tp_dense_ffn == tp_attention "
                f"rather than folding cp into it"
            )

        if moe_group > 1 and moe_group != per_stage:
            errs.append(
                f"MoE parallelism (ep_moe × tp_moe_expert "
                f"= {self.ep_moe} × {self.tp_moe_expert} = {moe_group}) "
                f"must equal per-stage size ({per_stage}); EP + expert-TP "
                f"span the whole stage (all ranks handle every batch "
                f"via all-to-all — no separate dp_moe)"
            )
        return errs

    @property
    def per_stage_size(self) -> int:
        """Ranks per pipeline stage (max across attention/dense/MoE worlds).

        Dense FFN spans `tp_dense_ffn × cp × dp_attention` ranks — cp
        (token-shard, weight replicated) and dp_attention (no separate
        dp_dense knob) both span the dense ranks without dividing the
        weight. A well-formed config has all three groups equal (or 1 for
        unused axes); `validate()` rejects mismatched non-unity values.

        Group sizes come from the per-region axis tables in `mesh` — the
        single source for the deployment's rank layout.
        """
        return max(
            mesh.group_size(self, "attention"),
            mesh.group_size(self, "dense"),
            mesh.group_size(self, "moe"),
        )

    @property
    def cluster_size(self) -> int:
        """Total GPUs: pp × per_stage × dp_replicas."""
        return self.pp * self.per_stage_size * self.dp_replicas

    @staticmethod
    def from_dict(d: dict) -> "ParallelConfig":
        allowed = {"pp", "tp_attention", "tp_dense_ffn", "tp_moe_expert",
                   "ep_moe", "cp", "dcp", "dp_attention", "dp_replicas"}
        d = d or {}
        kwargs = {k: int(v) for k, v in d.items()
                  if k in allowed and v is not None}
        if d.get("sequence_parallel") is not None:
            kwargs["sequence_parallel"] = bool(d["sequence_parallel"])
        if d.get("cp_scheme") is not None:
            kwargs["cp_scheme"] = d["cp_scheme"]
        if d.get("ep_load_factor") is not None:
            raise ValueError(
                "ep_load_factor is the batch's routing skew, a workload "
                "input, not a deployment field: pass it beside the workload "
                "(the estimate's `ep_load_factor`)")
        if d.get("world_degrees"):
            kwargs["world_degrees"] = {str(k): int(v) for k, v
                                       in dict(d["world_degrees"]).items()
                                       if v is not None}
        return ParallelConfig(**kwargs)

    def to_dict(self) -> dict:
        """The fields `from_dict` reads, so a deployment round-trips."""
        return {
            "pp": self.pp, "tp_attention": self.tp_attention,
            "tp_dense_ffn": self.tp_dense_ffn,
            "tp_moe_expert": self.tp_moe_expert, "ep_moe": self.ep_moe,
            "cp": self.cp, "dcp": self.dcp, "dp_attention": self.dp_attention,
            "dp_replicas": self.dp_replicas,
            "sequence_parallel": self.sequence_parallel,
            "cp_scheme": str(self.cp_scheme),
            "world_degrees": dict(self.world_degrees),
        }

    def during(self, phase: str) -> "ParallelStep":
        """This deployment, running one phase — see `ParallelStep`."""
        return ParallelStep(parallel=self, phase=phase)


@dataclass(frozen=True)
class ParallelStep:
    """A deployment AND which phase it is running: the pair the comm
    derivation needs, carried as one thing so it cannot arrive half-formed.

    The two used to travel as adjacent parameters — `(parallel_config, phase)`
    on five entry points, plus a `phase` on every policy's constructor and the
    config on its `assign_modes`. That is a drift surface with a silent
    failure: `phase` defaulted to prefill, so a caller that dropped it did not
    raise, it priced the wrong LAYOUT. It happened. `validate_comm_closure`
    spent a while walking a decode tree against the prefill layout, agreeing
    with nothing, and only surfaced because a test grew an argument.

    Undefaultable by construction is the point. `tp` / `ep` / `cp` / `dcp` are
    never dropped on the way down because nobody passes them one at a time;
    the phase now travels the same way. `ParallelConfig.during(phase)` is the
    only way to make one, which reads at the call site as the sentence it is:
    this deployment, during decode.

    A deployment is NOT phase-scoped and a step is — one `ParallelConfig`
    prices both phases of the same serving setup, which is why the phase is
    here and not a field on `ParallelConfig`. It mirrors
    `eval.transforms.Deployment.during`, deliberately: the two layers name the
    same act the same way.

    One phase per step, which is an ASSUMPTION and not a fact. vLLM's chunked
    prefill hands a single batch both split prefill tokens and replicated
    decode ones (`attention/pcp.py::_gather_prefill_cache_inputs` partitions
    the prefill tail and hands the decode prefix back untouched), so a real
    mixed step is both at once. Everything here — and `inference.phases`, and
    the SOL numbers — prices a step as wholly one or wholly the other. That is
    the right scope for a steady-state roofline, whose question is "what does
    a step of THIS shape cost", and it is the wrong scope for a scheduler
    question. Modelling a step as a token COMPOSITION (`{prefill_tokens,
    decode_tokens}`, with PCP cutting only the first) is the honest
    generalisation and is recorded as future work rather than half-built here.
    """

    parallel: ParallelConfig
    phase: str

    def __post_init__(self) -> None:
        if self.phase not in dims.PHASES:
            raise ValueError(
                f"unknown phase {self.phase!r}, not one of "
                f"{list(dims.PHASES)}. The registry names the phases a world "
                f"may restrict itself to (`dims.yaml: phases:`), and a step "
                f"spelled differently would match no restriction and so run "
                f"every world at full degree — silently, and in the direction "
                f"that looks like more parallelism."
            )
