"""Weight-axis and weight-copy endpoint contracts."""
from __future__ import annotations

from typing import (Any, Dict, List, Literal, Optional)

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)




class WeightAxisPlacementSpec(BaseModel):
    """A direct or symbol-dispatched runtime placement for one axis."""

    model_config = ConfigDict(extra="forbid")

    replicated: bool = False
    world: Optional[str] = None
    parts: Optional[str] = None
    fit: Optional[Literal["exact", "replica_cells", "pad"]] = None
    placement_by_symbol: Dict[str, "WeightAxisPlacementSpec"] = Field(
        default_factory=dict)

    def _validate_placement(self) -> None:
        split = bool(self.world)
        if self.placement_by_symbol:
            if self.replicated or split or self.parts or self.fit:
                raise ValueError(
                    "symbol-dispatched weight axis cannot declare another layout")
            if any(not symbol for symbol in self.placement_by_symbol):
                raise ValueError("weight-axis placement symbols cannot be empty")
            return
        if self.replicated:
            if split or self.parts or self.fit:
                raise ValueError(
                    "replicated weight axis cannot also declare split fields")
            return
        if not split or not self.parts or not self.fit:
            raise ValueError(
                "weight-axis placement must declare `replicated: true`, a "
                "symbol dispatch, or a split with world + parts + fit")

    @model_validator(mode="after")
    def _one_placement_form(self) -> "WeightAxisPlacementSpec":
        self._validate_placement()
        return self



class WeightAxisSegmentSpec(WeightAxisPlacementSpec):
    """One named slice of a fused checkpoint axis and its placement."""

    segment: str
    extent: str

    @field_validator("segment", "extent")
    @classmethod
    def _nonempty_segment_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("weight-axis segment and extent must be non-empty")
        return value



class WeightAxisSpec(BaseModel):
    """One axis of a persistent weight's runtime endpoint contract.

    This is deliberately an occurrence declaration.  ``dims.yaml`` says only
    which worlds MAY shard a symbol; this object says which world this stored
    axis actually selects, which symbol counts its parts, and how a degree
    that does not divide that part count is handled.
    """

    model_config = ConfigDict(extra="forbid")

    replicated: bool = False
    world: Optional[str] = None
    parts: Optional[str] = None
    fit: Optional[Literal["exact", "replica_cells", "pad"]] = None
    # Reusable components may bind one local axis name to different model
    # symbols.  This table makes every such choice explicit: the key is a
    # resolved model symbol present in the axis expression and the value is
    # its complete placement.  It is a dispatch table, not capability
    # inference; zero or multiple matching keys fail at instantiation.
    placement_by_symbol: Dict[str, WeightAxisPlacementSpec] = Field(
        default_factory=dict)
    # A fused checkpoint axis is a named, ordered list.  Each child declares
    # its own extent and placement; arithmetic spelling is not a segment
    # contract and is never parsed to invent these boundaries.
    segments: Optional[List[WeightAxisSegmentSpec]] = None

    @model_validator(mode="after")
    def _one_layout_form(self):
        if self.segments is not None:
            if (self.replicated or self.world or self.parts or self.fit
                    or self.placement_by_symbol):
                raise ValueError(
                    "segmented weight axis cannot also declare a direct placement"
                )
            if not self.segments:
                raise ValueError("segmented weight axis must contain at least one segment")
            return self
        WeightAxisPlacementSpec._validate_placement(self)
        return self



#: How the checkpoint stores an expert bank's expert axis — the leading
#: `n_expert_copies` axis its weights declare in their `shape:` — the one
#: thing about the bank the spec has to say beyond its shapes: `unrolled` —
#: one entry per expert, the expert's index in the name just before the
#: weight's own module (`experts.17.gate_proj`); `leading_axis` — one entry
#: whose leading axis is the experts. Where the axis is placed is its
#: `weight_axes` entry, like any other axis.
ExpertWeight = Literal["unrolled", "leading_axis"]


def check_stored_axes(
    stored_axes: Optional[List[int]],
    weight_axes: Optional[List[WeightAxisSpec]],
) -> None:
    """The load-time grammar of an occurrence's ``stored_axes``.

    ``stored_axes[i] = j`` says checkpoint axis ``i`` is the weight's axis
    ``j`` — the same positional coordinates a sibling ``TensorDecl.axis_refs``
    uses, leading axes included — so a packed expert bank stored ``[E, in,
    out]`` under a ``[E, out, in]`` weight writes ``[0, 2, 1]``. The facts that
    need the op and the stored tensors (rank, owner, format) are checked where
    those are known, by ``layout.weight_endpoints.WeightGeometry``.
    """
    if stored_axes is None:
        return
    rank = len(stored_axes)
    if not rank or sorted(stored_axes) != list(range(rank)):
        raise ValueError(
            f"`stored_axes` {stored_axes!r} must be a permutation of "
            f"0..{rank - 1 if rank else 0}: checkpoint payload axis i is the "
            "op weight-shape axis stored_axes[i]")
    if weight_axes is None:
        raise ValueError(
            "`stored_axes` requires `weight_axes`: it orders the axis "
            "contracts of a declared stored weight")
    if len(weight_axes) != rank:
        raise ValueError(
            f"`stored_axes` names {rank} axes but `weight_axes` declares "
            f"{len(weight_axes)}; both list the checkpoint payload axes")


class StoredSplitSpec(BaseModel):
    """The checkpoint stores this one logical tensor as `parts` tensors, cut
    along its logical weight axis `axis` into equal consecutive blocks.

    Qwen3.8-Flash-Next's n-gram table is one `nn.Embedding(320001536, 160)`
    written as 128 tensors `ngram_embedding.shard_{i}.weight`, each
    `[2500012, 160]`, which transformers concatenates back along dim 0 at load
    (`conversion_mapping.py` `Concatenate(dim=0)`). The model reads one
    table; the file holds `parts`. Beside `stored_axes`: both say how the file
    lays out a tensor the op computes with whole, and neither changes the
    runtime tensor (`weight_axes` still describes it).

    `axis` is a position in the op's `weight_shape`; with `stored_axes` the
    tensor is cut first and each part's axes are then ordered as declared.
    Shard i is named `<module>.<segment with {i}>` + the weight's suffix.
    """

    model_config = ConfigDict(extra="forbid")

    axis: int
    parts: str
    segment: str

    @field_validator("axis")
    @classmethod
    def _axis_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                "`stored_split.axis` is a position in the op's weight shape "
                "(0 = rows); it cannot be negative")
        return value

    @field_validator("parts", mode="before")
    @classmethod
    def _parts_expression(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("`stored_split.parts` names the number of stored tensors")
        if text.isdigit() and int(text) < 2:
            raise ValueError(
                f"`stored_split.parts` is {text}: a tensor the checkpoint "
                "stores whole declares no `stored_split`; remove the field")
        return text

    @field_validator("segment")
    @classmethod
    def _one_index(cls, value: str) -> str:
        if value.count("{i}") != 1 or not value.strip():
            raise ValueError(
                f"`stored_split.segment` {value!r} must contain `{{i}}` exactly "
                "once — the shard index in the checkpoint name, e.g. "
                "`shard_{i}`")
        return value

    def segment_name(self, index: int) -> str:
        return self.segment.replace("{i}", str(index))


class StoredWithoutGraphReaderSpec(BaseModel):
    """The checkpoint stores this parameter and no node of the modelled
    graph reads it. `kind` says which of two facts that is:

    - `forward_unread`: the forward never reads it; the framework loads it
      and ignores it (Kimi-K3's layer-0 `self_attention_res_*`: the AttnRes
      history is empty there, so the attention-side combine does not run;
      HunyuanOCR's `image_sep`, read by no PatchMerger forward);
    - `unmodelled_reader`: the forward reads it, in a step the graph does
      not model (Qwen3.8-Flash-Next's PLE index tables, read by the n-gram
      id hash — integer index math on the token ids, which the graph models
      no more than an embedding's token ids).

    Either way the tensor is stored like any other — it counts in the
    resident weights, the checkpoint ledger and the coverage gate — and no
    graph node is billed a read of it. Only a parameter source (an op whose
    output is its stored tensor) declares it; validate holds the declaration
    to the graph (`validate._unread_parameters`): an undeclared parameter
    with no graph reader, or a declared one a node reads, is an error.

    `reason` says why no graph node reads it; `source_anchor` cites the
    forward source (file:line) that shows it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["forward_unread", "unmodelled_reader"]
    reason: str
    source_anchor: str

    @field_validator("reason", "source_anchor")
    @classmethod
    def _stated(cls, value: str, info: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError(
                f"`stored_without_graph_reader.{info.field_name}` is empty: "
                f"say why no graph node reads the tensor (`reason`) and cite "
                f"the forward source that shows it (`source_anchor`, "
                f"file:line)")
        return text


def check_stored_split(
    stored_split: Optional[StoredSplitSpec],
    expert_weight: Optional[str],
) -> None:
    """The load-time grammar of an occurrence's ``stored_split``; the facts
    that need the op and the bound dims are checked where the tensor is
    built."""
    if stored_split is None:
        return
    if expert_weight is not None:
        raise ValueError(
            "`stored_split` cuts one stored weight into parts; an expert bank's "
            "stacked axis is laid out by `expert_weight`, and how a split "
            "composes with it is not declared — declare one or the other")
