"""CacheSpec — what a component keeps between decode steps."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from backend.arch import cache_families
from backend.arch.cache_formats import CacheFormat
from backend.arch.dim_resolution import config_path_value
from backend.arch.dtypes import UnknownDtype, canonical


#: The fallback a config-bound `dtype:` may end on: the precision the model
#: states for its own tensors (`quant_context.default_dtype`, then
#: `config.torch_dtype`).
MODEL_DTYPE = "model"


def _dtype_binding(text: str) -> List[str]:
    """The `??` chain of a cache entry's `dtype:`, validated.

    Either one dtype spelling (`fp32`, `BF16`), or config fields tried in
    order and ending, optionally, on a dtype spelling or `model`:
    `config.mamba_ssm_cache_dtype ?? F32`. The same `config.x ?? y` form
    `dims:` bindings are written in."""
    parts = [part.strip() for part in str(text).split("??")]
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if part.startswith("config.") and len(part) > len("config."):
            continue
        if not last:
            raise ValueError(
                f"cache entry `dtype: {text}`: every term before the last is "
                f"a config field (`config.<path>`); {part!r} is not. Write "
                f"the config fields first and end on a dtype or `model`.")
        if part == MODEL_DTYPE:
            if len(parts) == 1:
                raise ValueError(
                    f"cache entry `dtype: {text}` names no config field. "
                    f"`model` is the fallback after one "
                    f"(`config.<field> ?? model`); an entry that takes the "
                    f"model's precision declares no `dtype:`.")
            continue
        try:
            canonical(part)
        except UnknownDtype:
            raise ValueError(
                f"cache entry `dtype: {text}`: {part!r} is neither a "
                f"config field (`config.<path>`), `model`, nor a dtype "
                f"spelling in `backend.arch.dtypes.DTYPES`.") from None
    return parts



class CacheEntry(BaseModel):
    """One thing a component keeps across decode steps.

    Exactly one of `node` or `state`, because those are the two storage
    laws and a slot obeys one of them:

      • `node` — a node in this component's `graph:` whose resolved
        `shape_out` (at B=S=1) IS the per-token contribution. Storage grows
        one entry per token. Its shape is the node's, never declared.
      • `state` — a fixed-size recurrent state (linear attention / SSM).
        No graph node outputs it, so its `shape:` is written down: the
        state tensor's axes, outermost first, as the HF forward allocates
        it, so the last axis is the one quant scales group over. Each axis
        is a dim expression resolved in the CARRIER's frame — the
        component's dims plus the carrier's own call-site `dims:` — so a
        conv window is written `[C, kernel_size]` and reads the conv's own
        channel count.
        Axes are global extents; what a rank holds follows the carrier's
        own sharding. `carrier` names the graph node whose kernel reads and
        rewrites it every step: the scan owns the recurrent matrix, a
        causal `conv1d` owns the window of `kernel_size` inputs it convolves
        over. A component keeps as many states as it has such kernels, one
        entry each. Its element count is the product of the shape. Billed
        once per request, independent of S.

    `kind` says what the slot holds — `kv`, `latent`, `rope_key`,
    `index_key`, `compressed_kv` for a `node:` entry; `recurrent_state`,
    `conv_state`, `compressor_state`, `token_window` for a `state:` entry
    (`slot_kinds:` in `cache_families.yaml`). A deployment's precision is
    chosen per kind, so it is declared on every entry rather than read off a
    node's name. A `node:` entry naming a sub-component that keeps a cache of
    its own gives the kind of that sub-cache's per-token tensors; its states
    keep their own.

    `dtype` is the precision the model stores this slot at when it is not
    the model's own dtype — a recurrent state the HF forward keeps in F32
    beside a BF16 model. Either a dtype spelling, or a
    config field the checkpoint states it in, tried in order and ending on
    the dtype the HF forward stores the slot at, or on `model` (the model's
    own stated precision): `config.mamba_ssm_cache_dtype ?? F32`. A config
    that states a dtype other than a literal last term is refused
    (`resolved_dtype`). Absent means the model's own dtype.

    `format` is the slot's whole storage format when the HF forward stores
    it quantised: a named cache format (`backend/arch/cache_formats/`, e.g.
    `fp8_group128` for DeepSeek-V3.2's indexer key) or a spelled-out
    `{dtype, scale, transform}`. A slot declares `dtype` or `format`, not
    both. Either is the shipped precision; a deployment's precision profile
    may choose another per slot kind.
    """

    model_config = ConfigDict(extra="forbid")

    node: Optional[str] = None
    state: Optional[str] = None
    kind: Optional[str] = None
    carrier: Optional[str] = None
    shape: Optional[List[Union[str, int]]] = None
    #: How many times one step's carrier reads (and rewrites) the state — a
    #: dim expression in the carrier's frame, like `shape:`. Absent is once:
    #: a scan streams its matrix every step. DSv4's compressor pools its ring
    #: once per window it completes, `s_ctx / compress_ratio` of them a step.
    reads: Optional[Union[str, int]] = None
    dtype: Optional[str] = None
    format: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, exclude_if=lambda v: v is None)

    @field_validator("format")
    @classmethod
    def format_is_a_cache_format(cls, value):
        if value is not None:
            CacheFormat.parse(value, "cache entry `format:`")
        return value

    @field_validator("dtype")
    @classmethod
    def dtype_is_a_spelling_or_binding(cls, value):
        if value is not None:
            _dtype_binding(value)
        return value

    @property
    def name(self) -> str:
        """The key a slot's dtype, kind and cached tensor are looked up by."""
        return self.node if self.node is not None else self.state

    def declared_format(self, config: Optional[Mapping[str, Any]],
                        model_dtype: Callable[[], Optional[str]]
                        ) -> Optional[Any]:
        """This slot's shipped `CacheFormat` for one checkpoint, or None
        when it declares neither `format:` nor `dtype:`."""
        if self.format is not None:
            return CacheFormat.parse(self.format, f"cache entry {self.name!r}")
        dtype = self.resolved_dtype(config, model_dtype)
        return None if dtype is None else CacheFormat.plain(dtype)

    def resolved_dtype(self, config: Optional[Mapping[str, Any]],
                       model_dtype: Callable[[], Optional[str]]
                       ) -> Optional[str]:
        """This slot's stored dtype for one checkpoint, or None when it
        declares none. A literal is returned as written; a config-bound one
        resolves to the canonical name of the first config field present,
        else its fallback. `model_dtype` is asked only when the chain falls
        through to `model`.

        A chain that ends on a dtype ends on the HF forward's own fact — the
        dtype the reference code stores the slot at, and the config class's
        default for the field. The shipped precision is the HF forward's, so
        a checkpoint whose config field states a different dtype is refused
        rather than either source silently winning: the two disagree, and
        the record has to say which one it means (the literal, or a
        deployment precision choice for the framework's dtype)."""
        if self.dtype is None:
            return None
        parts = _dtype_binding(self.dtype)
        if len(parts) == 1 and not parts[0].startswith("config."):
            return self.dtype
        forward = (None if parts[-1].startswith("config.")
                   or parts[-1] == MODEL_DTYPE else canonical(parts[-1]))
        for part in parts:
            if part.startswith("config."):
                value = config_path_value(part[len("config."):], config or {})
                if value is None:
                    continue
                try:
                    stated = canonical(str(value))
                except UnknownDtype:
                    raise UnknownDtype(
                        f"cache entry {self.name!r}: `{part}` is "
                        f"{value!r}, which is not a dtype spelling in "
                        f"`backend.arch.dtypes.DTYPES`.") from None
                if forward is not None and stated != forward:
                    raise ValueError(
                        f"cache entry {self.name!r}: `{part}` is {value!r} "
                        f"({stated}), but `dtype: {self.dtype}` ends on "
                        f"{forward}, the dtype the HF forward stores this "
                        f"slot at. The shipped precision is the HF "
                        f"forward's; a config that disagrees is not picked "
                        f"silently. Declare the literal the forward stores, "
                        f"and leave the framework's {stated} to a "
                        f"deployment precision choice.")
                return stated
            elif part == MODEL_DTYPE:
                stated = model_dtype()
                if stated is None:
                    raise ValueError(
                        f"cache entry {self.name!r}: `dtype: {self.dtype}` "
                        f"falls back to the model's precision, and the model "
                        f"states none (no `quant_context.default_dtype`, no "
                        f"`config.torch_dtype`). End the chain on a dtype.")
                return stated
            else:
                return part
        raise ValueError(
            f"cache entry {self.name!r}: `dtype: {self.dtype}` reads config "
            f"fields this checkpoint does not have, and names no fallback. "
            f"End the chain on a dtype or `model`.")

    @model_validator(mode="before")
    @classmethod
    def elements_is_derived(cls, data):
        if isinstance(data, dict) and "elements" in data:
            raise ValueError(
                f"cache entry {data.get('state') or data.get('node')!r} "
                f"declares `elements: {data['elements']}`. A cache entry's "
                f"size is derived: a node's from its resolved shape, a "
                f"state's as the product of its `shape:`. Replace `elements:` "
                f"with `shape: [...]` listing the state's axes, outermost "
                f"first, as the HF forward allocates it, e.g. "
                f"`shape: [n_lin_v_heads, d_lin_k_head, d_lin_v_head]` for a "
                f"delta-rule matrix and `shape: [C, kernel_size]` for a conv "
                f"window.")
        return data

    @model_validator(mode="after")
    def exactly_one_storage_law(self) -> "CacheEntry":
        if (self.node is None) == (self.state is None):
            raise ValueError(
                "cache entry declares exactly one of `node:` (per-token, "
                "shape from the node's output) or `state:` "
                "(constant recurrent state, `shape:` declared)"
            )
        self._check_kind()
        if self.node is not None and self.shape is not None:
            raise ValueError(
                f"cache entry {self.node!r} is a graph node: its shape is the "
                f"node's resolved output. Declaring `shape:` here is a second "
                f"answer that can disagree with it; delete it."
            )
        if self.node is not None and self.reads is not None:
            raise ValueError(
                f"cache entry {self.node!r} is a graph node: its reads follow "
                f"the cache length law. `reads:` belongs to a `state:` entry.")
        if self.node is not None and self.carrier is not None:
            raise ValueError(
                f"cache entry {self.node!r} is a graph node, which is read and "
                f"written through its own edges. `carrier:` belongs to a "
                f"`state:` entry; delete it here."
            )
        if self.state is not None and self.shape is None:
            raise ValueError(
                f"recurrent state {self.state!r} has no `shape:`. No graph "
                f"node outputs a state, so its axes are written here: list "
                f"them outermost first, as the HF forward allocates the "
                f"state, as global extents, e.g. `shape: [n_lin_v_heads, "
                f"d_lin_k_head, d_lin_v_head]` for a delta-rule matrix and "
                f"`shape: [C, kernel_size]` for a conv window."
            )
        if self.state is not None and not self.shape:
            raise ValueError(
                f"recurrent state {self.state!r} declares `shape: []`. A "
                f"state has at least one axis; list each of them."
            )
        if self.state is not None and not self.carrier:
            raise ValueError(
                f"recurrent state {self.state!r} names no `carrier:`. No edge "
                f"reads a state, so the kernel that streams it has to be "
                f"named: add `carrier: <graph node>` — the scan op "
                f"(`gated_delta_rule`, `mamba2_scan`, a custom recurrence) for "
                f"the recurrent matrix, the causal `conv1d` for its conv state."
            )
        return self

    def _check_kind(self) -> None:
        """`kind:` is declared, is in the vocabulary, and lives under the
        storage law this entry declares."""
        storage = "per_token" if self.node is not None else "state"
        allowed = sorted(cache_families.slot_kinds_stored(storage))
        law = "node" if self.node is not None else "state"
        if self.kind is None:
            raise ValueError(
                f"cache entry {self.name!r} declares no `kind:`. Say what "
                f"the slot holds: a `{law}:` entry is one of {allowed} "
                f"(`slot_kinds:` in backend/arch/cache_families.yaml), e.g. "
                f"`{{{law}: {self.name}, kind: {allowed[0]}}}`.")
        known = cache_families.slot_kinds()
        if self.kind not in known:
            raise ValueError(
                f"cache entry {self.name!r} declares `kind: {self.kind}`, "
                f"which is not a cache slot kind. A `{law}:` entry is one of "
                f"{allowed}.")
        if self.kind not in allowed:
            raise ValueError(
                f"cache entry {self.name!r} is a `{law}:` entry and declares "
                f"`kind: {self.kind}`, a kind stored "
                f"{known[self.kind].storage.replace('_', ' ')}. A `{law}:` "
                f"entry is one of {allowed}.")
        if self.dtype is not None and self.format is not None:
            raise ValueError(
                f"cache entry {self.name!r} declares both `dtype:` and "
                f"`format:`; a format names its dtype — keep `format:`.")
        if not known[self.kind].storage_format and self.format is not None:
            raise ValueError(
                f"cache entry {self.name!r} is `kind: {self.kind}`, which no "
                f"storage format applies to; declare its `dtype:` instead.")
        if not known[self.kind].storage_format and self.dtype is None:
            raise ValueError(
                f"cache entry {self.name!r} is `kind: {self.kind}`, which no "
                f"storage format applies to, and declares no `dtype:`; "
                f"without one it would take the model's dtype. "
                f"Declare the dtype it is stored at (`dtype: I64` for token "
                f"ids).")



class CacheSpec(BaseModel):
    """What a component keeps between decode steps, in one place.

    This replaced four parallel top-level fields — `kv_cached`,
    `kv_cache_dtype`, `kv_state_elements`, `kv_label` — that had to be read
    together to know anything and could each be edited without the others.
    The dtype map keyed slots by a name that lived in a different list; the
    recurrent-state expression was a fifth field that meant "ignore the
    first two". Here a slot is one entry and carries its own dtype.

    `family` is the KV-cache classification tag ("MLA", "GQA", "SWA",
    "GatedDeltaNet", …) — a STORAGE fact, deliberately independent of the
    component name, since `gqa_attention` and `gqa_qknorm_attention` are the
    same cache. For the compute-flavor label shown to users, see
    `BlockGraph.attn_variant`.
    """

    model_config = ConfigDict(extra="forbid")

    family: Optional[str] = None
    entries: List[CacheEntry] = []

    @field_validator("entries", mode="before")
    @classmethod
    def entries_declare_their_kind(cls, value):
        """A bare name says which node is cached but not what it holds; every
        entry is a mapping that carries its `kind:`."""
        for item in value or ():
            if isinstance(item, str):
                kinds = sorted(cache_families.slot_kinds_stored("per_token"))
                raise ValueError(
                    f"cache entry {item!r} is a bare node name. Write it as "
                    f"`{{node: {item}, kind: <kind>}}`, where the kind says "
                    f"what the slot holds: one of {kinds} "
                    f"(`slot_kinds:` in backend/arch/cache_families.yaml).")
        return value
