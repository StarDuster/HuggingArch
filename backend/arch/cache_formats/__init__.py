"""Cache storage formats — data-driven from the yaml files in this package.

A cached tensor is stored in a FORMAT, not merely a dtype: the element
dtype, how its quant scales are organised (how many, over what, at what
width), whether the write rotates it first, and whether the last channels
of each row stay unquantised (`Tail`). One `CacheFormat` per cached tensor;
a slot that declares nothing is a format with no scale.

The named formats here are what a precision profile's cache rule may name
(`fp8_per_tensor`, `fp8_group128`, ...), and what a component's cache entry
may declare with `format:` when the HF forward stores the slot quantised.
Adding one is a file, not a branch.

Scale counts per copy of one cached tensor, per granularity:

    granularity     static (once per tensor)   dynamic (per stored position)
    per-tensor      1                          0
    per-channel     elements per position      0
    per-token-head  0                          rows (heads; 1 without a head axis)
    per-group       0                          rows x ceil(inner / group)
    per-tile        0                          rows x ceil(inner / group) / group

`inner` is the tensor's innermost (channel) extent and `rows` the product of
its other non-token axes, both read off the walk's shape (`CachedTensor`).
With a tail, `inner` is the quantised channels only: the tail carries no
scale.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import os
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

from backend.arch.dtypes import UnknownDtype, canonical, display_name, dtype_bytes

@dataclass(frozen=True)
class Scale:
    """How a quantised tensor's scales are organised."""

    #: The scale organisations a format may declare. Closed: the loader and
    #: a precision rule both reject anything else.
    GRANULARITIES: ClassVar[Tuple[str, ...]] = (
        "per-tensor", "per-channel", "per-token-head", "per-group", "per-tile")

    granularity: str
    dtype: str = "F32"
    group: int = 128
    power_of_two: bool = False

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"granularity": self.granularity,
                               "dtype": self.dtype}
        if self.granularity in ("per-group", "per-tile"):
            out["group"] = self.group
        if self.power_of_two:
            out["power_of_two"] = True
        return out

    @property
    def rides_positions(self) -> bool:
        """Whether every stored position carries its own scales by this
        organisation alone (one per token per head) — something a state,
        which has no positions, cannot hold."""
        return self.granularity == "per-token-head"


@dataclass(frozen=True)
class Tail:
    """The last `channels` of every row, stored as they are at `dtype`
    beside the quantised rest. DeepSeek-V4 writes its KV with
    `act_quant(kv[..., :-rd], 64, ...)`: the RoPE tail stays BF16.

    `channels` is a dim expression (`d_qk_rope`) until the cache binds it in
    the frame of the component that keeps the tensor (`bound`), then a
    number."""

    channels: Any
    dtype: str

    def to_json(self) -> Dict[str, Any]:
        return {"channels": str(self.channels), "dtype": self.dtype}


@dataclass(frozen=True)
class CacheFormat:
    """How one cached tensor is stored: element dtype, scales, transform.

    Equality is by content: a named format and its spelled-out body are
    the same format."""

    #: What a write may do to the tensor before storing it, and the op that
    #: does it. The one definition of both: a stored transform is a launch
    #: of that op on the graph (`inference.materialize_cast` grafts it on
    #: the edge into the cache, and on the operand every reader multiplies
    #: the cached tensor with, since the product is only right in one
    #: basis), never a property of the bytes alone.
    TRANSFORMS: ClassVar[Mapping[str, Optional[str]]] = {
        "none": None, "hadamard": "hadamard_transform"}

    dtype: str
    scale: Optional[Scale] = None
    transform: str = "none"
    tail: Optional[Tail] = None

    @property
    def transform_op(self) -> Optional[str]:
        """The op a write of this format runs on the tensor before storing
        it (`TRANSFORMS`); None when it stores the tensor as computed."""
        return self.TRANSFORMS[self.transform]

    @property
    def element_bytes(self) -> float:
        """Bytes of one quantised element; a tail's are `bytes_per_element`'s."""
        return float(dtype_bytes(self.dtype))

    @property
    def dtypes(self) -> Tuple[str, ...]:
        """Every element dtype a row is stored at: the format's, and its
        tail's when it has one."""
        return (self.dtype,) if self.tail is None else (
            self.dtype, self.tail.dtype)

    def bound(self, channels: Any) -> "CacheFormat":
        """This format with its tail's channel count resolved to `channels`."""
        if self.tail is None:
            return self
        return replace(self, tail=replace(self.tail, channels=channels))

    def tail_channels(self, inner: Optional[int]) -> int:
        """How many of a row's `inner` channels the tail holds: 0 without a
        tail. Refuses an unbound tail, a row whose split the walk did not
        record, and a tail wider than the row."""
        if self.tail is None:
            return 0
        if isinstance(self.tail.channels, str):
            raise ValueError(
                f"cache format tail `{self.tail.channels}` is not bound to "
                f"the frame of the component that keeps the tensor")
        if inner is None:
            raise ValueError(
                "a cache format with an unquantised tail needs the row's "
                "channel extent, and the walk recorded none for this tensor")
        tail = int(self.tail.channels)
        if not 0 <= tail <= int(inner):
            raise ValueError(f"cache format tail of {tail} channels does not "
                             f"fit a row of {inner}")
        return tail

    def bytes_per_element(self, inner: Optional[int]) -> float:
        """Mean bytes per element of a row `inner` channels wide: the
        element width, or with a tail the two widths weighted by how many
        channels each covers."""
        tail = self.tail_channels(inner)
        if not tail:
            return self.element_bytes
        return ((int(inner) - tail) * self.element_bytes
                + tail * float(dtype_bytes(self.tail.dtype))) / int(inner)

    @property
    def scale_bytes(self) -> float:
        return float(dtype_bytes(self.scale.dtype)) if self.scale else 0.0

    def scale_counts(self, elements: float, inner: Optional[int],
                     rows: Optional[int]) -> Tuple[float, float]:
        """`(dynamic_per_position, static)` scale counts for one copy of a
        tensor with `elements` per position, innermost extent `inner` and
        `rows` other non-token rows (numbers, read off the walk's shape).
        Zero for an unscaled format."""
        if self.scale is None:
            return 0.0, 0.0
        tail = self.tail_channels(inner)
        if tail:
            elements -= (rows or 1) * tail
            inner = int(inner) - tail
        g = self.scale.granularity
        if g == "per-tensor":
            return 0.0, 1.0
        if g == "per-channel":
            return 0.0, float(elements)
        if g == "per-token-head":
            return float(rows or 1), 0.0
        width = int(inner or 0)
        count = (rows or 0) * (-(-width // self.scale.group)) if width else 0
        if g == "per-group":
            return float(count), 0.0
        if g == "per-tile":
            return count / self.scale.group, 0.0
        raise ValueError(f"unknown scale granularity {g!r}")

    def to_json(self) -> Dict[str, Any]:
        out = {"dtype": self.dtype,
               "scale": None if self.scale is None else self.scale.to_json(),
               "transform": self.transform}
        if self.tail is not None:
            out["unquantized_tail"] = self.tail.to_json()
        return out

    @property
    def name(self) -> Optional[str]:
        """The registry name this format is, when it is one.

        A tail is compared as bound to this format's own channel count: a
        cached tensor holds its format with the tail resolved in the frame
        of the component that keeps it (`d_qk_rope` → 64), and it is still
        the named format whose tail binds to that row."""
        for name, fmt in CACHE_FORMATS.items():
            if self.tail is not None and fmt.tail is not None:
                fmt = fmt.bound(self.tail.channels)
            if fmt == self:
                return name
        return None

    def label(self) -> str:
        """`FP8 (E4M3) · per-group 128 · F32 scale`, or the dtype alone."""
        text = display_name(self.dtype)
        if self.scale is not None:
            gran = self.scale.granularity
            if gran in ("per-group", "per-tile"):
                gran = f"{gran} {self.scale.group}"
            text += f" · {gran} · {display_name(self.scale.dtype)} scale"
        if self.transform != "none":
            text += f" · {self.transform}"
        if self.tail is not None:
            text += (f" · last {self.tail.channels} channels "
                     f"{display_name(self.tail.dtype)}")
        return text

    @classmethod
    def plain(cls, dtype: str) -> "CacheFormat":
        return cls(dtype=canonical(dtype))

    @classmethod
    def parse(cls, raw: Any, where: str) -> "CacheFormat":
        """A named format (`fp8_per_tensor`) or a spelled-out body
        (`{dtype, scale: {granularity, group, dtype, power_of_two},
        transform, unquantized_tail: {channels, dtype}}`); `where` names the
        declaration in the error."""
        if isinstance(raw, str):
            fmt = CACHE_FORMATS.get(raw)
            if fmt is None:
                raise ValueError(
                    f"{where}: unknown cache format {raw!r}; named formats "
                    f"are {sorted(CACHE_FORMATS)} (backend/arch/cache_formats/)"
                    f", or spell one out as {{dtype, scale, transform}}.")
            return fmt
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{where}: a cache format is a name or a mapping "
                f"{{dtype, scale, transform}}, not {raw!r}.")
        unknown = sorted(set(raw) - {"dtype", "scale", "transform",
                                     "unquantized_tail"})
        if unknown:
            raise ValueError(
                f"{where}: cache format fields {unknown} are not "
                f"`dtype`, `scale`, `transform`, `unquantized_tail`.")
        if "dtype" not in raw:
            raise ValueError(f"{where}: a cache format names its `dtype`.")
        try:
            dtype = canonical(str(raw["dtype"]))
        except UnknownDtype:
            raise ValueError(
                f"{where}: {raw['dtype']!r} is not a dtype spelling in "
                f"backend.arch.dtypes.DTYPES.") from None
        scale = None
        if raw.get("scale") is not None:
            scale = _scale(raw["scale"], where)
        transform = str(raw.get("transform") or "none")
        if transform not in cls.TRANSFORMS:
            raise ValueError(
                f"{where}: transform {transform!r} is not one of "
                f"{list(cls.TRANSFORMS)}.")
        tail = None
        if raw.get("unquantized_tail") is not None:
            tail = _tail(raw["unquantized_tail"], where)
        return cls(dtype=dtype, scale=scale, transform=transform, tail=tail)


def _tail(raw: Any, where: str) -> Tail:
    if not isinstance(raw, Mapping) or set(raw) != {"channels", "dtype"}:
        raise ValueError(f"{where}: `unquantized_tail` is a mapping "
                         f"{{channels, dtype}}, not {raw!r}.")
    try:
        dtype = canonical(str(raw["dtype"]))
    except UnknownDtype:
        raise ValueError(f"{where}: tail dtype {raw['dtype']!r} is not a "
                         f"dtype spelling.") from None
    channels = raw["channels"]
    if not isinstance(channels, (int, str)) or isinstance(channels, bool):
        raise ValueError(f"{where}: tail `channels` is a dim expression or "
                         f"a count, not {channels!r}.")
    return Tail(channels=channels, dtype=dtype)


def _scale(raw: Any, where: str) -> Scale:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: `scale` is a mapping {{granularity, "
                         f"group, dtype, power_of_two}}, not {raw!r}.")
    unknown = sorted(set(raw) - {"granularity", "group", "dtype",
                                 "power_of_two"})
    if unknown:
        raise ValueError(f"{where}: scale fields {unknown} are not "
                         f"`granularity`, `group`, `dtype`, `power_of_two`.")
    gran = raw.get("granularity")
    if gran not in Scale.GRANULARITIES:
        raise ValueError(f"{where}: scale granularity {gran!r} is not one of "
                         f"{list(Scale.GRANULARITIES)}.")
    try:
        sdtype = canonical(str(raw.get("dtype") or "F32"))
    except UnknownDtype:
        raise ValueError(f"{where}: scale dtype {raw.get('dtype')!r} is not "
                         f"a dtype spelling.") from None
    group = int(raw.get("group") or 128)
    if group <= 0:
        raise ValueError(f"{where}: scale group must be positive.")
    return Scale(granularity=str(gran), dtype=sdtype, group=group,
                 power_of_two=bool(raw.get("power_of_two") or False))


def _load() -> Dict[str, CacheFormat]:
    here = os.path.dirname(__file__)
    out: Dict[str, CacheFormat] = {}
    for fname in sorted(os.listdir(here)):
        if not fname.endswith(".yaml"):
            continue
        name = fname.removesuffix(".yaml")
        with open(os.path.join(here, fname), encoding="utf-8") as f:
            data = load_yaml(f) or {}
        for required in ("description", "source", "dtype"):
            if not data.get(required):
                raise ValueError(f"cache_formats/{fname}: no `{required}:`.")
        body = {k: data[k] for k in ("dtype", "scale", "transform",
                                     "unquantized_tail")
                if k in data}
        out[name] = CacheFormat.parse(body, f"cache_formats/{fname}")
    return out


#: Every named cache format, by file stem.
CACHE_FORMATS: Mapping[str, CacheFormat] = _load()


def describe() -> Dict[str, Dict[str, Any]]:
    """`{name: {label, format, description, source}}` — the vocabulary a
    precision editor lists."""
    here = os.path.dirname(__file__)
    out: Dict[str, Dict[str, Any]] = {}
    for name, fmt in CACHE_FORMATS.items():
        with open(os.path.join(here, f"{name}.yaml"), encoding="utf-8") as f:
            data = load_yaml(f) or {}
        out[name] = {"label": fmt.label(), "format": fmt.to_json(),
                     "description": data["description"],
                     "source": data["source"]}
    return out
