"""Activation formats — how a matmul's activation operand is quantised.

A weight scheme that runs its matmul on quantised activations names one of
these (`activation_options:` in `backend/arch/quant_schemes/*.yaml`): the
activation's element dtype, which decides the MMA the kernel runs at, and
the scales a dynamic quantisation writes beside it. Data-driven like the
scheme registry: one yaml file per format, the stem is the name.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, FrozenSet, Mapping, Tuple

from backend.arch.dtypes import canonical, display_name

@dataclass(frozen=True)
class ActivationFormat:
    #: How the activation's scales are organised. Closed; the loader rejects
    #: anything else.
    GRANULARITIES: ClassVar[Tuple[str, ...]] = ("per-token", "per-group")

    name: str
    dtype: str
    scale_granularity: str
    scale_dtype: str
    group: int
    description: str
    source: str

    @property
    def granularity(self) -> str:
        """Which activation elements share one scale, in the words the weight
        side already uses (`TensorFormat.weight_granularity`): `per-token` —
        one scale for the whole row — or `per-group:<g>` — one per `g`
        elements along it.

        One producer, because this string travels: it is the
        `activation_granularity` slot of a measured row's dtype environment
        (`benchmarks.check_dtype_slots`), and per-token, per-128-group and
        per-32-group are three different kernels on the same two dtypes. A
        consumer that spelled it itself would be restating what the yaml
        declared.
        """
        if self.scale_granularity == "per-group":
            return f"per-group:{self.group}"
        return self.scale_granularity

    def scale_block(self, row: Any) -> Any:
        """How many consecutive elements along a row of width `row` share one
        scale: the group, or the whole row for a per-token organisation. A
        re-indexing that maps every such run onto one such run leaves the
        quantised tensor element for element and scale for scale what it was
        (`fusion.ViewChains`)."""
        return self.group if self.scale_granularity == "per-group" else row

    def label(self) -> str:
        gran = (f"per-token group {self.group}"
                if self.scale_granularity == "per-group" else "per-token")
        return (f"{display_name(self.dtype)} · {gran} · "
                f"{display_name(self.scale_dtype)} scale")

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "label": self.label(), "dtype": self.dtype,
                "scale": {"granularity": self.scale_granularity,
                          "dtype": self.scale_dtype, "group": self.group},
                "description": self.description, "source": self.source}


def _load() -> Dict[str, ActivationFormat]:
    here = os.path.dirname(__file__)
    out: Dict[str, ActivationFormat] = {}
    for fname in sorted(os.listdir(here)):
        if not fname.endswith(".yaml"):
            continue
        name = fname.removesuffix(".yaml")
        with open(os.path.join(here, fname), encoding="utf-8") as f:
            data = load_yaml(f) or {}
        for required in ("description", "source", "dtype", "scale"):
            if not data.get(required):
                raise ValueError(
                    f"activation_formats/{fname}: no `{required}:`.")
        scale = data["scale"]
        if scale.get("granularity") not in ActivationFormat.GRANULARITIES:
            raise ValueError(
                f"activation_formats/{fname}: scale granularity "
                f"{scale.get('granularity')!r} is not one of "
                f"{list(ActivationFormat.GRANULARITIES)}.")
        out[name] = ActivationFormat(
            name=name, dtype=canonical(str(data["dtype"])),
            scale_granularity=str(scale["granularity"]),
            scale_dtype=canonical(str(scale.get("dtype") or "F32")),
            group=int(scale.get("group") or 0),
            description=str(data["description"]),
            source=str(data["source"]))
    return out


#: Every named activation format, by file stem.
ACTIVATION_FORMATS: Mapping[str, ActivationFormat] = _load()

#: The granularities a declared format actually states — the vocabulary a
#: measured row's `activation_granularity` slot is validated against. DERIVED
#: from the formats, not listed beside them: a cell's granularity comes from
#: the format the checkpoint runs under, so a string outside this set is a
#: cell no declaration produced.
ACTIVATION_GRANULARITIES: FrozenSet[str] = frozenset(
    fmt.granularity for fmt in ACTIVATION_FORMATS.values())
