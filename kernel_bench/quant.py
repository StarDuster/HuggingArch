"""What a format's grid IS — owned here, not by any kernel.

A weight-only-quantized GEMM (marlin W4A16, humming) is compared against a
reference computed from the CANONICAL bf16 weight. The kernel quantizes that
weight to 4 bits inside its own build, so the comparison includes the entire
4-bit quantization error, and the tolerance it is judged at is keyed on the
ACTIVATION dtype (bf16, 2e-2) — the slot the error does not come from. The
result: 2,124 marlin and 1,980 humming measurements rejected in a single pass,
both libraries contributing zero rows, for kernels that were computing exactly
what they should.

The fix is not to loosen the band — at ~0.8 it would accept GPTQ's uniform int4
grid used for E2M1's non-uniform one (0.213 vs 0.095), which is a real bug this
harness has already met. The fix is to put the canonical weight ON the format's
grid before anyone sees it. Then:

  * the reference is exact and format-aware, and depends on no kernel's code;
  * the kernel's own quantization of an already-representable weight is
    lossless, so what remains is the kernel's arithmetic — which is what the
    activation-dtype band was always the right band for;
  * a kernel that packs with the WRONG grid still cannot reproduce these
    values, so the check keeps every bit of its discriminating power.

Asking the provider to dequantize its own packing would give that last property
away: a kernel that quantizes and dequantizes through the same wrong grid agrees
with itself perfectly.

FIXED POINT, NOT ONE PASS. A scale derived from the group's amax moves when the
group is rounded. For the MX formats it moves far enough to change the answer:
a group with amax 3.41 takes a power-of-two scale of 1.0, its max rounds to the
grid point 3.0, and a second pass then picks scale 0.5. Rounding once would
leave the weight OFF the grid the kernel derives, which is the lossy comparison
this module exists to remove. So `round_to_grid` iterates until the tensor stops
moving and raises if it does not converge — never silently returning a
non-fixed-point.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, Optional

# E2M1: four bits as 0, 0.5, 1, 1.5, 2, 3, 4, 6 (times a sign). NOT evenly
# spaced — which is why a per-group scale, absorbing a common factor, cannot
# turn an int4 grid into this one.
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Weight dtypes that store the value as-is. Nothing to place on a grid —
# the `quantized: False` rows of `arch.dtypes.DTYPES`, not a list of ours.
from backend.arch.dtypes import UNQUANTIZED_TOKENS as UNQUANTIZED

# How many rounds to allow before declaring non-convergence. Observed: 1 for the
# fixed-scale formats, 2 for the MX ones. A ceiling well above that turns a
# pathological input into an error rather than a silent near-miss.
_MAX_ROUNDS = 8


@dataclass(frozen=True)
class WeightFormat:
    """One storage format's grid.

    `group` is the number of elements along the reduction axis that share a
    scale; 0 means one scale for the whole tensor. It is NOT a property of the
    format alone — marlin packs NVFP4 in blocks of 16 while humming groups the
    same element encoding by 128 — so it is supplied by the caller from the
    kernel's own declaration, and the format only says how a group is encoded.
    """
    name: str
    # Largest magnitude the grid represents, in scaled units.
    peak: float
    # Levels the mantissa can take, or None for "whatever the torch cast does".
    levels: Optional[tuple] = None
    # Cast used instead of `levels` (fp8), applied to the scaled value.
    torch_cast: str = ""
    # MX formats constrain the scale to a power of two (e8m0). The others take
    # the scale encoding from the KERNEL — see `ScaleEncoding`.
    pow2_scale: bool = False
    # Two's complement reaches one step further negative than positive: int4 is
    # -8..7, not -7..7. With an EXACT scale of amax/7 the extra level is
    # unreachable, so a symmetric model looks right — but marlin holds its scale
    # in the activation dtype, and a scale rounded DOWN puts |w/scale| above 7,
    # which lands real values on -8. Modelling it symmetrically left int4 at
    # cosine 0.982 against the kernel while fp4 and fp8 sat at 1.00000.
    asymmetric: bool = False

    @property
    def step(self) -> float:
        """The grid's relative spacing: the largest gap between neighbouring
        normal values over the smaller of them — how far one rounding step
        moves a value, at most. `torch.finfo(...).eps` for the cast formats
        (e4m3: 2^-3), the level table's for the others (E2M1: 1 → 1.5), one
        level over the peak for an integer grid."""
        if self.torch_cast:
            import torch
            return float(torch.finfo(getattr(torch, self.torch_cast)).eps)
        if self.levels is not None:
            normal = [v for v in self.levels if v >= 1.0]
            return max((b - a) / a for a, b in itertools.pairwise(normal))
        return 1.0 / self.peak


FORMATS: Dict[str, WeightFormat] = {
    "int4": WeightFormat("int4", peak=7.0, asymmetric=True),     # -8..7
    "int8": WeightFormat("int8", peak=127.0, asymmetric=True),   # -128..127
    "fp4_e2m1": WeightFormat("fp4_e2m1", peak=6.0, levels=_E2M1_LEVELS),
    "mxfp4": WeightFormat("mxfp4", peak=6.0, levels=_E2M1_LEVELS, pow2_scale=True),
    "fp8_e4m3": WeightFormat("fp8_e4m3", peak=448.0, torch_cast="float8_e4m3fn"),
    "fp8_e5m2": WeightFormat("fp8_e5m2", peak=57344.0, torch_cast="float8_e5m2"),
    "mxfp8": WeightFormat("mxfp8", peak=448.0, torch_cast="float8_e4m3fn",
                          pow2_scale=True),
}


# How a kernel STORES the per-group scale. Part of its grid, not decoration: a
# scale rounded to bf16 puts the weight somewhere an exact fp32 scale does not.
# Measured, not assumed — modelling the element levels alone left marlin's NVFP4
# at 0.19 relative error against the reference, still an order of magnitude over
# the bf16 band; modelling the scale too takes the residual to exactly 0.
#
#   "exact"       fp32, as computed (marlin's fp8 per-tensor scale)
#   "activation"  cast to the activation dtype — marlin's int4 path does
#                 `scales.to(dt)`, so on a bf16 cell the scale is bf16
#   "e4m3_global" NVFP4's two levels: an e4m3 block scale times one fp32 global
#                 scale, `bscale = (amax/6/gscale).to(e4m3)`
#   "pow2"        e8m0, a power of two (the MX formats)
SCALE_ENCODINGS = frozenset({"exact", "activation", "e4m3_global", "pow2"})


class UnknownWeightFormat(ValueError):
    """Raised for a weight dtype that is neither unquantized nor described here.

    Deliberately an error and not a pass-through: a format added to the schema
    vocabulary but not to this table would silently keep the old behaviour —
    a lossy comparison judged at the activation band — which is exactly the
    condition this module was written to end, and it would look like nothing
    had changed.
    """


def _pow2_ceil(x):
    """Smallest power of two >= x, computed exactly.

    Via frexp rather than `2 ** ceil(log2(x))`: log2 of an exact power of two
    can land a hair above the integer, and the second rounding pass hits that
    boundary EXACTLY (the group's amax becomes peak x scale by construction),
    so the naive form is wrong precisely where it matters.
    """
    import torch
    m, e = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(m == 0.5, e - 1, e))


def _encode_scale(scale, encoding: str, act_dtype, fmt: WeightFormat):
    """The scale as the kernel will actually hold it."""
    import torch
    if encoding == "exact":
        return scale
    if encoding == "activation":
        return scale.to(act_dtype).float()
    if encoding == "pow2":
        return _pow2_ceil(scale)
    if encoding == "e4m3_global":
        # NVFP4: one fp32 global scale over the whole tensor, and a per-block
        # e4m3 scale relative to it. `scale` arrives as amax/peak, so the global
        # factor divides out and what is quantized is the ratio.
        gscale = (scale.max() / 448.0).clamp_min(1e-12)
        bscale = (scale / gscale).to(torch.float8_e4m3fn).float()
        return bscale * gscale
    raise ValueError(f"unknown scale encoding {encoding!r}")


def quantise_groups(x, fmt: WeightFormat, group: int, encoding: str,
                    act_dtype, eps: float = 1e-30):
    """ONE pass of `fmt`'s grid over `x`: the elements as the format holds
    them, and the scale each group is held against.

    Returns `(q, scale)`, both grouped `[rows, K // g, …]`: `q` is the
    narrow tensor itself where torch has a dtype for the format (the two
    fp8s, `mxfp8`), and the grid's level VALUES in `x`'s working precision
    where it does not (fp4, int4 — those are packed formats, and nothing
    here packs). `scale` is `[rows, K // g]`, fp32.

    One pass, and the one implementation of the grid. A weight is placed on
    the grid by multiplying these back together until it stops moving
    (`round_to_grid`); an ACTIVATION cast kernel's whole output is this pair
    (`ActivationCast.quantise`), and its oracle is the pair multiplied back
    once (`ActivationCast.on_grid`). Written twice, the oracle for a cast
    would be a second opinion about what the cast computes.

    `eps` floors a group's amax so an all-zero group gets a usable scale.
    It is the kernel's, not ours: the activation casts take `eps=1e-10`
    (vLLM `fp8_utils.py:632`, sglang `fp8_kernel.py:599`, both at the
    releases their bindings are audited at), while the weight grid keeps
    the 1e-30 it was measured with.
    """
    import torch
    K = x.shape[-1]
    g = K if not group else int(group)
    if K % g:
        raise ValueError(
            f"the reduction extent {K} is not a multiple of the kernel's "
            f"group {g} — the harness must group exactly as the kernel does or "
            f"its rounding is not the kernel's")
    xg = x.reshape(-1, K // g, g)

    # group 0 means ONE scale for the whole tensor, so the max is taken over
    # everything. Reducing over the last axis alone would give one scale PER ROW
    # — a finer grid than the kernel's, which leaves the weight off it: marlin's
    # per-tensor fp8 path went from 0.12 relative error to 0.15 that way, worse
    # than doing nothing, while the fixed-point check still passed (a per-row
    # grid is a perfectly good fixed point; it is just not this kernel's).
    #
    # The amax is read in `x`'s own dtype and widened after: a maximum over
    # representable values is itself representable, so this is the same
    # number an fp32 copy of the whole tensor would have given, without the
    # copy. The copy is not free on the cast path — it is 4 bytes an element
    # of traffic the kernel never moves, and `launch_trace.disagreement`
    # reads the launch's widest operand.
    if not group:
        amax = xg.abs().amax().float().clamp_min(eps).expand(
            xg.shape[0], xg.shape[1])
    else:
        amax = xg.abs().amax(dim=2).float().clamp_min(eps)
    scale = _encode_scale(amax / fmt.peak,
                          "pow2" if fmt.pow2_scale else encoding,
                          act_dtype, fmt)
    v = xg / scale.unsqueeze(2)

    if fmt.levels is not None:
        lv = torch.tensor(fmt.levels, device=v.device, dtype=v.dtype)
        idx = (v.abs().unsqueeze(-1) - lv).abs().argmin(-1)
        q = torch.sign(v) * lv[idx]
    elif fmt.torch_cast:
        q = v.to(getattr(torch, fmt.torch_cast))
    else:
        lo = -(fmt.peak + 1.0) if fmt.asymmetric else -fmt.peak
        q = v.round().clamp(lo, fmt.peak)

    return q, scale


def _round_once(w, fmt: WeightFormat, group: int, encoding: str, act_dtype):
    q, scale = quantise_groups(w, fmt, group, encoding, act_dtype)
    return (q.float() * scale.unsqueeze(2)).reshape(w.shape).to(w.dtype)


def round_to_grid(w, weight_dtype: str, group: int, *,
                  scale_encoding: str = "exact", act_dtype=None):
    """Place `w` on `weight_dtype`'s grid, at the kernel's group and scale.

    The returned tensor is a FIXED POINT: quantizing it again with the same
    (format, group) returns it unchanged, so a correct kernel's own
    quantization is lossless and the cross-check sees only its arithmetic.

    Unquantized dtypes come back untouched. An unrecognized one raises.
    """
    import torch
    if weight_dtype in UNQUANTIZED:
        return w
    fmt = FORMATS.get(weight_dtype)
    if fmt is None:
        raise UnknownWeightFormat(
            f"no grid declared for weight dtype {weight_dtype!r} "
            f"(known: {sorted(FORMATS) + sorted(UNQUANTIZED)})")

    if scale_encoding not in SCALE_ENCODINGS:
        raise ValueError(f"unknown scale encoding {scale_encoding!r} "
                         f"(known: {sorted(SCALE_ENCODINGS)})")
    if act_dtype is None:
        act_dtype = w.dtype

    cur = w
    for _ in range(_MAX_ROUNDS):
        nxt = _round_once(cur, fmt, group, scale_encoding, act_dtype)
        if torch.equal(nxt, cur):
            return cur
        cur = nxt
    raise RuntimeError(
        f"{weight_dtype} at group {group} did not reach a fixed point in "
        f"{_MAX_ROUNDS} rounds — returning it anyway would put the weight off "
        f"the grid the kernel derives, which is the lossy comparison this "
        f"exists to remove")


def effective_weights(inputs: Dict[str, Any], weight_dtype: str,
                      group: Optional[int], *, scale_encoding: str = "exact",
                      act_dtype=None) -> Optional[Dict[str, Any]]:
    """`inputs` with `w` placed on the kernel's grid, or None if unchanged.

    None means "use the shared inputs": either the weight is stored as-is, or
    the kernel declared no group and so made no claim about how it quantizes.
    """
    if weight_dtype in UNQUANTIZED or group is None or "w" not in inputs:
        return None
    out = dict(inputs)
    out["w"] = round_to_grid(inputs["w"], weight_dtype, int(group),
                             scale_encoding=scale_encoding, act_dtype=act_dtype)
    return out


# ---------------------------------------------------------------------------
# The activation side of the same grid: the cast a quantised matmul runs on
# its input operand, as a cell of its own (`quantise_activation`).
# ---------------------------------------------------------------------------

#: The eps the frameworks floor a group's amax at, so an all-zero group gets a
#: usable scale instead of a division by nothing. Theirs, not ours — vLLM
#: `model_executor/layers/quantization/utils/fp8_utils.py:632`
#: (`per_token_group_quant_fp8(..., eps: float = 1e-10, ...)`) and sglang
#: `kernels/ops/quantization/fp8_kernel.py:599`
#: (`sglang_per_token_group_quant_fp8(..., eps: float = 1e-10, ...)`), both
#: read at the release each framework's binding is audited at (v0.32.0 /
#: v0.5.22). The oracle has to floor where the kernels floor or the two
#: disagree on a degenerate group and the kernel is blamed for it.
ACTIVATION_EPS = 1e-10


def activation_scale_encoding(scale_dtype: str) -> str:
    """How a kernel holds a quantised activation's dynamic scale, read off
    the dtype the cell says it writes those scales in.

    Each of these dtypes IS an encoding, so this is a reading and not a
    choice:

      * `fp32` — the scale as computed, `amax / peak`: "exact".
      * `ue8m0` — eight exponent bits and no mantissa, so the only values it
        represents ARE powers of two; a kernel writing one has rounded the
        scale to one ("pow2"). vLLM `per_token_group_quant_fp8(use_ue8m0=
        True)`, sglang `sglang_per_token_group_quant_fp8(scale_ue8m0=True)`.
      * `fp8_e4m3` — four exponent bits reach 448, which is not enough range
        for an absolute scale, so a format that stores the block scale this
        narrow carries a per-tensor fp32 global factor beside it and encodes
        the ratio: NVFP4's two levels ("e4m3_global").

    Raises for anything else rather than defaulting: a scale dtype nobody
    has worked out the encoding of would otherwise be measured as "exact"
    and the kernel blamed for the difference.
    """
    name = str(scale_dtype).lower()
    if name == "fp32":
        return "exact"
    if name == "ue8m0":
        return "pow2"
    if name == "fp8_e4m3":
        return "e4m3_global"
    raise UnknownWeightFormat(
        f"no scale encoding worked out for a {scale_dtype!r} activation "
        f"scale; `kernel_bench.quant.activation_scale_encoding` says what "
        f"each scale dtype means")


def scale_group(inv, *, width: int) -> int:
    """How many elements of a `width`-wide row share one scale, for a cell
    that carries a quantised activation — the granularity the activation
    format states (`ActivationFormat.granularity`, on the cell as
    `activation_granularity`), read against that row.

    One reading for every launch that holds such a row: the cast, the
    gated-activation epilogue whose row is its `d_ffn` half, the dispatch
    that moves the record. A cell stating no granularity raises: per-token,
    per-128-group and per-32-group are three kernels.
    """
    granularity = inv.activation_granularity
    if not granularity:
        raise ValueError(
            f"the cell states no `activation_granularity`, so which "
            f"elements share a scale is unknown and per-token, "
            f"per-128-group and per-32-group are three kernels "
            f"(dtype_env {sorted(inv.dtype_env)})")
    return _group_of(granularity, int(width))


def gathers_a_record(signature: str) -> bool:
    """Whether a launch of `signature` writes its operand's elements
    unchanged in another order — so a quantised operand's RECORD (its
    elements and the scale each row carries) is moved, not computed.

    The signature says it: `OpSignature.gathers_from` names the input
    position whose activation the output is an index gather of, one of its
    elements per element written (`moe_permute`'s routed rows, `gather`'s
    per-token pick) — this side's reading of the arch op's own
    `OpSpec.gather_source`, held to it by
    `tests/backend/inference/test_gathering_signatures.py` because the
    harness may not look an op up by name.

    Asked of that declaration rather than of the cell's dtypes, because two
    slots naming one format say only that the element format survived the
    launch: the dispatch's own sibling `combine` reduces, an elementwise
    launch can read and write FP8 while computing, and a requantisation
    rewrites the scales. None of those is a move, and each stays a refusal
    until a provider declares what it computes.
    """
    from .signatures import get_signature
    sig = get_signature(signature)
    return sig is not None and sig.gathers_from is not None


@dataclass(frozen=True)
class ActivationCast:
    """What one `quantise_activation` cell casts between, read off the cell.

    The direction is not a flag anyone sets: the cell states the dtype the
    launch READS (`activation`) and the dtype it WRITES (`output`), and one
    of the two is a quantised format while the other is not. So quantise and
    dequantise are two cells of one signature rather than two signatures,
    and `on_grid` is the same oracle for both — dequantise(quantise(x)) is
    "x as the narrow format holds it" whichever end you start from.
    """
    #: Corpus token of the quantised end (`fp8_e4m3`, `fp4_e2m1`, …).
    narrow: str
    #: Corpus token of the end that holds values as they are (`bf16`).
    wide: str
    #: Elements along the row that share one scale; 0 is one for the tensor.
    group: int
    #: How the scale is held (`activation_scale_encoding`).
    scale_encoding: str
    #: True when the launch reads wide and writes narrow.
    narrows: bool

    @classmethod
    def of(cls, inv, *, width: Optional[int] = None) -> "ActivationCast":
        """Read the cast off an invocation of `quantise_activation`.

        Every field comes from a declaration: the two dtypes from the cell's
        own slots, the group from the granularity the activation format
        states (`ActivationFormat.granularity`, carried in the cell as
        `activation_granularity`), the encoding from the scale dtype. The
        row width defaults to `d_model`, the `quantise_activation` axis; a
        fused span whose quantised output has another last extent passes that
        graph extent as `width` rather than naming a second default. A cell
        missing any of them raises — guessing one measures a kernel the row
        does not describe.

        A launch that MOVES its operand's record rather than casting to one
        is `moved_by`, and which launches those are is the op's own
        declaration, not this cell's dtypes.
        """
        read, written = inv.dtype_for("activation"), inv.dtype_for("output")
        narrow = [d for d in (read, written) if d not in UNQUANTIZED]
        wide = [d for d in (read, written) if d in UNQUANTIZED]
        if len(narrow) != 1 or len(wide) != 1:
            raise ValueError(
                f"a quantised-activation cell casts between one quantised "
                f"format and one that is not; this one reads {read!r} and "
                f"writes {written!r}")
        row_width = (int(inv.dims["d_model"]) if width is None
                     else int(width))
        return cls(narrow=narrow[0], wide=wide[0],
                   group=scale_group(inv, width=row_width),
                   scale_encoding=activation_scale_encoding(
                       inv.dtype_for("metadata")),
                   narrows=written == narrow[0])

    @classmethod
    def moved_by(cls, inv, *, width: Optional[int] = None) -> "ActivationCast":
        """The record a launch of this cell MOVES — its elements and the
        scale of each row, written unchanged in another order.

        Which launches move a record is the OP's declaration
        (`gathers_a_record`), never the cell's two dtype slots agreeing:
        that agreement says only that the element FORMAT is preserved, and
        a launch can preserve it while computing — the MoE dispatch's own
        sibling `combine` reads an FP8 record and writes one through a
        weighted sum, and a requantisation rewrites the scales. So a cell
        of a signature that computes its output raises here instead of
        being measured as a move.

        The record is read off the cell the same way a cast's is (the
        format, the granularity's group, the scale encoding); both slots
        name it, and `wide` is the fp32 the record decodes to, so `narrows`
        is False — nothing is narrowed.
        """
        if not gathers_a_record(inv.signature):
            raise ValueError(
                f"{inv.signature!r} computes its output, so a quantised "
                f"operand's record is not moved through it: no op it binds "
                f"declares `gather_source` (the input port whose activation "
                f"the output is an index gather of)")
        read, written = inv.dtype_for("activation"), inv.dtype_for("output")
        if read != written or read in UNQUANTIZED:
            raise ValueError(
                f"a moved record is one record: this cell reads {read!r} and "
                f"writes {written!r}")
        row_width = (int(inv.dims["d_model"]) if width is None
                     else int(width))
        return cls(narrow=read, wide="fp32",
                   group=scale_group(inv, width=row_width),
                   scale_encoding=activation_scale_encoding(
                       inv.dtype_for("metadata")),
                   narrows=False)

    @property
    def format(self) -> WeightFormat:
        """The grid the narrow end is on."""
        fmt = FORMATS.get(self.narrow)
        if fmt is None:
            raise UnknownWeightFormat(
                f"no grid declared for {self.narrow!r} "
                f"(known: {sorted(FORMATS)})")
        return fmt

    def _torch_wide(self):
        from .refs import torch_dtype
        return torch_dtype(self.wide)

    def quantise(self, x):
        """`x` → `(elements, scales)`, the pair a cast kernel writes."""
        return quantise_groups(x, self.format, self.group,
                               self.scale_encoding, self._torch_wide(),
                               eps=ACTIVATION_EPS)

    def dequantise(self, pair):
        """`(elements, scales)` → the wide tensor, fp32, flat rows.

        fp32 rather than the cell's wide dtype: this is what the cross-check
        compares, and the comparison is in fp32 anyway. Rounding to bf16
        first would put a whole bf16 ulp between a kernel and an oracle that
        computed the same product.
        """
        q, scale = pair[0], pair[1]
        rows = q.reshape(scale.shape[0], scale.shape[1], -1)
        return (rows.float() * scale.unsqueeze(2)).reshape(-1, rows.shape[1]
                                                           * rows.shape[2])

    def on_grid(self, x):
        """`x` as the narrow format holds it — the oracle, in fp32.

        The kernel's own pair, multiplied back. Not a second formula: a cast
        measured against one would be measured against somebody's opinion of
        what it should compute.
        """
        return self.dequantise(self.quantise(x))


def _group_of(granularity: str, width: int) -> int:
    """Elements that share one scale, from the granularity string the
    activation format states and the row width the cell carries.

    `per-token` is one scale for the whole row, which is this row's width —
    not 0. (0 means one scale for the whole TENSOR, which is a different
    kernel: a static per-tensor scale, and no activation format declares
    one today.)
    """
    if granularity == "per-token":
        return int(width)
    if granularity.startswith("per-group:"):
        return int(granularity[len("per-group:"):])
    raise ValueError(
        f"unknown activation granularity {granularity!r}; "
        f"`arch/activation_formats/*.yaml` declares what there is")
