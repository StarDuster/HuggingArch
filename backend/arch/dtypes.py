"""How wide each dtype is, in one place.

There were five of these tables — `materialize`, `eval.formats`, `hf.metadata`,
`inference.deployment_dtype`, `rendering._consts` — holding 22, 14, 13, 7 and 7
entries, and after they were folded two more grew back: the benchmark
identity's scale-dtype widths and the kernel_bench planner's, each with its own
miss default. Those now query this table too, and the benchmark CSV vocabulary
(`inference.benchmarks.CORPUS_TOKENS`) is checked at import against it, so a
token the corpus accepts is a width capacity can price. They never contradicted each other, which is the part that made the
duplication survive: the failure mode is not a wrong number in one table, it is
a MISSING number in one table while another has it.

Four of the five answered a miss with a default. `materialize` returned 2.0 and
said so in its docstring ("a soft fallback for unrecognized author input");
`hf.metadata` returned 2 in one place and 4 in another, on the HEADER side of
the byte reconciliation, which is the side that is supposed to be ground truth.
So an unknown dtype did not fail — it silently became BF16, and a checkpoint's
scales came out at twice their width against a header total computed the same
wrong way.

`eval.formats` was the exception and its comment records why: `F8_E8M0FNU` had
no entry, an unknown dtype cost zero, and 49 GB of V4-Pro went missing. Its
answer was to return `None` so the caller could refuse. This module makes that
the only answer.

The same dtype cost the other direction on the header side, and the size of it
is the argument: DeepSeek-V4-Pro's 49,150,268,416 MXFP4 block scales were
counted at the metadata default of four bytes each, so
`quantization_metadata_bytes` read 196,601,073,664 against a file holding
49,150,268,416, and `checkpoint_storage_bytes` — the two summed, not a size
read from anywhere — carried the error into the comparison.

Spellings vary by source and both are kept as data rather than normalised at
every call site: a safetensors header writes `F8_E8M0`, torch and the scheme
yamls write `F8_E8M0FNU`, and config JSON writes `float8_e4m3fn`.
"""
from __future__ import annotations


from typing import Any, Dict, FrozenSet, Mapping, NamedTuple, Optional, Tuple


class UnknownDtype(KeyError, ValueError):
    """Raised instead of guessing a width.

    A dtype this table does not know is a dtype nobody has checked the width
    of, and the two ways to proceed are to fail here or to be wrong later by a
    factor the size of the guess. It is a `KeyError` for the callers that
    look a name up and a `ValueError` for the API boundary, which turns a
    bad input into a 4xx and would turn a bare `KeyError` into a 500.
    """


class _Dtype(NamedTuple):
    """One dtype. `width` is bytes per logical element; `peak` is the GPU
    CSV column family, or None when this dtype is not an MMA (scales,
    indices, booleans); `token` is the spelling the benchmark corpus and the
    calibration overlay use, or None when the corpus has no such row;
    `torch` is the torch / config.json spelling; `display` is the label a
    page shows; `quantized` says whether, as a WEIGHT dtype, this is a
    quantized format (None for dtypes that are never a weight's logical
    dtype — scales, indices, bool); `spellings` are every way a source
    writes it (safetensors header, scheme yaml, UI knob, torch)."""
    width: float
    peak: Optional[str] = None
    token: Optional[str] = None
    torch: Optional[str] = None
    display: Optional[str] = None
    quantized: Optional[bool] = None
    spellings: Tuple[str, ...] = ()


#: One row per dtype, keyed by its canonical spelling. This table replaced
#: nine: the width table it already was, plus the torch-name map in
#: `eval.formats`, the UI-alias map and the corpus vocabulary in
#: `inference.benchmark_identity` / `benchmarks`, the display labels in
#: `inference.deployment_dtype`, the reverse torch map in `rendering._consts`,
#: the safetensors-spelling map in `kernel_bench.plan`, the unquantized set in
#: `kernel_bench.quant`, and the exponent-scale alias in
#: `validate._packing_consistency`. Each was a column of this one.
#:
#: Two entries encode a serving convention rather than a spelling: `FP8` and
#: `F8` mean `F8_E4M3` (vLLM's default for an unqualified fp8), which is what
#: every consumer of the old alias tables already assumed.
DTYPES: Mapping[str, _Dtype] = {
    "F64": _Dtype(8.0, None, None, "float64", "FP64", False, ("F64", "FP64", "FLOAT64", "DOUBLE")),
    "F32": _Dtype(4.0, "fp32", "fp32", "float32", "FP32", False, ("F32", "FP32", "FLOAT32", "FLOAT")),
    "F16": _Dtype(2.0, "fp16", "fp16", "float16", "FP16", False, ("F16", "FP16", "FLOAT16", "HALF")),
    "BF16": _Dtype(2.0, "bf16", "bf16", "bfloat16", "BF16", False, ("BF16", "BFLOAT16")),
    "F8_E4M3": _Dtype(1.0, "fp8", "fp8_e4m3", "float8_e4m3fn", "FP8 (E4M3)", True,
                      ("F8_E4M3", "FP8_E4M3", "FLOAT8_E4M3FN", "FLOAT8_E4M3", "FP8", "F8")),
    "F8_E5M2": _Dtype(1.0, "fp8", "fp8_e5m2", "float8_e5m2", "FP8 (E5M2)", True,
                      ("F8_E5M2", "FP8_E5M2", "FLOAT8_E5M2")),
    "F8_E8M0": _Dtype(1.0, None, "ue8m0", "float8_e8m0fnu", "UE8M0", None,
                      ("F8_E8M0", "F8_E8M0FNU", "UE8M0", "FLOAT8_E8M0FNU")),
    "MXFP8": _Dtype(1.0, "fp8", "mxfp8", None, "MXFP8", True, ("MXFP8",)),
    "FP4_E2M1": _Dtype(0.5, "fp4", "fp4_e2m1", None, "FP4", True,
                       ("FP4_E2M1", "FP4", "NVFP4", "FLOAT4_E2M1")),
    "MXFP4": _Dtype(0.5, "fp4", "mxfp4", None, "MXFP4", True, ("MXFP4",)),
    "NF4": _Dtype(0.5, "fp4", None, None, "NF4", True, ("NF4",)),
    "INT4": _Dtype(0.5, "fp4", "int4", None, "INT4", True, ("INT4", "I4")),
    "I8": _Dtype(1.0, "int8", "int8", "int8", "INT8", True, ("I8", "INT8")),
    "U8": _Dtype(1.0, "int8", None, "uint8", "U8", None, ("U8", "UINT8")),
    "I16": _Dtype(2.0, None, None, "int16", "I16", None, ("I16", "INT16")),
    "U16": _Dtype(2.0, None, None, "uint16", "U16", None, ("U16", "UINT16")),
    # A token: int32 is a dtype kernels are measured at — the index tensors
    # a top-k selection writes (`dsa_topk_mask`'s output, vLLM / sglang
    # allocate them int32) are keyed by it, as every other written tensor is.
    "I32": _Dtype(4.0, "fp32", "int32", "int32", "I32", None, ("I32", "INT32")),
    "U32": _Dtype(4.0, None, None, "uint32", "U32", None, ("U32", "UINT32")),
    "I64": _Dtype(8.0, None, None, "int64", "I64", None, ("I64", "INT64")),
    "U64": _Dtype(8.0, None, None, "uint64", "U64", None, ("U64", "UINT64")),
    "BOOL": _Dtype(1.0, None, None, "bool", "BOOL", None, ("BOOL",)),
}

#: Spelling → row, derived. Every spelling belongs to exactly one row, and
#: the canonical name is one of its own spellings; both are checked below.
_DTYPES: Dict[str, _Dtype] = {}
_CANONICAL: Dict[str, str] = {}
for _name, _row in DTYPES.items():
    if _name not in _row.spellings:
        raise ValueError(f"dtypes: canonical {_name!r} is not among its own spellings")
    for _sp in _row.spellings:
        if _sp in _DTYPES:
            raise ValueError(f"dtypes: spelling {_sp!r} belongs to both {_CANONICAL[_sp]!r} and {_name!r}")
        _DTYPES[_sp] = _row
        _CANONICAL[_sp] = _name
    if _row.torch is not None and _row.torch.upper() not in _DTYPES:
        _DTYPES[_row.torch.upper()] = _row
        _CANONICAL[_row.torch.upper()] = _name
_tokens = [r.token for r in DTYPES.values() if r.token]
if len(_tokens) != len(set(_tokens)):
    raise ValueError("dtypes: a corpus token belongs to two rows")
del _name, _row, _sp, _tokens


def _row_of(dtype: str) -> _Dtype:
    try:
        return _DTYPES[str(dtype).strip().upper()]
    except KeyError:
        raise UnknownDtype(
            f"unknown dtype spelling {dtype!r} — add it to a row of "
            f"`backend.arch.dtypes.DTYPES`") from None


def canonical(dtype: str) -> str:
    """The canonical spelling for any spelling (`F8_E8M0FNU` → `F8_E8M0`,
    `float16` → `F16`), or `UnknownDtype`."""
    _row_of(dtype)
    return _CANONICAL[str(dtype).strip().upper()]


def corpus_token(dtype: str) -> str:
    """The benchmark-corpus / calibration spelling of a dtype (`fp8_e4m3`),
    or `UnknownDtype` when the spelling is unknown or the dtype has no row
    in the corpus vocabulary."""
    row = _row_of(dtype)
    if row.token is None:
        raise UnknownDtype(
            f"dtype {dtype!r} has no benchmark-corpus token; it is not a "
            f"dtype a kernel is measured at")
    return row.token


def has_corpus_token(dtype: str) -> bool:
    """Whether the spelling is known and its dtype has a corpus token —
    the predicate behind an API knob's vocabulary check."""
    row = _DTYPES.get(str(dtype).strip().upper())
    return row is not None and row.token is not None


def display_name(dtype: str) -> str:
    """The label a page shows for a dtype (`FP8 (E4M3)`)."""
    row = _row_of(dtype)
    return row.display or canonical(dtype)


def display_label(dtype: str) -> str:
    """`display_name` for a spelling the table knows; any other format keeps
    its own name. For a label, where an unlisted format is still shown."""
    row = _DTYPES.get(str(dtype).strip().upper())
    if row is None:
        return str(dtype)
    return row.display or _CANONICAL[str(dtype).strip().upper()]


def torch_name(dtype: str) -> Optional[str]:
    """The torch / config.json spelling, or None when torch has none
    (packed formats such as MXFP4 are not torch dtypes)."""
    return _row_of(dtype).torch


def from_torch_name(name: str) -> Optional[str]:
    """Canonical spelling for a torch / config.json dtype name, or None when
    no row lists that spelling (the caller says what an unknown name means
    in its context)."""
    key = str(name).strip().upper()
    return _CANONICAL.get(key)


#: The benchmark corpus vocabulary: every token is a row of the table.
CORPUS_TOKENS: FrozenSet[str] = frozenset(r.token for r in DTYPES.values() if r.token)
#: Weight dtypes that store values as-is (nothing to place on a grid).
UNQUANTIZED_TOKENS: FrozenSet[str] = frozenset(
    r.token for r in DTYPES.values() if r.token and r.quantized is False)


def is_unquantized(dtype: str) -> bool:
    """Whether `dtype` stores values as-is (F64 / F32 / F16 / BF16): the
    dtypes type promotion ranks by width. False for a quantised format and
    for a dtype that is never a weight's logical dtype (scales, indices);
    `UnknownDtype` for a spelling the table does not know."""
    try:
        return _DTYPES[str(dtype).upper()].quantized is False
    except KeyError:
        raise UnknownDtype(f"unknown dtype {dtype!r}") from None


def accepted_dtype_names() -> FrozenSet[str]:
    """Every lower-cased spelling that resolves to a corpus token — the
    vocabulary an API knob accepts."""
    return frozenset(sp.lower() for name, row in DTYPES.items()
                     if row.token for sp in row.spellings)


#: Width column, for callers that already index a name to a float.
#: Derived from `_DTYPES`, not a second vocabulary.
DTYPE_BYTES: Mapping[str, float] = {
    name: row.width for name, row in _DTYPES.items()
}


def dtype_bytes(dtype: str) -> float:
    """Bytes per logical element, or `UnknownDtype`.

    Case-insensitive on the spelling, because the same dtype arrives as
    `bf16` from a config and `BF16` from a header.
    """
    try:
        return _DTYPES[str(dtype).upper()].width
    except KeyError:
        raise UnknownDtype(
            f"no byte width known for dtype {dtype!r} — add it to "
            f"`backend.arch.dtypes._DTYPES`. Guessing a width is "
            f"how {'F8_E8M0FNU'!r} came to cost nothing at all."
        ) from None


def peak_kind(dtype: str) -> str:
    """`fp4` / `fp8` / `int8` / `bf16` / `fp16` / `fp32`, or `UnknownDtype`.

    A dtype with a width and no `peak` (block scales, indices) is not
    an MMA family — the missing column is declared on the same row,
    not guessed from the width.
    """
    try:
        row = _DTYPES[str(dtype).upper()]
    except KeyError:
        raise UnknownDtype(
            f"no GPU peak column for dtype {dtype!r} — add it to "
            f"`backend.arch.dtypes._DTYPES`"
        ) from None
    if row.peak is None:
        raise UnknownDtype(
            f"no GPU peak column for dtype {dtype!r} — it has a width "
            f"but is not an MMA family"
        )
    return row.peak


def dtype_bits(dtype: str) -> float:
    """Bits per logical element. Same table, for the callers that display
    widths rather than multiply by them."""
    return dtype_bytes(dtype) * 8


def native_dtype(config: Any) -> Optional[str]:
    """The precision a checkpoint stores anything no scheme covers at.

    `config.torch_dtype` — `dtype` in newer transformers — and it is a
    DECLARATION, not a guess: gemma-2 says `float32` and ships F32 weights,
    Llama-2 says `float16` and ships F16. Defaulting to BF16 instead put both of
    those out by exactly a factor of two against their own headers, which is the
    kind of error that a reconciliation finds and a plausible default hides.

    Note what it is NOT. For a quantised checkpoint this names the precision of
    the tensors the scheme leaves alone — MiMo and DeepSeek-V4-Pro both say
    `bfloat16` while their bulk weights are FP8 and INT8. So it is the DEFAULT
    dtype in the precise sense `FormatPolicy` means, and never the answer for a
    tensor a role rule covers.
    """
    if config is None:
        return None
    for key in ("torch_dtype", "dtype"):
        raw = (config.get(key) if isinstance(config, Mapping)
               else getattr(config, key, None))
        if raw:
            name = from_torch_name(raw)
            if name is None:
                # The key is PRESENT and names a dtype the table does not
                # know. Answering None here let the caller's `native or
                # "BF16"` price a float8 checkpoint at two bytes.
                raise UnknownDtype(
                    f"config.{key}={raw!r} is not a dtype spelling in "
                    f"`backend.arch.dtypes.DTYPES`")
            return name
    return None


def served_dtype(shipped: str) -> str:
    """The width a default deployment RUNS a checkpoint's own tensors at.

    `shipped` is the width the checkpoint stores them at
    (`stated_default_dtype`). The rule is one monotone sentence and names
    no framework:

        **Nothing ships running wider than 16 bits. Anything already 16
        bits is left exactly as it is stored, and nothing is ever widened.**

    So: F64 / F32 → BF16 (narrowed, because no deployment runs 32-bit
    weights), F16 → F16 and BF16 → BF16 (untouched). It is a property of the
    MODEL, not of a tensor: a checkpoint has one such width and every
    unquantised parameter is held at it.

    Why the rule is stated this way rather than as "what the frameworks
    load it as". The repo already holds the companion rule that a BF16
    checkpoint stays BF16 even though vLLM materialises fp32 parameters for
    some of it at load — we model the width the weight IS, not every cast a
    loader happens to perform. "Follow the framework's loader" would
    contradict that; "never run wider than 16 bits, never widen" agrees
    with it, and the two cases stop being two policies: one monotone rule
    that happens to coincide with the checkpoint on a 16-bit model and with
    the loader on a 32-bit one. A reader can answer the next case from the
    sentence alone, which is the test a rule has to pass — the alternative
    ("the corpus's only fp32 models are gemma-2, and both frameworks agree
    there") is a proxy condition of exactly the shape §1.3 refuses.

    Running wider IS expressible, just not as a factory width: it is a
    CHOSEN precision, and the one entrance for choosing one is the
    precision profile (`arch/precision_profile.py`, request field
    `precision`). This function answers what a model ships as.

    CORROBORATION, not the premise — both loaders read at vLLM v0.32.0
    (`5df63a7`) and sglang v0.5.22 (`baa9bef`), and both do narrow to 16
    bits:

      - vLLM `_resolve_auto_dtype` (`vllm/config/model.py:2405`), reached
        whenever `--dtype` is `auto`, its default (`:2464-2471`), replaces a
        float32 config dtype with `supported_dtypes[0]` (`:2422-2424`), and
        on CUDA SM80+ that list is `[bfloat16, float16, float32]`
        (`vllm/platforms/cuda.py:255-258`). An already-supported config
        dtype comes back unchanged (`:2426-2427`).
      - sglang `_get_and_verify_dtype`
        (`python/sglang/srt/configs/model_config.py:2172`) narrows a float32
        config dtype to bfloat16 for the gemma family (`:2193`, `:2203`) and
        to float16 otherwise (`:2207`); any other config dtype is kept
        (`:2209`).

    The two therefore pick DIFFERENT 16-bit floats for a non-gemma float32
    checkpoint. Under this rule that is two names for one width, not two
    widths: both are 2 bytes (the table above) and
    `inference/gpu_specs.csv` prices `fp16_tflops == bf16_tflops` on every
    row (held by `test_gpu_specs`), so no published number can tell them
    apart — the same razor the fusion conditions use. One of the two names
    has to be written down and BF16 is written down. What a panel LABELS it
    is the open end: a single label is wrong for one of the two frameworks,
    and rendering per framework would mean a framework dimension on the
    walk, which is a dimension and not an `if`.

    `shipped` that is not a plain float raises `UnknownDtype`. Not because
    the rule has nothing to say about a narrow width — it says "leave it
    alone" — but because such a value is not a width a model's own tensors
    come in: `quant_context.default_dtype` names what the checkpoint stores
    its UNQUANTISED tensors at, so an FP8 or integer there is a broken
    declaration, and §2.4 is explicit that a missing weight dtype raises
    rather than defaulting to BF16.
    """
    row = _row_of(shipped)
    # A plain float, read off the column that already says it: a dtype that
    # holds its values as-is rather than on a grid (`quantized is False`:
    # F64 / F32 / F16 / BF16). The ceiling, then: two bytes is at it and
    # stays exactly as stored, anything wider comes down to it. Never the
    # other way — this expression cannot return something wider than it was
    # given, which is what keeps it consistent with a BF16 checkpoint
    # staying BF16 through a loader that materialises fp32 parameters.
    if row.quantized is False:
        return canonical(shipped) if row.width == 2.0 else "BF16"
    # Not a plain float, so not a width a model's own tensors come in. The
    # ceiling rule would happily leave a narrow width alone; the reason to
    # refuse is that the QUESTION is wrong here —
    # `quant_context.default_dtype` names what the checkpoint stores its
    # unquantised tensors at, and a scheme's width arriving in it is a
    # broken declaration. Answering BF16 would be the plausible default
    # §2.4 forbids: the move that made `F8_E8M0FNU` cost nothing and lost
    # 49 GB of V4-Pro.
    raise UnknownDtype(
        f"a model's own tensors cannot be stored at {canonical(shipped)}: "
        f"`served_dtype` narrows a checkpoint's plain float width to the "
        f"16-bit ceiling nothing runs above, and {canonical(shipped)} is "
        f"not a plain float. Correct the spec's "
        f"`quant_context.default_dtype` — it names the dtype the checkpoint "
        f"stores its UNQUANTISED tensors at, never a scheme's width — or, "
        f"to RUN a width rather than ship one, state it in a precision "
        f"profile (`arch/precision_profile.py`), the one entrance for a "
        f"chosen precision")


def stated_default_dtype(quant_context: Any, config: Any,
                         language: Tuple[str, Mapping],
                         checkpoint: Optional[Mapping[str, int]] = None,
                         ) -> Tuple[Optional[str], Optional[str]]:
    """`(dtype, source)` of the precision the model ships the tensors no
    quant scheme covers at — its activations, the cache slots that declare
    no format, the weights a scheme leaves alone.

    Read off the checkpoint, never off `config.torch_dtype`: a quantised
    repo labels that field however its exporter liked. In order:

      1. the spec's `quant_context.default_dtype`, a declaration — it must
         be one of the float dtypes the header stores those tensors at;
      2. `checkpoint` — `{dtype: elements}` of the unquantised float
         tensors the header stores (`CheckpointHeader.
         unquantised_float_dtypes`): one dtype is the answer; several (BF16
         bulk beside F32 norms) are refused until the spec declares which
         is the model's (`quant_context.default_dtype`);
      3. with no header to read at all (a resolution without a
         checkpoint), the config's own statement — `language`'s fields
         first (`text_config.dtype` on Qwen3-VL), then the root's.

    The config's dtype otherwise only cross-checks the header: a
    disagreement is named in `source`, and decides nothing. `(None, None)`
    when nothing states one."""
    floats = dict(checkpoint or {})
    declared = (quant_context.declared_default_dtype
                if quant_context is not None else None)
    if declared:
        if floats and canonical(declared) not in floats:
            raise ValueError(
                f"quant_context.default_dtype is {declared}, and the "
                f"checkpoint stores no unquantised tensor at it (it stores "
                f"them at {sorted(floats)}). Declare the dtype the header "
                f"stores the model's unquantised tensors at.")
        return declared, "quant_context"
    scope, fields = language
    stated = (native_dtype(fields) if scope else None) or native_dtype(config)
    if floats:
        if len(floats) > 1:
            raise ValueError(
                f"the checkpoint stores its unquantised tensors at "
                f"{ {d: floats[d] for d in sorted(floats)} } (elements per "
                f"tensor pattern), so it states no single shipped dtype. "
                f"Declare `quant_context.default_dtype` in the spec: the "
                f"dtype of the model's bulk unquantised weights.")
        (dtype,) = floats
        source = "checkpoint"
        if stated is not None and canonical(stated) != dtype:
            source += f" (config states {stated}, which decides nothing)"
        return dtype, source
    if scope:
        native = native_dtype(fields)
        if native:
            return native, f"config.{scope} (no checkpoint header)"
    native = native_dtype(config)
    if native:
        return native, "config (no checkpoint header)"
    return None, None
