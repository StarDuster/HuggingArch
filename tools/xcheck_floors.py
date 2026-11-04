"""What the cross-check's bands are allowed to be: each dtype's own floor.

A band in `kernel_bench.driver._XCHECK_BANDS` says how far a kernel may sit
from the fp32 oracle before it is computing the wrong function rather than
the right one in low precision. That number is only honest if it is read off
the dtype itself, which is what this measures.

HOW the floor is measured is the whole of it, and the obvious way is wrong.
Rounding a cell's operands to the dtype and running the fp32 oracle on them
measures the INPUT quantisation — an error the gate never sees, because the
harness builds its inputs at the cell's dtype already and the oracle eats
those same inputs upcast. Floors measured that way came out 5x to 40x too
high (a bf16 GQA cell read 0.08 where a correct kernel sits at 0.003), and
a band set above them is a band that measures nothing.

So the floor is the same arithmetic performed AT THE CELL'S PRECISION:
every op's result written in the dtype, each op accumulating internally in
fp32 — which is what a fused kernel does, and the only thing left between a
correct implementation and the oracle. Nothing is reimplemented per
signature: the signature's own reference runs under a dispatch mode that
narrows each result as it is produced (`_InDtype`).

    python -m tools.xcheck_floors                    # every dtype, default cells
    python -m tools.xcheck_floors --signature gemm   # one signature
    python -m tools.xcheck_floors --jobs <plan.jsonl>  # real cells from a plan

Each row is one (signature, dtype): the floor's RMS and its worst element
over the cells sampled, and what margin the current bands leave above them.
A band under its own floor cannot be passed arithmetically; one far above it
is not measuring the dtype.

KNOWN VACUOUS ROW. `nvfp4` is in the dtype list for a reason the comment
there states — the fp8 band's job is to sit below it — but no grid is
declared under that name (`kernel_bench.quant.FORMATS` has `fp4_e2m1` and
`mxfp4`; NVFP4 is E2M1 elements with an e4m3 block scale per 16 channels
and one fp32 global factor, which is the `nvfp4_dynamic` ACTIVATION
format). So nothing quantises for that row and its reported floor is zero,
which is the oracle compared with itself. It has read zero since the
ceiling table it used to consult had two entries; giving it a real floor
means naming the format rather than the dtype.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

#: The cells measured when no plan is given: one shape per signature, in the
#: middle of what the corpus holds, with the axes each signature declares.
#: Deliberately small enough to run beside anything else on the card.
DEFAULT_CELLS: Tuple[Tuple[str, str, Dict[str, int]], ...] = (
    ("gemm", "linear", {"B": 1, "s_query": 4096, "d_reduce": 4096, "d_out": 4096}),
    ("gemm", "linear", {"B": 1, "s_query": 512, "d_reduce": 16384, "d_out": 2048}),
    ("rope", "fused_qk_rope",
     {"B": 4, "s_query": 4096, "n_heads": 32, "n_kv_heads": 8, "d_rope": 64}),
    ("rmsnorm", "rmsnorm", {"B": 4, "s_query": 4096, "d_model": 8192}),
    ("silu_and_mul", "silu_and_mul", {"B": 4, "s_query": 4096, "d_ffn": 8192}),
    ("gqa_attention", "gqa_attention",
     {"B": 4, "n_heads": 32, "n_kv_heads": 8, "s_query": 1024, "s_kv": 1024,
      "d_head": 128}),
    ("mla_attention", "mla_attention",
     {"B": 4, "n_heads": 16, "s_query": 1, "s_kv": 4096, "d_head": 576,
      "d_v_head": 512, "kv_block_size": 64}),
)

#: The dtypes a band exists for, plus the one that must NOT pass as another:
#: nvfp4 is four-bit elements under an e4m3 block scale, so a kernel running
#: it where the cell declares fp8 is a real failure mode and the fp8 band's
#: job is to be below it.
DTYPES: Tuple[str, ...] = ("fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2",
                           "nvfp4")

_FP8_MAX = {"fp8_e4m3": 448.0, "fp8_e5m2": 57344.0}


def _torch_dtype(name: str):
    """`name` as torch spells it, through the harness's own mapping."""
    from kernel_bench.refs import torch_dtype
    return torch_dtype(str(name))


class _InDtype:
    """A dispatch mode that writes every floating result in `dtype`.

    The arithmetic a narrow kernel performs: each op accumulates internally
    at full width and writes its result narrow, which is what the next op
    then reads. Composed over a signature's reference, that is the best any
    implementation of it can do at this precision — the floor a band has to
    clear. Integer and boolean results pass through: an index is not a
    precision.

    It does not describe every signature, and `_MODEL_LIMIT` is how a row
    says so rather than reporting a number: a chunked recurrent scan carries
    its state and its log-space decay in fp32 in every implementation (fla's
    `chunk_kda` asserts on anything else), so narrowing those models an
    implementation nobody runs — RMS 12 where the kernels sit near 0.002.
    """

    def __init__(self, dtype: str) -> None:
        self.torch_dtype = _torch_dtype(dtype)
        self._mode = None

    def __enter__(self) -> "_InDtype":
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        narrow = self.torch_dtype

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))

                def _cast(value):
                    if (torch.is_tensor(value) and value.is_floating_point()
                            and value.dtype is not narrow):
                        return value.to(narrow).to(value.dtype)
                    return value

                if isinstance(out, (tuple, list)):
                    return type(out)(_cast(v) for v in out)
                return _cast(out)

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exception) -> None:
        if self._mode is not None:
            self._mode.__exit__(*exception)


#: Past this many times the dtype's own relative epsilon, a "floor" is not
#: one: no correct implementation of anything sits 100 rounding steps away
#: from the oracle, so the number is measuring the MODEL rather than the
#: arithmetic (the chunked scans, whose state the model narrows and the
#: implementations do not). Derived from the dtype, not listed per signature.
_MODEL_LIMIT = 100.0

_EPSILON = {"fp32": 2 ** -24, "tf32": 2 ** -11, "fp16": 2 ** -11,
            "bf16": 2 ** -8, "fp8_e4m3": 2 ** -4, "fp8_e5m2": 2 ** -3,
            "nvfp4": 2 ** -2}


def _quantised(inputs: Dict[str, Any], dtype: str) -> Dict[str, Any]:
    """`inputs` with every float operand through `dtype` and back, at the
    per-row scale a narrow kernel computes.

    The other situation, and the same question: where the canonical inputs
    are WIDER than the cell's dtype — every fp8 and fp4 cell, whose operands
    the provider casts — what the kernel does that the oracle does not is
    quantise them, so that is the floor.

    Through `kernel_bench.quant.round_to_grid`, the harness's one statement
    of what each format's grid is: the amax, the scale and the cast used to
    be written out again here, and the same arithmetic in three places
    (this, the GEMM builds' cast helper, and the grid) is three answers
    waiting to disagree about one quantisation. Per-row grouping — a scale
    for each row of the operand, which is what `amax(dim=-1)` was — is the
    `group = row width` case of that grid.

    A dtype the grid table does not describe (`nvfp4`, which names an
    activation FORMAT rather than an element dtype) passes through
    unquantised, exactly as it did when the ceiling table had no entry for
    it. Its row therefore reports a floor of zero, which is not one; see
    the note in the module docstring.
    """
    import torch
    from kernel_bench.quant import FORMATS, round_to_grid
    if dtype not in FORMATS:
        return inputs
    out = {}
    for name, value in inputs.items():
        if (torch.is_tensor(value) and value.is_floating_point()
                and value.numel() > 1):
            out[name] = round_to_grid(value.float(), dtype,
                                      int(value.shape[-1])).to(value.dtype)
        else:
            out[name] = value
    return out


def _statistics(out: Any, ref: Any) -> Tuple[float, float, float, int]:
    """The gate's own two numbers, plus the old one and the element count."""
    scale = ref.abs().mean().clamp_min(1e-6)
    error = (out - ref).abs().div(ref.abs() + scale)
    return (error.pow(2).mean().sqrt().item(), error.max().item(),
            ((out - ref).abs().max() / scale).item(),
            error.numel())


def _cells(args: argparse.Namespace) -> Iterable[Tuple[str, str, Dict[str, int]]]:
    if not args.jobs:
        for signature, op, dims in DEFAULT_CELLS:
            if not args.signature or signature == args.signature:
                yield signature, op, dims
        return
    seen = set()
    with open(args.jobs) as handle:
        lines = handle.readlines()
    for line in lines:
        job = json.loads(line)
        signature = job.get("signature") or ""
        if args.signature and signature != args.signature:
            continue
        key = (signature, json.dumps(job.get("dims"), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        yield signature, job.get("op") or signature, job["dims"]


def _floor(signature: str, op: str, dims: Dict[str, int], dtype: str,
           device: str) -> Optional[Tuple[float, float, float, int]]:
    """One (cell, dtype) floor, or None when the signature has no oracle.

    The cell's inputs are built as the harness builds them for a
    measurement, and the fp32 oracle runs on exactly those — so no input
    quantisation enters where the harness has none, just as it does not in
    the gate. What the kernel then does that the oracle does not is either
    narrow its intermediates (a dtype the inputs are already in) or quantise
    its operands (a dtype the provider casts to), and the floor is that.
    """
    import torch
    from kernel_bench.invocation import Invocation
    from kernel_bench.refs import get_ref
    spec = get_ref(signature)
    if spec is None or spec.reference is None:
        return None
    buildable = dtype in ("fp32", "tf32", "fp16", "bf16")
    carried = dtype if buildable else "bf16"
    env = {slot: carried for slot in ("activation", "weight", "kv", "output")}
    env["metadata"] = "fp32"
    inv = Invocation.from_job({"signature": signature, "op": op, "dims": dims,
                               "phase": "decode" if dims.get("s_query") == 1
                               else "prefill", "dtype_env": env})
    gen = torch.Generator(device=device).manual_seed(0)
    inputs = spec.make_inputs(inv, device, gen)
    ref = spec.reference(inputs).float()
    if buildable:
        with _InDtype(dtype):
            narrow = spec.reference(inputs)
    else:
        narrow = spec.reference(_quantised(inputs, dtype))
    return _statistics(narrow.float(), ref)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--signature", default="")
    parser.add_argument("--jobs", default="",
                        help="a plan's jobs.jsonl, for real cells")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtypes", default=",".join(DTYPES))
    args = parser.parse_args(argv)

    import torch
    from kernel_bench.driver import _XCHECK_BANDS
    device = args.device if torch.cuda.is_available() else "cpu"
    rows: Dict[Tuple[str, str], List[Tuple[float, float, float, int]]] = {}
    for signature, op, dims in _cells(args):
        for dtype in args.dtypes.split(","):
            try:
                got = _floor(signature, op, dims, dtype, device)
            except Exception as error:
                print(f"  {signature} {dtype}: {type(error).__name__}: "
                      f"{str(error)[:110]}", file=sys.stderr)
                continue
            if got is not None:
                rows.setdefault((signature, dtype), []).append(got)
    print(f"{'signature':<24}{'dtype':<18}{'RMS':>9}{'worst':>9}"
          f"{'old':>9}{'band':>8}{'RMS x':>8}{'worst x':>9}  cells")
    for (signature, dtype), got in sorted(rows.items()):
        rms = max(value[0] for value in got)
        limit = _MODEL_LIMIT * _EPSILON.get(dtype.split("/")[0], 1.0)
        if rms > limit:
            print(f"{signature:<24}{dtype:<18}  not a floor: {rms:.3g} is "
                  f"{rms / max(limit / _MODEL_LIMIT, 1e-12):.0f}x the dtype's "
                  f"epsilon — the narrowing model does not describe this "
                  f"signature")
            continue
        worst = max(value[1] for value in got)
        old = max(value[2] for value in got)
        band = _XCHECK_BANDS.get(dtype.split("/")[0])
        margin = ("" if band is None
                  else f"{band.typical / max(rms, 1e-9):>8.1f}")
        worst_margin = ("" if band is None
                        else f"{band.worst / max(worst, 1e-9):>9.1f}")
        print(f"{signature:<24}{dtype:<18}{rms:>9.4f}{worst:>9.4f}{old:>9.4f}"
              f"{'-' if band is None else format(band.typical, '.3g'):>8}"
              f"{margin}{worst_margin}  {len(got)}")
    print("\nRMS x / worst x: how much room the current band leaves above the "
          "floor.\nUnder 1 the band cannot be passed; far above it, the band "
          "is not measuring the dtype.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
