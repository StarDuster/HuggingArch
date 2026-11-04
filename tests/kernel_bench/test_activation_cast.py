"""The activation cast: one signature, two directions, read off the cell.

`quantise_activation` is the launch a quantised matmul runs on its operand —
vLLM quantises at `BlockScaledMMLinearKernel.py:119` and runs the GEMM at
`:132`, sglang at `fp8_utils.py:1355` — and until it was a cell the harness
performed the same arithmetic as an UNTIMED preparation step inside the GEMM
builds, which is exactly the launch a server pays for and nothing was
pricing.

What is tested here is the reading, not the kernels: which dtypes the cast is
between, which elements share a scale, and that the oracle is a grid. Those
are the facts a provider and the reference must agree on, and a cell that
cannot state one of them has to raise rather than take a default — per-token,
per-128-group and per-32-group are three different kernels between the same
two dtypes, so a cast measured under one and billed under another is a row
nothing can tell apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch")
import torch

from kernel_bench.invocation import Invocation
from kernel_bench.quant import ActivationCast

_DIMS = {"B": 2, "s_query": 8, "d_model": 256}


def _cell(*, reads: str, writes: str, granularity: str = "per-group:128",
          metadata: str = "fp32", dims=None) -> Invocation:
    return Invocation.from_job({
        "signature": "quantise_activation",
        "op": "quantise_activation",
        "dims": dict(dims or _DIMS),
        "phase": "prefill",
        "dtype_env": {"activation": reads, "output": writes,
                      "metadata": metadata,
                      "activation_granularity": granularity},
    })


class TestTheDirectionIsTheCells:
    def test_quantise_and_dequantise_are_one_cast_read_two_ways(self):
        """The same pair of dtypes, differing only in which end is written.

        So they are two cells of one signature rather than two signatures,
        and they share an oracle: a dequantise kernel is right exactly when
        it returns what the narrow format holds, which is what a quantise
        kernel's output dequantises to.
        """
        down = ActivationCast.of(_cell(reads="bf16", writes="fp8_e4m3"))
        up = ActivationCast.of(_cell(reads="fp8_e4m3", writes="bf16"))

        assert (down.narrow, down.wide) == ("fp8_e4m3", "bf16")
        assert (up.narrow, up.wide) == ("fp8_e4m3", "bf16")
        assert down.group == up.group
        assert down.narrows is True and up.narrows is False

    def test_a_cast_between_two_wide_ends_is_not_a_cast(self):
        with pytest.raises(ValueError, match="one quantised format"):
            ActivationCast.of(_cell(reads="bf16", writes="fp16"))

    def test_a_cast_between_two_narrow_ends_is_not_one_either(self):
        with pytest.raises(ValueError, match="one quantised format"):
            ActivationCast.of(_cell(reads="fp8_e4m3", writes="fp8_e5m2"))

    def test_one_format_on_both_slots_is_not_a_cast_either(self):
        """fp8 on both slots is not a cast — and `of` does not quietly read
        it as a MOVE either: whether a launch moves a record is the op's
        declaration (`TestWhichLaunchesMoveARecord`), not the two slots
        agreeing. Read as a move here, a same-format launch that COMPUTES
        (a requantisation, an FP8-in FP8-out activation) would be measured
        as a permutation of its input."""
        with pytest.raises(ValueError, match="one quantised format"):
            ActivationCast.of(_cell(reads="fp8_e4m3", writes="fp8_e4m3"))


class TestWhichLaunchesMoveARecord:
    """A launch that writes its operand's elements unchanged in another
    order moves the whole RECORD — the elements and each row's scale. Which
    launches those are comes from the op the signature binds, whose
    `gather_source` names the input its output is an index gather of; an
    MoE dispatch's FP8 cells had no provider at all before (934 on the
    H100) and this is what they are now measured as."""

    def test_the_moving_launches_are_the_ops_that_declare_a_gather(self):
        from kernel_bench.quant import gathers_a_record
        assert gathers_a_record("moe_permute") is True
        assert gathers_a_record("gather") is True
        # Its own sibling REDUCES (the weighted sum), an activation
        # epilogue computes new values, and a cast writes another format:
        # the dtype slots cannot tell those from a permutation, the op can.
        assert gathers_a_record("moe_combine") is False
        assert gathers_a_record("quantise_activation") is False
        assert gathers_a_record("silu_and_mul") is False

    def test_a_moved_record_is_one_record_on_the_cells_own_grid(self):
        inv = Invocation.from_job({
            "signature": "moe_permute", "op": "permute", "phase": "decode",
            "dims": {"B": 1, "s_query": 4, "d_model": 256, "n_experts": 2,
                     "s_expert_rows": 2, "n_tokens": 4},
            "dtype_env": {"activation": "fp8_e4m3", "output": "fp8_e4m3",
                          "metadata": "fp32",
                          "activation_granularity": "per-group:128"}})
        moved = ActivationCast.moved_by(inv)
        assert (moved.narrow, moved.wide) == ("fp8_e4m3", "fp32")
        assert moved.narrows is False
        assert moved.group == 128
        x = torch.randn(4, 256)
        assert torch.equal(moved.on_grid(x), moved.on_grid(moved.on_grid(x)))

    def test_a_computing_launch_is_refused_a_moved_record(self):
        """The rule the review asked for: a same-format cell of a launch
        that computes raises instead of being read as a move."""
        inv = Invocation.from_job({
            "signature": "quantise_activation", "op": "quantise_activation",
            "phase": "decode", "dims": dict(_DIMS),
            "dtype_env": {"activation": "fp8_e4m3", "output": "fp8_e4m3",
                          "metadata": "fp32",
                          "activation_granularity": "per-group:128"}})
        with pytest.raises(ValueError, match="computes its output"):
            ActivationCast.moved_by(inv)

    def test_a_moved_record_of_two_formats_is_refused(self):
        inv = Invocation.from_job({
            "signature": "moe_permute", "op": "permute", "phase": "decode",
            "dims": {"B": 1, "s_query": 4, "d_model": 256, "n_experts": 2,
                     "s_expert_rows": 2, "n_tokens": 4},
            "dtype_env": {"activation": "fp8_e4m3", "output": "fp8_e5m2",
                          "metadata": "fp32",
                          "activation_granularity": "per-group:128"}})
        with pytest.raises(ValueError, match="one record"):
            ActivationCast.moved_by(inv)


class TestWhichElementsShareAScale:
    def test_per_token_is_the_rows_width_not_one_scale_for_the_tensor(self):
        # 0 would mean one scale for the whole TENSOR, which is a static
        # per-tensor scale and a different kernel — no activation format
        # declares one.
        cast = ActivationCast.of(_cell(reads="bf16", writes="fp8_e4m3",
                                       granularity="per-token"))
        assert cast.group == _DIMS["d_model"]

    def test_a_group_is_the_group_the_format_states(self):
        for width in (32, 128):
            cast = ActivationCast.of(
                _cell(reads="bf16", writes="fp8_e4m3",
                      granularity=f"per-group:{width}"))
            assert cast.group == width

    def test_a_cell_that_states_no_granularity_refuses(self):
        """Because per-token, per-128 and per-32 are three kernels.

        Defaulting would measure one and bill another, and the row would
        carry no trace of which.
        """
        cell = _cell(reads="bf16", writes="fp8_e4m3")
        cell.dtype_env.pop("activation_granularity")
        with pytest.raises(ValueError, match="activation_granularity"):
            ActivationCast.of(cell)


class TestTheOracleIsAGrid:
    def test_casting_what_is_already_on_the_grid_changes_nothing(self):
        """`on_grid` is idempotent, which is what makes it a grid.

        It is also the property a cast kernel must satisfy and the reason
        this cell's cross-check expects ~0 rather than a band: the kernel
        and the oracle run the same algorithm, so a disagreement is about
        the granularity or the scale convention — a bug in one of them, not
        precision loss to tolerate.
        """
        cast = ActivationCast.of(_cell(reads="bf16", writes="fp8_e4m3"))
        x = torch.randn(16, _DIMS["d_model"], dtype=torch.float32)
        once = cast.on_grid(x)
        twice = cast.on_grid(once)
        assert torch.equal(once.float(), twice.float())

    def test_a_narrower_group_holds_the_values_at_least_as_well(self):
        """One scale per 32 elements cannot be worse than one per 128.

        A scale covers a group's dynamic range, so splitting the group can
        only help — the check that a group size is being USED rather than
        parsed and dropped.
        """
        x = torch.randn(16, _DIMS["d_model"], dtype=torch.float32)
        # One outlier per 128-wide group: the coarse scale must stretch to
        # cover it and the fine one need not.
        x[:, 0] *= 400.0
        wide = ActivationCast.of(_cell(reads="bf16", writes="fp8_e4m3",
                                       granularity="per-group:128"))
        fine = ActivationCast.of(_cell(reads="bf16", writes="fp8_e4m3",
                                       granularity="per-group:32"))
        err = lambda cast: (cast.on_grid(x).float() - x).abs().sum().item()
        assert err(fine) < err(wide)


class TestTheScaleDtypeChoosesTheEntryPoint:
    """The scale dtype the cell states is the encoding the oracle quantises
    to (`quant.activation_scale_encoding`): exact fp32, or a power of two
    (`ue8m0`). An entry point that writes the other encoding quantises the
    same row to other values, so each provider's declaration must decline
    the cell of the encoding it does not write — before this gate the
    fp32 and ue8m0 quantisers each accepted the other's cells, and only
    the cross-check could have said so, after the GPU time was spent."""

    @staticmethod
    def _accepting(cell):
        from kernel_bench.providers import all_providers, discover
        discover()
        return {p.kernel_impl for p in all_providers()
                if p.serves is not None and not p.disabled_reason
                and p.serves.signature == "quantise_activation"
                and p.serves.accepts(cell)}

    def test_an_e8m0_cell_goes_to_the_kernels_that_write_powers_of_two(self):
        served = self._accepting(_cell(reads="bf16", writes="fp8_e4m3",
                                       metadata="ue8m0"))
        assert served == {"torch_activation_quantise",
                          "vllm_per_token_group_quant_fp8_ue8m0",
                          "vllm_per_token_group_quant_fp8_packed_for_deepgemm",
                          "sglang_per_token_group_quant_fp8_ue8m0"}

    def test_an_exact_cell_goes_to_the_kernels_that_write_fp32(self):
        served = self._accepting(_cell(reads="bf16", writes="fp8_e4m3",
                                       metadata="fp32"))
        assert served == {"torch_activation_quantise",
                          "vllm_per_token_group_quant_fp8",
                          "sglang_per_token_group_quant_fp8"}
