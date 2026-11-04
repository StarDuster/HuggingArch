"""Regression tests for the prefill/decode batch-size split.

The activation-bound prefill ceiling and the KV-bound decode ceiling are
two physically-distinct constraints — coupling them as one `max_batch`
materially over-counts prefill compute time. These tests lock the three
contracts the split must preserve:

  1. `estimate_phases(prefill_batch=P, decode_batch=D)` puts each phase
     at its own B independently.
  2. `estimate_capacity(...)` ships BOTH a `max_batch_per_gpu` legacy
     alias AND a `max_prefill_batch_per_gpu` activation-bound sibling.
  3. The sweep-route per-seq entry carries `max_prefill_batch_per_gpu`
     (otherwise the frontend can't enforce its per-phase ceiling at
     scaling-chart B's).
"""
from __future__ import annotations

import sys
import unittest

from tests.snapshots import resolve_snapshot, shared_loader, snapshot_config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.inference.capacity import (
    estimate_capacity,
    estimate_capacity_sweep,
)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import PhaseInputs, estimate_phases


_MODEL = "meta-llama/Llama-3.2-3B"
_GPU = "H100-SXM"


def _skip_if_no_spec():
    cfg = snapshot_config(_MODEL)
    resolve_snapshot(_MODEL)
    return cfg


def _phases_kwargs():
    """A real spec and a one-leaf breakdown per phase. Neither is what these
    tests are about — a phase is a view of a step, so `estimate_phase`
    requires both, and the claim below is only about which batch each phase
    is put at."""
    _skip_if_no_spec()
    row = [{"access_bytes_per_rank": 1_000_000, "t_compute_s": 0.001,
            "t_memory_s": 0.001, "t_comm_s": 0.0, "t_launch_s": 0.0,
            "t_op_s": 0.001}]
    return dict(
        prefill_rows=list(row),
        decode_rows=list(row),
    )


def _phase_inputs():
    """The deployment half, which these tests hold fixed."""
    cfg = _skip_if_no_spec()
    return PhaseInputs(
        avg_input_tokens=2048,
        avg_output_tokens=128,
        weight_bytes_per_rank=1_000_000_000,
        per_token_kv_bytes_per_rank=1024,
        static_kv_bytes_per_rank=0,
        resolved_spec=shared_loader().match(cfg, model_id=_MODEL),
        parallel=ParallelConfig(),
    )


class TestEstimatePhasesBatchSplit:
    """Phase-level kwarg resolution."""

    def test_split_kwargs_decouple_phases(self):
        result = estimate_phases(
            _phase_inputs(),
            prefill_batch=1,
            decode_batch=8,
            **_phases_kwargs(),
        )
        assert result["prefill"]["batch"] == 1
        assert result["decode"]["batch"] == 8

    def test_no_batch_kwargs_defaults_to_one(self):
        result = estimate_phases(_phase_inputs(), **_phases_kwargs())
        assert result["prefill"]["batch"] == 1
        assert result["decode"]["batch"] == 1


class TestEstimateCapacityResponseKeys:
    """`estimate_capacity` response must carry the new prefill ceiling
    alongside the legacy decode alias."""

    def _kwargs(self):
        return dict(
            gpu_sku=_GPU,
            avg_input_tokens=2048,
            avg_output_tokens=128,
            parallel=ParallelConfig(),
            model_id=_MODEL,
            selection_mode="theoretical_only",
        )

    def test_response_has_legacy_alias_and_split_siblings(self):
        cfg = _skip_if_no_spec()
        result = estimate_capacity(config=cfg, **self._kwargs())
        # Legacy alias preserved for /compare.html and external callers.
        assert "max_batch_per_gpu" in result
        # New per-phase siblings.
        assert "max_decode_batch_per_gpu" in result
        assert "max_prefill_batch_per_gpu" in result
        assert "max_decode_batch_total" in result
        assert "max_prefill_batch_total" in result
        # Alias === decode (KV-bound) is the explicit contract: external
        # callers reading the legacy key get decode's number, not a
        # silently-changed activation-bound one.
        assert result["max_batch_per_gpu"] == result["max_decode_batch_per_gpu"]

    def test_max_prefill_batch_is_activation_bound(self):
        """Prefill ceiling derives from the residual activation pool
        ((1 − gpu_memory_fraction) × HBM − overhead) ÷ peak_act_per_b1,
        NOT the KV budget. It must therefore be independent of
        per-request KV bytes — sweeping avg_input_tokens (which scales
        per-request KV linearly) leaves the prefill ceiling unchanged
        modulo activation peak's own seq_len dependence."""
        cfg = _skip_if_no_spec()
        kw = self._kwargs()
        short = estimate_capacity(config=cfg, **{**kw, "avg_input_tokens": 1024})
        # Decode ceiling (KV-bound) MUST drop as seq doubles. Prefill
        # ceiling (activation-bound) may also move because peak_act is
        # measured at this seq, but the two move on different curves —
        # we just lock that both are present and positive when the
        # model fits.
        long = estimate_capacity(config=cfg, **{**kw, "avg_input_tokens": 4096})
        if short.get("fits") and long.get("fits"):
            assert short["max_prefill_batch_per_gpu"] >= 1
            assert long["max_prefill_batch_per_gpu"] >= 1
            # The legacy KV-bound alias must drop with longer context
            # (more KV per request → fewer requests fit). This is the
            # behavior the split decouples from prefill.
            assert short["max_batch_per_gpu"] >= long["max_batch_per_gpu"]

    def test_chunked_prefill_plateaus_activation_bound_batch(self):
        """A serving chunk caps the prefill activation seq, so
        `max_prefill_batch` (and the peak itself) stay put once the
        prompt is longer than the chunk. Unchunked (chunk=0) keeps
        decaying with seq."""
        cfg = _skip_if_no_spec()
        kw = {**self._kwargs(), "config": cfg}
        chunk = 2048
        at_chunk = estimate_capacity(
            **{**kw, "avg_input_tokens": chunk, "prefill_chunk_size": chunk})
        past_chunk = estimate_capacity(
            **{**kw, "avg_input_tokens": chunk * 4, "prefill_chunk_size": chunk})
        unchunked = estimate_capacity(
            **{**kw, "avg_input_tokens": chunk * 4, "prefill_chunk_size": 0})
        if not (at_chunk.get("fits") and past_chunk.get("fits")
                and unchunked.get("fits")):
            return
        assert at_chunk["max_prefill_batch_per_gpu"] == (
            past_chunk["max_prefill_batch_per_gpu"])
        at_peak = (at_chunk.get("activation_peak") or {}).get(
            "peak_activation_bytes_per_rank")
        past_peak = (past_chunk.get("activation_peak") or {}).get(
            "peak_activation_bytes_per_rank")
        unchunked_peak = (unchunked.get("activation_peak") or {}).get(
            "peak_activation_bytes_per_rank")
        assert at_peak == past_peak
        assert unchunked_peak > past_peak
        assert unchunked["max_prefill_batch_per_gpu"] <= (
            past_chunk["max_prefill_batch_per_gpu"])


class TestSweepResponseKeys:
    """The sweep route's per-seq entry must carry the same prefill key
    so the scaling chart can clamp per-seq B sweeps to each phase's own
    ceiling."""

    def test_each_seq_carries_max_prefill_batch(self):
        cfg = _skip_if_no_spec()
        seq_lens = [1024, 4096, 16384]
        results = estimate_capacity_sweep(
             seq_lens=seq_lens,
            config=cfg,
            gpu_sku=_GPU,
            avg_output_tokens=128,
            parallel=ParallelConfig(),
            model_id=_MODEL,
            selection_mode="theoretical_only",
        )
        assert len(results) == len(seq_lens)
        for seq, r in zip(seq_lens, results, strict=True):
            assert "max_prefill_batch_per_gpu" in r, (
                f"seq={seq}: sweep result missing "
                f"max_prefill_batch_per_gpu — scaling-chart dropdown "
                f"won't be able to cap its synced sweep without it"
            )
            assert "max_batch_per_gpu" in r  # legacy alias preserved


if __name__ == "__main__":
    unittest.main()
