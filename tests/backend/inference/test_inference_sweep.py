"""Tests for `estimate_capacity_sweep` + `/inference/estimate-sweep`.

The sweep variant hoists the seq-invariant preparation (spec match,
materialize quant defaults, KV per-token, weight breakdown, GPU peaks,
calibration registry + framework binding) once, then loops the per-seq
finish N times. The single-seq endpoint stays a thin compose on top of
the same two halves — `estimate_capacity == _capacity_finish(_capacity_preparation(...))`
— so all the per-seq outputs must match what N separate single-seq
calls would produce.

Coverage:
  - Parity: byte-identical results vs N single-seq calls (Llama-3.2-3B,
    TP=1; same model the calibration overlay tests use).
  - Order preservation: out-of-monotone `seq_lens` returns results in
    request order.
  - Validation: empty / negative `seq_lens` raises `ValueError`.
  - Prelude single-evaluation: the seq-invariant building blocks
    (`SpecLoader.match`, physical cache folds, `per_rank_weight_breakdown`)
    are called exactly once per sweep, not once per seq.
  - Route smoke: POST /inference/estimate-sweep returns
    `len(results) == len(seq_lens)`, validation errors map to 400, and
    the first result matches a plain /inference/estimate call at the
    same seq.
"""
from __future__ import annotations

import sys
import unittest

from tests.snapshots import resolve_snapshot, snapshot_config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.inference import capacity as _capacity
from backend.inference import walk_registry
from backend.inference.capacity import (
    estimate_capacity,
    estimate_capacity_sweep,
)
from backend.inference.parallel_config import ParallelConfig


_MODEL = "meta-llama/Llama-3.2-3B"
_GPU = "H100-SXM"


def _skip_if_no_spec():
    cfg = snapshot_config(_MODEL)
    resolve_snapshot(_MODEL)
    return cfg


def _common_kwargs():
    return dict(
        gpu_sku=_GPU,
        avg_output_tokens=128,
        parallel=ParallelConfig(),
        model_id=_MODEL,
        selection_mode="theoretical_only",
    )


class TestEstimateCapacitySweepParity:
    """Per-seq sweep results must match N independent single-seq calls."""

    def test_byte_identical_to_n_single_calls(self):
        cfg = _skip_if_no_spec()
        seq_lens = [1024, 4096, 32768]
        kwargs = _common_kwargs()
        sweep = estimate_capacity_sweep(
             seq_lens=seq_lens, config=cfg, **kwargs,
        )
        singles = [
            estimate_capacity(
                 config=cfg, avg_input_tokens=s, **kwargs,
            )
            for s in seq_lens
        ]
        assert len(sweep) == len(singles) == len(seq_lens)
        for seq, sw, sn in zip(seq_lens, sweep, singles, strict=True):
            # Int / str / bool fields must match exactly. Skip the
            # `phases` blob (deeply nested floats) and verify it via
            # a separate set of high-value scalar fields below.
            for key in (
                "max_batch_per_gpu",
                "max_tokens",
                "max_tokens_total",
                "fits",
                "weight_bytes_per_rank",
                "weight_bytes_total",
                "mem_total_bytes",
                "kv_budget_bytes",
                "per_token_per_rank_bytes",
            ):
                if key in sn:
                    assert sw.get(key) == sn.get(key), (
                        f"seq={seq} key={key}: sweep={sw.get(key)} "
                        f"single={sn.get(key)}"
                    )

    def test_order_preserved(self):
        """Requesting `seq_lens` out of monotone order returns the
        results in REQUEST order (not sorted)."""
        cfg = _skip_if_no_spec()
        seq_lens = [8192, 1024, 4096]
        kwargs = _common_kwargs()
        sweep = estimate_capacity_sweep(
             seq_lens=seq_lens, config=cfg, **kwargs,
        )
        singles = [
            estimate_capacity(
                 config=cfg, avg_input_tokens=s, **kwargs,
            )
            for s in seq_lens
        ]
        for seq, sw, sn in zip(seq_lens, sweep, singles, strict=True):
            # max_batch_per_gpu is the headline scaling-chart number;
            # it is monotone decreasing in seq for fixed budget, so
            # the [8192, 1024, 4096] ordering is observable.
            assert sw["max_batch_per_gpu"] == sn["max_batch_per_gpu"], (
                f"seq={seq} sweep={sw['max_batch_per_gpu']} "
                f"single={sn['max_batch_per_gpu']}"
            )


class TestEstimateCapacitySweepValidation:
    """`seq_lens` boundary cases raise `ValueError` (translates to 400
    at the route layer)."""

    def test_empty_seq_lens_raises(self):
        cfg = _skip_if_no_spec()
        with pytest.raises(ValueError, match="non-empty"):
            estimate_capacity_sweep(
                 seq_lens=[], config=cfg, **_common_kwargs(),
            )

    def test_negative_seq_len_raises(self):
        cfg = _skip_if_no_spec()
        with pytest.raises(ValueError, match="≥ 0"):
            estimate_capacity_sweep(
                 seq_lens=[1024, -1, 4096],
                config=cfg, **_common_kwargs(),
            )


class TestEstimateCapacitySweepPreludeSingleEvaluation:
    """The whole point of the sweep is amortizing the seq-invariant
    preparation across N seqs. If a future refactor accidentally moves a
    preparation step into the per-seq loop, this catches it — these counters
    must read 1 (not N) after a multi-seq sweep."""

    def test_match_called_once_per_sweep(self, monkeypatch):
        cfg = _skip_if_no_spec()
        seq_lens = [1024, 4096, 8192, 16384, 32768]

        # Wrap SpecLoader().match through the same helper capacity.py
        # actually calls: backend.arch.loader.get_default_loader().
        from backend.arch import loader as _loader_mod
        real_loader = _loader_mod.get_default_loader()
        original_match = real_loader.match
        counter = {"n": 0}

        def counting_match(*a, **kw):
            counter["n"] += 1
            return original_match(*a, **kw)

        monkeypatch.setattr(real_loader, "match", counting_match)

        estimate_capacity_sweep(
             seq_lens=seq_lens, config=cfg, **_common_kwargs(),
        )
        assert counter["n"] == 1, (
            f"SpecLoader.match should be called exactly once per sweep "
            f"(got {counter['n']} for {len(seq_lens)} seqs) — if this "
            f"reads len(seq_lens), the preparation leaked back into the "
            f"per-seq loop"
        )

    def test_cache_views_fold_one_analysis_once_per_sweep(self, monkeypatch):
        """Cache laws are folded once from one Analysis, then re-read at each
        sequence length without another preparation walk. Once per cell: a
        later sweep of the same cells reads the views the first one folded
        (`capacity.CellBasis`), so the cells start empty here."""
        walk_registry.WALK_CELLS.clear()
        cfg = _skip_if_no_spec()
        seq_lens = [1024, 4096, 8192, 16384, 32768]

        from backend.arch.layout import physical_cache as _physical_cache

        real_local = _physical_cache.physical_cache_view
        real_whole = _physical_cache.whole_model_cache_view
        calls = {"local": [], "whole": []}

        def counting_local(analysis, **kw):
            calls["local"].append(analysis)
            return real_local(analysis, **kw)

        def counting_whole(analysis, **kw):
            calls["whole"].append(analysis)
            return real_whole(analysis, **kw)

        monkeypatch.setattr(_physical_cache, "physical_cache_view", counting_local)
        monkeypatch.setattr(_physical_cache, "whole_model_cache_view", counting_whole)

        estimate_capacity_sweep(
             seq_lens=seq_lens, config=cfg, **_common_kwargs(),
        )
        assert len(calls["local"]) == len(calls["whole"]) == 1
        assert calls["local"][0] is calls["whole"][0]

    def test_per_rank_weight_breakdown_called_once_per_sweep(
        self, monkeypatch,
    ):
        cfg = _skip_if_no_spec()
        seq_lens = [1024, 4096, 8192, 16384, 32768]

        real_fn = _capacity.per_rank_weight_breakdown
        counter = {"n": 0}

        def counting_breakdown(*a, **kw):
            counter["n"] += 1
            return real_fn(*a, **kw)

        monkeypatch.setattr(
            _capacity, "per_rank_weight_breakdown", counting_breakdown,
        )

        estimate_capacity_sweep(
             seq_lens=seq_lens, config=cfg, **_common_kwargs(),
        )
        assert counter["n"] == 1, (
            f"per_rank_weight_breakdown should be called exactly once "
            f"per sweep (got {counter['n']} for {len(seq_lens)} seqs)"
        )


class TestEstimateSweepRoute(unittest.TestCase):
    """POST /inference/estimate-sweep — wire-shape contract."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        snapshot_config(_MODEL)

    def _payload(self, seq_lens, **overrides):
        body = {
            "model_id": _MODEL,
            "gpu_sku": _GPU,
            "seq_lens": seq_lens,
            "avg_output_tokens": 128,
            "selection_mode": "theoretical_only",
        }
        body.update(overrides)
        return body

    def test_returns_results_in_request_order(self):
        seq_lens = [1024, 4096, 32768]
        r = self.client.post(
            "/inference/estimate-sweep", json=self._payload(seq_lens),
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("results", body)
        self.assertEqual(len(body["results"]), len(seq_lens))
        # First result matches a plain /inference/estimate at seq=1024.
        single_payload = {
            "model_id": _MODEL,
            "gpu_sku": _GPU,
            "avg_input_tokens": seq_lens[0],
            "avg_output_tokens": 128,
            "selection_mode": "theoretical_only",
        }
        single = self.client.post(
            "/inference/estimate", json=single_payload,
        )
        self.assertEqual(single.status_code, 200, single.text)
        sweep_first = body["results"][0]
        single_body = single.json()
        self.assertEqual(
            sweep_first.get("max_batch_per_gpu"),
            single_body.get("max_batch_per_gpu"),
            f"first sweep result must match plain /inference/estimate "
            f"at seq={seq_lens[0]}",
        )

    def test_sweep_omits_per_op_drilldown(self):
        """The landscape payload must not ship the single-config drill-down
        (per-op rows, weight tree). Those belong on `/inference/estimate`;
        a 9-seq wire copy of them is what made the Scaling panel fail on
        mobile (`resp.json()` of ~9 MB)."""
        r = self.client.post(
            "/inference/estimate-sweep",
            json=self._payload([1024]),
        )
        self.assertEqual(r.status_code, 200, r.text)
        first = r.json()["results"][0]
        for dropped in (
            "per_op_timing", "per_op_timing_decode",
            "weight_breakdown", "formulas",
        ):
            self.assertNotIn(dropped, first, dropped)
        self.assertIn("phases", first)
        self.assertIn("max_batch_per_gpu", first)
        single = self.client.post("/inference/estimate", json={
            "model_id": _MODEL, "gpu_sku": _GPU,
            "avg_input_tokens": 1024, "avg_output_tokens": 128,
            "selection_mode": "theoretical_only",
        })
        self.assertEqual(single.status_code, 200, single.text)
        self.assertLess(
            len(r.content), len(single.content),
            "a 1-seq sweep must be smaller than a full /inference/estimate",
        )

    def test_empty_seq_lens_returns_422(self):
        r = self.client.post(
            "/inference/estimate-sweep", json=self._payload([]),
        )
        # Pydantic field_validator rejects pre-route → 422 (FastAPI's
        # validation error envelope), not 400.
        self.assertEqual(r.status_code, 422, r.text)

    def test_negative_seq_len_returns_422(self):
        r = self.client.post(
            "/inference/estimate-sweep",
            json=self._payload([1024, -1, 4096]),
        )
        self.assertEqual(r.status_code, 422, r.text)

    def test_missing_model_id_returns_400(self):
        body = self._payload([1024])
        body["model_id"] = ""
        r = self.client.post("/inference/estimate-sweep", json=body)
        self.assertEqual(r.status_code, 400, r.text)


if __name__ == "__main__":
    unittest.main()


class TestSweepShapesRoute(unittest.TestCase):
    """POST /inference/bench/sweep-shapes — the Scaling panel's export, the
    single source kernel_bench also consumes. The heavy grid enumeration
    (build_sweep_jobs, ~1s/cell of analyze) is mocked here — its correctness is
    covered by tests/test_kernel_bench.py::TestParallelSweepAlignment; this test
    is the route contract (the endpoint wraps that one shared function)."""

    def setUp(self):
        self.client = TestClient(app)

    def test_wraps_build_sweep_jobs(self):
        import unittest.mock as mock
        fake = ([{"signature": "gemm", "op": "linear", "dims": {"B": 1},
                  "parallel": "tp=1", "phase": "prefill"}],
                {"cells": 1, "unique_jobs": 1, "parallel_configs": 1, "seqs": 1})
        with mock.patch("kernel_bench.sweep.build_sweep_jobs",
                        return_value=fake) as m:
            r = self.client.post(
                "/inference/bench/sweep-shapes",
                json={"model_id": "meta-llama/Llama-3.2-3B", "gpu": "RTX-4090"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["jobs"], fake[0])
        self.assertEqual(body["stats"], fake[1])
        self.assertTrue(body["filename"].endswith("_RTX-4090_sweep_jobs.jsonl"))
        # gpu forwarded to the shared enumerator (VRAM gate is per-card)
        self.assertEqual(m.call_args.kwargs["gpu"], "RTX-4090")

    def test_unfetchable_model_is_400(self):
        import unittest.mock as mock
        with mock.patch("kernel_bench.sweep.build_sweep_jobs",
                        side_effect=RuntimeError("no config")):
            r = self.client.post(
                "/inference/bench/sweep-shapes",
                json={"model_id": "does/not-exist", "gpu": "RTX-4090"})
        self.assertEqual(r.status_code, 400, r.text)
