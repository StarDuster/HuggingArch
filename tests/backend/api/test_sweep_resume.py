"""What a cancelled Scaling sweep leaves behind, and how far it says it got.

Two defects these pin, both of them about work that is slow rather than
broken:

  * A (context × batch) sweep is ONE cell — one walk of the model serves
    every (context, batch) point by substitution — so neither the cells the
    server schedules nor the compute jobs it counts move for the sweep's
    whole run. On a cold large model that is half an hour of silence, past
    the page's inactivity budget, and the panel says the load failed while
    the server is still computing it. The cell reports its contexts as they
    finish instead.
  * Work no request waits on any more is cancelled (`offload.UNWATCHED_S`),
    which is what stops a user's clicks from stacking up — but a sweep
    stored nothing until the very end, so a cancelled sweep threw away
    everything it had priced. Each context is kept as it finishes, so the
    next request prices only what is left.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from backend.api import offload
from backend.measure.cache import KeyedCache, MemoryBackend
from backend.inference.sol_pool import SolPool
from tests.snapshots import snapshot_config
from tests.sweep_parity import FixtureBench, diff

MODEL = "meta-llama/Llama-3.2-3B"
CONTEXTS = [1024, 4096, 16384]


class _Stopped(Exception):
    """A sweep that stops part way, as a cancelled one does."""


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """The real sweep compute, offline, on fresh in-memory caches — the
    answers and the steps each in its own, as in the server — recording the
    contexts every cell was asked to price."""
    snapshot_config(MODEL)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    import backend.api.sweep_runtime as rt
    monkeypatch.setattr(rt, "fetch_config",
                        lambda m, overlay_dir=None: snapshot_config(m))
    monkeypatch.setattr(rt, "_SWEEP",
                        KeyedCache("sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(rt, "_SWEEP_STEP", KeyedCache("sweep-step", MemoryBackend()))
    monkeypatch.setattr(rt, "sharedSolPool", SolPool(tmp_path / "sol-pool"))
    asked: list = []
    real = rt._ESTIMATE.steps

    def recorded(cell, prices):
        asked.append(prices)
        yield from real(cell, prices)

    monkeypatch.setattr(rt, "_ESTIMATE",
                        dataclasses.replace(rt._ESTIMATE, steps=recorded))
    return rt, asked


def _request(scope, contexts=None):
    from backend.api._models import BatchPoint, InferenceEstimateSweepRequest
    return InferenceEstimateSweepRequest(
        model_id=MODEL, gpu_sku="H100-SXM",
        seq_lens=list(contexts or CONTEXTS),
        batch_grid=[BatchPoint(key="b1", prefill_batch_size=1, decode_batch_size=1),
                    BatchPoint(key="max", prefill_batch_size=-1,
                               decode_batch_size=None)],
        avg_output_tokens=256, selection_mode="latest",
    ).model_copy(update={"measurement_scope": scope})


def _run(rt, req):
    return asyncio.run(rt.compute_estimate_sweep_cached(req))


def test_a_sweep_cancelled_part_way_leaves_the_contexts_it_priced(
        runtime, monkeypatch, tmp_path):
    """A sweep that dies after two of three contexts has kept those two:
    the next request for it prices only the third, and answers exactly what
    an uninterrupted run answers.

    The walk is still read for every context — which walk a context reads
    off is decided by walking the ones before it — so a resumed sweep's
    anchors are the anchors of a run through, and its numbers are the same
    numbers, not merely close ones."""
    rt, asked = runtime
    with FixtureBench() as bench:
        whole = _run(rt, _request(bench.scope))

        # A fresh sweep of the same request, cancelled after two contexts.
        rt._SWEEP.clear()
        rt._SWEEP_STEP.clear()
        monkeypatch.setattr(rt, "sharedSolPool",
                            SolPool(tmp_path / "sol-cancelled"))
        asked.clear()
        real = rt._ESTIMATE.steps

        def dies(cell, prices):
            for n, (index, value) in enumerate(real(cell, prices)):
                if n == 2:
                    raise _Stopped("no request waits on this sweep any more")
                yield index, value

        cancelled = dataclasses.replace(rt._ESTIMATE, steps=dies)
        with pytest.raises(_Stopped):
            asyncio.run(rt._sweep_within_deadline(cancelled, _request(bench.scope),
                                                  False))

        # What it got through is kept; the answer is not.
        assert rt._SWEEP.get(
            rt.sweep_theory_identity("sweep", _request(bench.scope)).key) is None

        asked.clear()
        resumed = _run(rt, _request(bench.scope))

    # The resumed run priced only the context the cancelled one never reached.
    assert asked == [frozenset({2})], asked
    assert diff(whole, resumed) == [], diff(whole, resumed)


def test_a_sweep_whose_contexts_are_all_kept_prices_none_of_them(
        runtime, monkeypatch, tmp_path):
    """The steps alone answer the sweep again: with the answer and the
    theory half dropped but the contexts kept, the next request prices
    nothing and still answers the same."""
    rt, asked = runtime
    with FixtureBench() as bench:
        whole = _run(rt, _request(bench.scope))
        rt._SWEEP.clear()          # the answer and the theory half
        monkeypatch.setattr(rt, "sharedSolPool",
                            SolPool(tmp_path / "sol-steps-only"))
        asked.clear()
        again = _run(rt, _request(bench.scope))
    assert asked == [frozenset()], asked
    assert diff(whole, again) == [], diff(whole, again)


def test_a_sweeps_503_counts_its_contexts(runtime):
    """The sweep's progress is its contexts, so a request that misses the
    deadline reads a count that moves while the one cell runs. Before this,
    the cells were the count: 0 of 1 for the sweep's whole run."""
    rt, _asked = runtime
    seen: list = []

    async def scenario(req):
        async def watch():
            # Read the sweep's work as its one cell runs, as a request past
            # the deadline reads it for its 503.
            while True:
                # Fast enough to see the final tick before the successful work
                # leaves the table: the fold is memoised, so a context can be
                # cheaper than a human-scale poll interval.
                await asyncio.sleep(0.001)
                for work in list(offload._table().values()):
                    if work.progress is not None and (
                            not seen or seen[-1] != work.progress):
                        seen.append(work.progress)

        task = asyncio.ensure_future(rt.compute_estimate_sweep_cached(req))
        watcher = asyncio.ensure_future(watch())
        out = await task
        watcher.cancel()
        return out

    with FixtureBench() as bench:
        asyncio.run(scenario(_request(bench.scope)))

    assert seen, "the sweep reported no progress at all"
    assert all(total == len(CONTEXTS) for _done, total in seen), seen
    # It moved: the last count is past the first, and never went backwards.
    dones = [done for done, _total in seen]
    assert dones == sorted(dones) and dones[-1] == len(CONTEXTS), dones
