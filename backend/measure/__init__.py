"""Measurements: what an estimate's measured numbers are read under, and how
they are read (docs/measured_calibration.md §7).

Measured kernel times live in Kernel Bench, a lake only the Kernel Bench
service (`backend/service/bench/server.py`) opens. They are consumed at two
granularities, and this package is where both are named — with the caches
the results are kept in:

  scope.py    end to end — an estimate's latency, cached. `MeasurementScope`
              (the owner whose rows join the shared ones, the lake, its
              snapshot, the annotation rules — or the constant `NONE`) and
              `ResultIdentity`, the one answer to what identifies an
              estimate's result: its cache key (`make_key`, the one key
              hash), its remote-tier eligibility, and the scope digest a
              response reports (informational; the browser caches theory
              only, and keys on no digest). Every estimate cache in
              `backend/api` (single-flight, sweeps, gallery, warm) takes its
              key from here.
  lookup.py   per op, exact — a physical op's key form (signature, projected
              shape, dtype env, gpu, phase), the client of
              `/internal/bench/lookup` and `/internal/bench/version`
              (`HUGGINGARCH_BENCH_URL`, the shared secret; three outcomes:
              no service → nothing read, an answer, a failure → 503), and
              the coverage question the kernel_bench harness asks (`covered`:
              which implementations already measured a key).
  cache.py    the result caches: `KeyedCache` over memory / disk / tiered /
              write-behind backends, single-flight, and a remote tier the
              service layer binds (`set_remote`). Every cached result keys
              its entry through scope.py.
  pool.py     the persistent process pool the heavy work runs on: one job
              (`submit`, the server's requests) or a sweep's cells in order
              (`run_parallel`, `map_cells`).
  auto_sweep.py  the sweep engine: one walk per cell, substitution per
              point, the union of every point's keys looked up once, and the
              all-or-nothing measured track with its coverage
              (docs/auto_sweep.md). It knows nothing about models: the
              estimator supplies the walk, the laws, the keys and the
              selection, and imports it.

What lives elsewhere: indexing looked-up rows into the registry the overlay
reads, under the key that selected each (`backend/inference/op_calibration.py`,
`registry_for`); which keys an estimate asks (`calibration_overlay.OverlayWalk`);
the service's side (`backend/service/bench/lookup.py`, `routes.py`).

Who may import it: `backend.inference`, `backend.api`, `backend.service`,
`backend.playground` and the kernel_bench harness. It imports none of
`backend.inference`, `backend.api`, `backend.service`, `backend.rendering`
or kernel_bench (inference imports it; tests/meta/test_eval_boundary.py
`DIRECTION_BANS` holds the rule, test_service_layout.py the service half).
"""
