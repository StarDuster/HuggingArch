"""kernel_bench — cross-library operator perf harness.

Produces MEASURED kernel timings that fill the calibration corpus the
inference estimate consumes (`backend/inference/benchmarks.py` schema →
`hbench upload` → Kernel Bench → the estimates' lookup).

Design (see memory `project_kernel_bench_harness`):

- The reference for an op is the MEASURED ENVELOPE across many
  high-performance libraries (FA3, FlashInfer, FlashMLA, DeepGEMM, FLA,
  vLLM, sglang …), NOT any single framework's dispatch choice. Each
  library that implements an op-signature registers a `Provider`; the
  driver measures every applicable provider at a shape; the consumer's
  `best_of` picks the fastest.

- Libraries are OPTIONAL third-party extras, never core deps. The unit
  of deployment is one isolated env per pinned (library, version) stack.
  Provider modules are import-guarded: in a given env only the providers
  whose libraries are importable self-register.

- This package is a PRODUCER. It imports the schema from
  `backend.inference.benchmarks` (one contract, never forked) but the
  core (`backend.inference`) must NEVER import kernel_bench — kernel_bench
  carries GPU + heavy library deps that the core stays free of.

Two entrypoints (`python -m kernel_bench {plan,run}`):
  plan — CPU only: enumerate the (op, shape, dtype) jobs a real serving
         run would hit (reuses the analyze pipeline). Emits jobs.jsonl.
  run  — GPU: in a provider env, time every applicable provider for each
         job, cross-check numerics, emit measured CSV rows.
"""
