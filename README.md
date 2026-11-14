# Kimi-K3.1 reports

Reports and benchmark evidence for the Kimi-K3.1 SGLang implementation.
This branch contains research artifacts separately from the implementation PR.

- [Model and runtime](docs/kimi-k31-sglang.md)
- [DCP8 execution model](docs/kimi-k31-dcp.md) and [B300 validation](docs/kimi-k31-dcp-validation.md)
- [EP1/EP2 and prefill CUDA graph measurements](docs/kimi-k31-b300-benchmark.md)
- [Colocated serving recipe](docs/recipes/kimi-k31-colocated.md)
- [EP1/2/4/8 single-GPU microbench](benchmarks/ep148-sweep/summary.md)
- [Timing fixture measurements](benchmarks/timing-fixture/REPORT.md)
- [Review handoff](reviews/gemini-review-handoff.md) and [resolution](reviews/gemini-review-resolution.md)

The reports and raw calibration files are preserved from implementation commits
`43d0cf9` and `9f97291`. Statements about files being external to Git describe
that implementation branch; the archived files are included here. Review and
validation results identify the revision they tested. The implementation PR
head at archival time is `5fe9dcc`.

`manifest.json` records SHA-256 checksums for the archived files. Absolute
`/tmp/scratch-space` paths in reports refer to the original experiment workspace.
