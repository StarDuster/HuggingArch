# Review resolution

Reviewed against `gemini-review-handoff.md` on commit `43d0cf9`.

| Finding | Resolution |
|---|---|
| 1.1 | Removed silent missing-input fallback in `SparseDcpPlans.of`. Actual isolated-op callers permit an unbound Program; `LayoutPropagation.__post_init__` now owns that explicit empty-plan state. |
| 1.2 | Removed both getattr probes. Layout rules read their plan contract directly; identity rules use the existing `is_trivial` distinction. No reverse import into base Rule. |
| 1.3 | Moved the independent per-row scheduler oracle into tests. Production retains one bounded runtime-law implementation. |
| 1.4 | Replaced optional binding getattr with an explicit None condition and direct field access. |
| 2.1 | Removed the redundant length precheck; strict zip preserves mismatch rejection. |
| 2.2 | Newly built law dictionaries require all fields. Compatibility fallback remains only at the legacy serialized-wire reader. |
| 2.3 | `measure_case` requires the caller-resolved model; CLI resolution failure is reported at that boundary. Both real callers already supply it. |
| 3.1 | Removed the two happy-path formula echoes; kept conservation, trace, output-width and symbolic identity coverage. End-to-end scheduler/causal-query tests remain. |
| 3.2 | Removed the uniform subset test; independent periodic oracle comparison remains. Native partition boundary checks now exercise production stages. |
| 3.3 | Corrected the review premise: DCP1/prefill and DCP8/decode differ. Removed the vacuous DCP1 assertion and retained the effective DCP8 verification-specialization check. |
| 3.4 | Deleted the static assumptions-dictionary echo test. |
| 4 | Archived 16 raw files outside Git, preserving and checking SHA-256 against originals. Kept summaries, calibration rows and reproduction scripts; documented input regeneration. Evidence directory shrank about 50%; 45 relative links checked. |

Raw archive: `/tmp/scratch-space/k3-performance/benchmark/pr-review-raw-evidence/`.
Original reviewer handoff is preserved unmodified and untracked in the checkout.

Validation logs: `logs/gemini-review-validation.log`, `logs/gemini-precision-targeted.log`, `logs/gemini-precision-row-fixtures.log`.

Final validation: 469 passed + 98 subtests; 11 contract checks and 52 row-fixture checks passed. Backend Ruff and diff checks passed. Full-CI evidence is explicitly labeled as the prior 43d0cf9 run.
