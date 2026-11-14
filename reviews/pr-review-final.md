# Final author review

Reviewed against CONTRIBUTING.md, CLAUDE.md and the pull-request template on
origin/main 2c89ca7. Runtime and precision reviewers independently reviewed the
source contract and the final hardware classification. Their reports are
pr-review-runtime-final.md and pr-review-precision.md in this directory.

Resolved findings include framework admission for DCP1, architecture/runtime
allocation separation in walk identity, the corpus SOL instrument's framework,
SM count as a runtime hardware fact, trace-aware scheduler memoization, exact
cache-format byte inputs for the paged indexer, and explicit calibration-row
selection. Existing measurement identities are migrated, not retroactively
marked valid. Kimi-K3's existing convolution history remains four; K3.1 uses
three. No unrelated fingerprint golden was refreshed.

Scope limitations are public: target-step costs are a lower bound; measured
envelope coverage is partial. Fused MoE front and MXFP8/MXFP4 expert timing,
DFLASH acceptance, scheduler/graph/HiCache effects and allocator page/state-slot
budgets are not a calibrated server simulator. Candidate-owner scenarios are
inputs, not observed distributions. Native scheduler fallback outside the
audited shared-memory admission is explicitly rejected. These are accuracy
boundaries, not hidden success criteria or silently guessed defaults.

The B300 timing-test correction changes only a hardware-sized test fixture.
It keeps the 1.1 threshold and production clock unchanged. Independent repeated
measurements preserve both the original 16 MiB results and the fixed 3/4 L2
results. The three-test file passes on an isolated GPU.

Validation evidence is in pr-final-test-coverage.json (8,007 passed, 199 skipped; no missing or
unresolved failed default-CI node), pr-publication-validator.json, pr-boundaries-final.log,
pr-corpus-verify-final.log, pr-k31-pin-values.log, pr-stored-specs-final.log and
pr-web-build.log. The additional opt-in DeepSeek fingerprint mismatch was
reproduced on the unchanged baseline and remains documented rather than erased.

GitHub publication requires login: gh auth status reports no authenticated host.
The local branch and PR body can be completed without a GitHub credential.
