# Assigned Gemini handoff review

Reviewed CLAUDE and the actual callers before editing. No commits made.

- 1.4 accepted: FrameworkBinding is an explicit optional input. Replaced the getattr probe with the None branch and direct field access. The missing-binding refusal remains unchanged.
- 2.2 accepted in part: build_row_exprs owns a complete new law dictionary, so law_columns now requires every EXPR_NAMES field. Legacy serialized rows remain accepted only at LawBook.row_at, the reader boundary. Existing regression covers omission of comm_dependency_rounds on that legacy wire.
- 2.3 accepted after call-graph audit: both SOL_SURFACE.measure and the dump CLI pass a resolved snapshot. measure_case now requires that argument and no longer resolves a second time or falls back to an empty runtime requirement. The CLI owns its resolution failure and reports the snapshot name there. Framework selection itself is unchanged.
- 3.3 not accepted as stated: the first test is DCP1/prefill, where verification specialization is inapplicable regardless of runtime profile. The final test is DCP8/decode, where the profile guard matters. Removed the first test's vacuous specialization assertion; retained the distinct DCP8 regression, DCP1 allocation check, and walk-key check.

CLAUDE three questions: the facts involved are optional binding presence, completeness of newly built row laws, ownership of snapshot resolution, and eligibility for runtime verification specialization. Their owners are respectively the function's typed input, build_row_exprs, the corpus/CLI resolver, and Deployment's runtime profile. Defensive fallbacks obscured these contracts by allowing downstream consumers to re-infer or fabricate them. The changes make consumers read their owner and leave compatibility normalization only at the serialized-wire boundary. No numerical law, pin, default framework, or runtime scope changed.

Validation: 11 passed, 5 deselected in 5.09s across runtime_profile, admission, sparse dependency/legacy-wire, row metadata and SOL framework tests. Ruff passed for all four modified files. Full git diff --check reported an unrelated new blank line at EOF in test_sparse_dcp_contract.py, owned by the other review task; the four assigned files are clean.

Additional fixture verification: step_sum synthetic sum/dispatch classes, batch_sweep.TestByBound, and calibration_overlay_contracts completed with **52 passed in 39.14s**. All test call sites to law_columns use build_row_exprs, so no fixture fabricates an incomplete new-law dict. The explicit legacy-wire omission remains covered by test_sparse_dcp_dependency. Log: logs/gemini-precision-row-fixtures.log.
