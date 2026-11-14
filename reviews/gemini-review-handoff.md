# Gemini Code Review Handoff (`codex/kimi-k31-sglang` @ `43d0cf9`)

Target: `HuggingArch` branch `codex/kimi-k31-sglang` (commit `43d0cf9`, base `origin/main` `2c89ca7`, 184 files changed, `+19031 -217`).

**Verdict**: **Request Changes** before opening/merging PR.
Core architecture (K3.1 spec/snapshot, 0-diff checkpoint reconciliation, `RuntimeProfile` separation, `SparseWindow` closed runtime law, SGLang `precision_launches`, `sm_count` walk-class identity) is sound and well-anchored to SGLang `1b2870d33de7b5f31844fd74e97eea6d7896c53a`. However, there are **4 concrete code/defensive issues**, **3 test-redundancy/anti-pattern issues**, and **1 PR-hygiene issue** (massive raw evidence dump in `docs/evidence/`).

---

## 1. Code & Architecture Findings (Must Fix)

### 1.1 [P1] `backend/arch/layout/sparse_dcp.py:32-37` — Over-defensive `None` guards & duplicate `SparseDcpPlans.of` logic
- **Location**: `backend/arch/layout/sparse_dcp.py:34-37`
- **Issue**:
  ```python
  if runtime_profile == RuntimeProfile.ARCHITECTURE:
      return cls()
  if program is None or parallel is None:
      return cls()
  ```
  `LayoutPropagation` is always constructed with a `ParallelConfig`; silent `return cls()` on `program is None` or `parallel is None` masks broken caller invariants instead of failing fast.
- **Fix**: Remove `if program is None or parallel is None: return cls()`. Let missing `program`/`parallel` raise `AttributeError` or check `if not program.sparse_dcp_components: return cls()` directly.

### 1.2 [P1] `backend/arch/layout/physical.py:74-80` — `getattr` defensive probe in `physical_axis_partition_audit`
- **Location**: `backend/arch/layout/physical.py:74` & `backend/arch/layout/physical.py:856`
- **Issue**:
  - Line 74: `plans = getattr(sharding, "sparse_dcp", None)`
  - Line 856: `plans = getattr(analysis.rule, "sparse_dcp", None)`
  Per `CLAUDE.md §2.7`, `Rule` / `LayoutPropagation` owns the contract; `getattr(..., None)` probes hide contract drift. `LayoutPropagation.__post_init__` guarantees `self.sparse_dcp` is always a `SparseDcpPlans` instance.
- **Fix**: Access `sharding.sparse_dcp` and `analysis.rule.sparse_dcp` directly (or define `sparse_dcp = None` / `SparseDcpPlans()` on base `Rule` if non-layout rules hit this path).

### 1.3 [P2] `backend/arch/layout/flashmla_schedule.py` vs `backend/arch/runtime_laws.py` — Duplicated windowed scheduler algorithm
- **Location**: `backend/arch/layout/flashmla_schedule.py:31-69` (`windowed_schedule`) and `backend/arch/runtime_laws.py:15-67` (`windowed_counts`)
- **Issue**: Both implement the exact same `block_size=64, overhead=5` prefix-sum + binary-search + snap-vs-plain cut algorithm. `windowed_schedule` is only called by tests (`test_flashmla_schedule.py`), while production (`SparseMlaLocalStages`) uses `windowed_work` -> `SparseWindow` -> `windowed_counts`.
- **Fix**: Either have `windowed_schedule` share the core cut-computation helper with `windowed_counts`, or move `windowed_schedule` into `tests/` if it is purely a test oracle. Do not keep two copies of the same scheduler math in `backend/arch/`.

### 1.4 [P2] `backend/inference/serving.py:356-362` — `getattr(framework_binding, "framework", None)`
- **Location**: `backend/inference/serving.py:359`
- **Issue**:
  ```python
  if (required_framework is not None
          and getattr(framework_binding, "framework", None) != required_framework):
  ```
  `framework_binding` is `Optional[FrameworkBinding]`. Use `framework_binding is None or framework_binding.framework != required_framework` instead of `getattr`.

---

## 2. Over-defensive Patterns to Clean Up

1. **`backend/arch/layout/sparse_dcp_comm.py:64-67`**:
   Redundant `if len(semantic) != len(expected): raise ValueError(...)` followed by `zip(semantic, expected, strict=True)` + `require_equal`. `zip(..., strict=True)` already rejects length mismatch, or a single `require_equal` on tuple length is cleaner.
2. **`backend/inference/row_roofline.py:618, 775`**:
   `exprs.get(name, 1) if name == "comm_dependency_rounds" else exprs[name]` and `wire.get(name, sp.Integer(1)) if name == "comm_dependency_rounds" else wire[name]`.
   In `build_row_exprs`, `"comm_dependency_rounds"` is *always* emitted in the returned dict, so `law_columns` (line 618) never needs `.get(name, 1)`. Keep the legacy-wire fallback *only* in `LawBook.row_at` (line 775) if old serialized JSONs lack the key, or remove if `data_rules` / `fingerprint` version bumps invalidate old rows anyway.
3. **`scripts/dump_sol_surface.py:263-266`**:
   `if resolved is None: ... resolved = get_default_loader().match(config)` + `runtime_inputs = runtime_framework_inputs(resolved) if resolved is not None else {}`. Keep it simple: `measure_case` always has `config` and `resolved` when called from the corpus runner.

---

## 3. Redundant / Low-Value Tests to Prune or Consolidate

Per user instruction ("禁止写冗余的测试") and `CLAUDE.md §5.5`:

1. **`tests/backend/arch/layout/test_sparse_dcp_contract.py`**:
   - Overlaps heavily with `tests/backend/arch/layout/test_kimi_k31_sparse_dcp.py` and `tests/backend/arch/layout/test_sparse_dcp_comm.py`.
   - `test_local_mla_runtime_keeps_partials_and_combine_as_distinct_work` (lines 74-94) and `test_fixed_verification_block_counts_each_causal_length` (lines 67-71) re-test the exact same `SparseMlaLocalStages` and `sparse_dcp_query_pairs` formulas already asserted end-to-end in `test_kimi_k31_sparse_dcp.py` (`test_public_runtime_scheduler_reprices_context_and_owner_scenario`, `test_verify_pairs_follow_each_causal_query_not_last_query_rectangle`). Keep unit tests only for edge/error cases not covered end-to-end (e.g. `SparseDcpWorkload` conservation/trace errors, `OutputExchange` 128-wide push guard), and drop the redundant happy-path arithmetic echoes.
2. **`tests/backend/arch/layout/test_flashmla_schedule.py`**:
   - `test_uniform_work_matches_independent_source_schedule` (lines 20-33) is a strict subset of `test_periodic_runtime_counts_match_independent_oracle` (lines 68-75). Remove `test_uniform_work_matches_independent_source_schedule`.
3. **`tests/backend/inference/test_runtime_profile.py`**:
   - `test_architecture_dcp_does_not_specialize_sparse_verification` (lines 33-39) is already asserted in `test_architecture_profile_has_no_sglang_allocations` (`assert not deployment.specializes_verification`). Merge or remove the duplicate.
4. **`tests/backend/inference/test_sparse_dcp_assumptions.py`**:
   - 16-line file testing `deployment_assumptions(...)` dict values directly. This is a wiring/echo test of a static dict builder. Either fold one assertion into `test_kimi_k31_sparse_dcp.py` on the real `estimate_capacity` / `deployment_assumptions` output or delete.

---

## 4. PR Hygiene: `docs/evidence/` Bloat

- `docs/evidence/` adds **~11,500+ lines** across ~55 files, including raw calibration job configs (`gemm/jobs.json`, `indexer/jobs.json`, `train.csv`, `holdout.csv`, `train-env.json`, `holdout-env.json`), `final-gpu-processes.txt` (0 bytes), `final-tests.log` (2 lines), `gpu7-release.txt` (1 line), and `kimi-k31-dcp-continuous-trace-spans.json` (4,746 lines).
- **Recommendation**: Keep the compact summary JSONs (`kimi-k31-validation.json`, `kimi-k31-b300-runs.json`, `kimi-k31-dcp-comparison.json`, `calibration_rows.json`, `kimi-k31-final-test-coverage.json`) and `docs/*.md`, and move/drop the raw per-job calibration scratch files (`docs/evidence/kimi-k31-b300-calibration/{gemm,indexer}/`, empty `.txt` files, and 4,746-line trace dump) to `/tmp/scratch-space/k3-performance/benchmark/` (where they already live).

---

## 5. Checklist Before Opening PR

1. Apply fixes in §1.1–§1.4 and §2.
2. Prune redundant tests in §3.
3. Trim `docs/evidence/` raw scratch files in §4.
4. Re-run targeted pytest on modified files (`tests/backend/arch/layout/`, `tests/backend/inference/test_runtime_profile.py`, `tests/meta/test_eval_boundary.py`) + `uv run ruff check backend/`.
5. Refresh `kimi-k31-huggingarch.patch` if needed, then `gh auth login` and open PR using `PR_BODY.md`.
