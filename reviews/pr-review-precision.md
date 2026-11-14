# Kimi-K3.1 PR static review — precision, cache and runtime ownership

Reviewer: precision_sglang. Scope: working-tree diff against origin/main, including untracked runtime/layout/schema tests and providers. This is a moving working tree, not a review of a frozen commit. Reviewed CONTRIBUTING.md and CLAUDE.md. No GPU use and no large test suite rerun. One requested targeted regression test was run after fixing the finding below.

## P2, fixed: DCP1 bypassed the SGLang-only serving admission

Location: `backend/inference/serving.py:356`; producer: `backend/arch/layout/sparse_dcp.py:72`; consumer: `backend/arch/layout/runtime_weights.py:73`.

The original guard tested `sparse_dcp.plans`. That tuple contains only DCP > 1 execution plans, while DCP1 still binds `storage_plans` and allocates SGLang's separate absorbed Wkc/Wvc buffers. As a result, K3.1 DCP1 under a reference/vLLM serving binding could publish SGLang-specific resident memory without a refusal. This contradicted both the SGLang-only support scope and the DCP8 behavior.

CLAUDE three questions:

1. **Which fact was wrong, and who owns it?** Whether a serving estimate depends on the SGLang runtime contract. The bound runtime profile owns this fact; it remains true when only its load-time allocations are active.
2. **Why did the structure permit it?** Admission used the execution-only subset as a proxy for runtime-profile presence. The split between execution plans and storage plans made that proxy false at DCP1.
3. **What fixes ownership/dataflow?** Ask the existing bound profile's `storage_plans`, which contains every relevant binding, rather than re-derive eligibility from DCP degree. No new model-name condition is needed.

Parent authorized the minimal correction during review. The guard now reads `storage_plans`. The targeted test covers DCP1 and DCP8 against None/reference/vLLM. Command: `.venv313/bin/python -m pytest -q tests/backend/arch/layout/test_runtime_weights.py -k refuses_non_sglang_serving`; result: **1 passed, 25 deselected** (repository parameter folding). No other implementation was changed during this review.

## Audited consistency checks

- **Mixed checkpoint precision:** packed_unpack_factor now requires the variant's actual storage dtype as well as its suffix; BF16 `.weight` tensors no longer inherit the U8 packed factor. U8 plain packed weights remain packed. K3 metadata changes only the scheme identifier. The correction is generic; no model-ID branch was introduced.
- **Activation precision:** the SGLang MXFP4 launch selector is architecture- and CUDA-capability-gated, and both runner and precision flags must match. The BF16 override is a separate alternative. Program predicates are included in launch selection and unavailable-candidate reporting. Checkpoint activation metadata is not rewritten to imitate the runtime. The generic selector rejects launch-local predicates it cannot resolve here.
- **Source provenance:** newer binding evidence uses full per-location commit URLs without falsely moving the binding-wide v0.5.22 audit. Explicit SGLang snapshots stage only their frozen Python files for AST analysis and retain native evidence in the snapshot; the old reference/vLLM fallback remains for other source kinds.
- **Resident vs read bytes:** runtime allocation surfaces are separate from saved checkpoint parameters. Full Q uses an alias delta, absorbed K/V and preproject-V are separate allocations, and DCP1 retains its local absorbed copies. Decode aliases consume the full-head allocated operand; prefill remains local-head. BF16 preprojection is explicitly refused for F16; this does not change FP8 cache formats.
- **Cache storage vs invocation:** `storage_law` is the single padded striped storage producer. Capacity groups, formulas and totals consume it. Invocation read overrides are separate: decode index scans local context, selected MLA reads depend on selected ownership, and prefill reads the gathered full context. Persistent storage does not shrink to top-k. Default components have no position-padding flag, preserving their previous laws.
- **Verification:** fixed width is part of the sparse-DCP cell identity; causal pair sums use each query's context, including short contexts and striping/top-k boundaries. Context and batch remain symbolic. Indexer key-buffer traffic can be reused across queries while useful arithmetic follows the causal pair sum. Width admission prevents reusing one compiled sparse verification law at another width.
- **Prefill:** cache-write ownership permits a partial last striped owner. The two additional gathers derive record widths from cache formats (132-byte index and 656-byte MLA in the recipe) and pay padded equal-rank communication without padding useful attention FLOPs. Their count is obtained through the materializer's stage-aware `_count`, rather than a separate hard-coded layer count.
- **Hardware class:** `sm_count` is classified as a dispatch/walk fact and enters ServingDeployment/CardClass, so equal compute capability does not imply identical scheduler walks. Only the observed B300 value is filled; unknown devices are not assigned an inferred SM count. The CSV's previously comma-bearing notes are properly quoted in the edited rows.
- **Other models:** old K3 retains its original full convolution history through the new explicit `s_conv_history` binding. DCP overrides require a declared sparse component. Generic traffic defaults and dependency rounds retain the old behavior when the new metadata is absent.
- **Kernel providers:** the added absorbed-Q/V providers preserve serving's noncontiguous activation strides, contiguous transposed weights and BF16 output. The paged indexer provider explicitly constrains phase/query width/head count/page layout; packed-byte trace acceptance is scoped to the declared paged cache shape rather than accepting arbitrary uint8 buffers.

## Completion checks and explicit limitations

Follow-up: dcp_support has wired scheduler and split-combine runtime stages into sparse_dcp_comm. The compact generated SparseWindow law is registered centrally and unknown SM count is labeled `unresolved_sm_count; known_traffic_floor_only` rather than assuming 148. Producer write overrides distinguish FP32 split buffers from BF16 direct output. Verified the partition-width boundary with the source owner: dsa_backend.py:3784 flattens verification queries via q_all.view(-1,1,tp_q_head_num,head_dim), so the scheduler sees B*n_verify rows and query width 1; using SM-count partitions without another division by n_verify is correct for this wrapper. Public smoke/gates are owned by the parent.

Unsupported DCP geometry is explicitly refused. Arbitrary server options are rejected by FrameworkBinding.deployed; therefore an unrecognized override cannot silently change the fixed reference profile. The default fi_a2a/push1536/replicated-Q/preprojected-V behavior is a labeled reference profile, not a generic simulation of all SGLang flags. DCP1's short eager dense-prefill shortcut is outside the absorbed profile and documented with the threshold-zero way to force the modeled path. These scope statements should stay visible in final results/docs.

Process-wide workspaces, graph pools and recurrent state-slot multiplicity are not implied by raw resident weights/cache totals. This PR does not add an allocator/state-slot control model: capacity remains an explicit raw architectural upper bound, not the actual SGLang request-pool limit. Recompute checked-in example JSON/numbers after final resident/capacity/scheduler changes so stale evidence is not published as the final result.

Contribution gates still belong to the final author run: full pytest, backend Ruff, service/eval boundaries, stored-spec reload for the schema addition, validator valid/diff, and corpus admission/pins. This report does not substitute for those gates. No additional confirmed P1/P2 was found in the inspected final-state precision/cache/weight dataflow beyond the fixed admission issue.

## Follow-up: architecture admission and merged-front limitation

Publication replay exposed that simply strengthening the SGLang guard also rejected architecture-only THEORY walks. The underlying fact was deeper: declared runtime allocations had no deployment input separating them from the architectural graph. Added RuntimeProfile.ARCHITECTURE / DECLARED at Deployment and LayoutPropagation, and included it in StepCells/Cell/WalkCellKey. THEORY capacity selects ARCHITECTURE; other frameworks retain DECLARED and the admission check. Architecture mode has no SGLang storage or execution plans and no sparse verification specialization. This is a normal walk input, not a validator bypass. Targeted runtime-profile plus admission checks: **4 passed, 25 deselected**; changed-file Ruff passed. Parent owns the publication replay.

Confirmed timing-fidelity limitation: the actual CUDA Kimi merged MoE front emits FP32 and spans shared gate/up + router + latent-down. Current sibling-only generic fusion leaves those separate. Full width is 6144 at K3.1 TP8, while the default 18..256-row sharded-front branch changes the GEMM and adds a gather. Existing symbolic-cell conditions do not safely express C1-only structural specialization, and parallel fusion output dtype needs a corresponding declaration. No default recipe flag was changed to hide this boundary. The source, consequences, and prohibition on matching a full-front measurement to a single router row are documented in docs/kimi-k31-runtime-precision.md. Thus the PR must not claim a complete measured launch-stream simulation.

## Final cache-page audit (read-only)

SGLang's DCP allocator distinguishes physical and virtual pages. At audited main `1b2870d33de7b5f31844fd74e97eea6d7896c53a`, `python/sglang/srt/mem_cache/allocation_sizing.py:33-36` returns `schedule.page_size * attn_dcp_size`; `mem_cache/kv/alloc/band.py:421-434` explicitly keeps `pool_page_size` physical and widens the public allocator's `page_size` by DCP. Thus page64/DCP8 means 512 logical tokens per virtual page and 64 physical rows on each rank. A request occupying S logical positions reserves at least `64 * ceil(S / 512)` local rows in this page model.

The current HuggingArch path does **not** apply the incorrect `ceil(S / 64) * 64 / 8` law. It applies no page allocation rounding at all: `CachedTensor.storage_law` rounds only striped ownership (`ceil(S / 8)` local rows), while `capacity_ceilings` asks `reader.storage_bytes_at(request_tokens)`. `kv_block_size` is carried for kernel/phase measurement identity, not consumed by that storage law. Hence capacity is the raw persistent-tensor upper bound already documented as before page rounding; it is not a prediction of SGLang's page-pool capacity.

For example S=513 occupies 65 raw local rows but 128 allocator rows, a difference of 1,191,456 bytes/rank across 24 DSA layers at 788 bytes/row/layer. S=17 occupies 3 raw rows but 64 allocator rows. KDA fixed state remains separate. No implementation was changed by this audit. If page-pool capacity is added later, physical page size must become a cache-geometry/walk-key input and participate in the single storage-law producer; adding a consumer-side DCP multiplier would leave formulas, totals and breakdown inconsistent.

## P2, fixed: SOL corpus instrument used the default vLLM runtime

`scripts/dump_sol_surface.py:244` previously omitted the framework argument, so its K3.1 SOL pin was measured under the capacity API's default vLLM binding. Correct SGLang-only admission exposed this as a refusal; removing the refusal would silently restore the wrong experiment.

Three questions: (1) The wrong fact was the serving runtime under test; it belongs to the declared runtime backend, not the corpus model name. (2) The instrument assumed the global default applied to every snapshot, because the declaration did not publish its owning framework. (3) `SparseDcpSpec.framework` now states that ownership and `Program.required_runtime_framework` aggregates active declarations; the SOL instrument forwards that requirement, preserving the previous default for programs without one. It neither switches the corpus to THEORY nor bypasses admission.

Targeted tests: 3 passed; changed-file Ruff passed. Actual `SOL_SURFACE.measure` for K3.1 completed for both single and eight deployments under SGLang. Results: `logs/pr-k31-sol-sglang-values.json`; execution log: `logs/pr-k31-sol-sglang-measure.log`. Golden updates remain with the parent so the change is reviewed as a corrected runtime measurement, not an unexplained pin refresh.

## Final gate-contract corrections

Serving admission now reads `Program.required_runtime_framework`, the same backend-owned fact used by the SOL instrument, while requiring bound storage plans before enforcing it. No second hard-coded framework choice remains in that admission.

Pure architectural module bench entry points (`serving_step`, `phase_roofline`, `bench_component`, and isolated dimension walks) explicitly select RuntimeProfile.ARCHITECTURE. Their shipped-framework union fusions do not imply permission to bind one framework's allocation contract. The declared serving guard remains active.

Updated synthetic cache fixtures to expose the real `storage_law` contract, the walk-key field test to include runtime_profile, and the concurrent StepCells constructor mock to forward the profile. `cache.storage_law` now uses the existing `algebra` expression reader rather than raw sympify. The authored-expression-reader gate passed. Changed-file Ruff and diff checks passed; combined targeted results are recorded in `logs/pr-precision-final-gates.log`.

Final combined targeted run completed: **41 passed, 11 deselected in 317.63s**, including the complete views/test_diff.py file and all assigned fixture/key/concurrency/admission/runtime-profile regressions. The separate authored-expression-reader gate also passed. Implementation is frozen pending parent-owned final gates.
