import datetime,hashlib,json,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree'
final=json.loads((R/'whole-comparison.json').read_text());interim=json.loads((R/'runtime-comparison.json').read_text())
assert len(final)==2 and all(x['span_calls']==92 and x['fully_measured_s'] is None for x in final)
assert all(abs(x['error_pct'])<8 for x in final)
for name in ['final-runtime-tests.log','final-integration-tests.log','roofline-comm-regression-tests.log']:
    contents=(R/name).read_text();assert 'passed' in contents and 'FAILED' not in contents,name
checks=[];timing=[]
for batch in [1,8]:
    rows=[x for rank in range(8) for x in json.loads((R/f'whole-moe-rank{rank}.json').read_text()) if x['batch']==batch]
    assert len(rows)==8
    checks.extend(c for row in rows for c in row['checks'])
    parallel=max(x['variants']['two_streams']['p50_us'] for x in rows)
    serial=max(x['variants']['serial_control']['p50_us'] for x in rows)
    timing.append(dict(batch=batch,two_streams_us=parallel,serial_us=serial,saved_pct=100*(serial-parallel)/serial))
assert len(checks)==32 and all(c['passed'] for c in checks)
source_files=['backend/arch/eval/fusion.py','backend/arch/eval/launch.py',
    'backend/inference/capacity.py','backend/inference/framework_bindings/sglang.yaml',
    'backend/inference/op_signatures_catalog.py','backend/inference/phases.py',
    'backend/inference/row_roofline.py','backend/inference/serving.py','kernel_bench/plan.py',
    'scripts/estimate_kimi_k31_sglang.py',
    'backend/inference/fusion_plans/kimi_k31_trtllm_latent_span.yaml',
    'backend/inference/fusion_plans/kimi_k31_trtllm_moe_span.yaml',
    'tests/backend/arch/eval/test_moe_execution_span.py','tests/backend/inference/test_k31_runtime_span.py',
    'tests/backend/inference/test_fusion_registry.py','tests/kernel_bench/test_plan_points.py']
provenance={}
for name in source_files:
    src=REPO/name;dst=R/'source-snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst);provenance[name]=hashlib.sha256(src.read_bytes()).hexdigest()
sglang_files=['srt/models/kimi_k3.py','srt/layers/moe/moe_runner/flashinfer_trtllm.py',
    'kernels/ops/moe/kimi_k3_fused_routing.py','kernels/ops/communication/all_reduce_residual.py',
    'kernels/ops/communication/gemm_ag.py','kernels/ops/gemm/cutedsl_bf16_gemm.py',
    'srt/layers/communication/k3_ar_fusion.py']
source_compare=[]
for name in sglang_files:
    old=W/'sglang/python/sglang'/name;current=W/'benchmark/agentx-yyh/sglang-recipe/python/sglang'/name
    source_compare.append(dict(path=name,historical_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        current_sha256=hashlib.sha256(current.read_bytes()).hexdigest(),identical=old.read_bytes()==current.read_bytes()))
assert all(x['identical'] for x in source_compare)
release='https://github.com/StarDuster/k3-performance-results/releases/tag/b300-runtime-calibration-20261115'
summary=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    huggingarch_base=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
    huggingarch_source_sha256=provenance,sglang_moe_source_comparison=source_compare,
    source_revision='5792cf6e5e11aebc62d41cf19828f615a8fa7e0a',historical_source_revision='1b2870d33de7b5f31844fd74e97eea6d7896c53a',
    final_comparison=final,intermediate_latent_only_comparison=interim,
    full_moe_timing=timing,numerical_checks=32,numerical_typical_max=max(c['typical'] for c in checks),
    numerical_worst_max=max(c['worst'] for c in checks),original_nine_rows_modified=False,
    fitted_global_latency_factor=False,pr_created=False,private_results_release=release,
    limitations=['Historical B1 target-only serving baseline; no fresh AgentX test',
        'Batch8 validated at execution-span level, not end-to-end',
        'Synthetic weights/routes with fixed audited shapes; route skew and cache regime remain workload inputs',
        'About21% of final predicted time remains theoretical; fully measured latency is null'])
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# B300 K3.1 runtime calibration repair','',
    'Final scope: target-only Kimi-K3.1-Pro, B300 TP8/DCP8/EP2, KV64, batch1, output512. No speculation or AgentX result is implied. The original nine rows on kimi-k31-reports are unchanged. No PR.', '',
    '## Final comparison','',
    '| Input | Predicted ms/token | Historical measured ms/token | Error | Predicted time using measurements |',
    '|---:|---:|---:|---:|---:|']
for x in sorted(final,key=lambda x:x['input_tokens']):
    lines.append(f"| {x['input_tokens']} | {x['partial_ms']:.6f} | {x['baseline_ms']:.6f} | {x['error_pct']:+.2f}% | {x['measured_share']:.2%} |")
lines+=['','These are partial predictions compared against the existing unprofiled serving baseline, not new end-to-end runs. No aggregate multiplier, serving-latency offset, or latency-threshold row selection was fitted. The uncalibrated remainder is still theoretical and fully measured latency remains null.', '',
    '## What changed','',
    '- Propagated the actual TP degree into both planner and estimator dispatch. Previously TP-conditioned bindings were silently disabled.',
    '- Added source-anchored latent-path and complete-MoE execution spans. The complete span includes merged dense front, prepared routing and MXFP8 quantization, TRTLLM SiTU expert GEMMs, shared-expert side stream, join, deferred finalize/AR/norm, and GEMV/all-gather/add. It is a multi-kernel span, not a claim of one CUDA kernel.',
    '- Allowed a span that owns the full dispatch/expert/combine region to consume original tokens. Partial boundary crossings still refuse. Whole spans do not inherit internal expert-row padding or dispatch membership.',
    '- Preserved mathematical operands when a computation span contains communication; primitive wire shapes no longer replace its input signature.',
    '- Retained both same-group collective payloads and their independent fixed floors and dispatch rungs. A payload aggregate is used for byte totals, never as a fabricated single launch.',
    '- Replaced the entire measured MoE block once per layer. Its routing, quantization, shared activation, Norm and collective measurements are not charged again.', '',
    '## Direct concurrency measurement','',
    '| Batch | Source two-stream execution µs | Serial control µs | Time saved |',
    '|---:|---:|---:|---:|']
for x in timing:lines.append(f"| {x['batch']} | {x['two_streams_us']:.3f} | {x['serial_us']:.3f} | {x['saved_pct']:.2f}% |")
lines+=['','Both controls use the same kernels, inputs and weights; the serial control removes the shared side stream. The source fork/join and low-SM pull collective are reproduced, followed by deferred finalize/push/norm and GEMV/all-gather/add. The logical MoE boundary uses prefix_sum=None; the outer decoder residual remains outside this measurement.', '',
    'Every timing uses all eight ranks, maximum rank median, thirty graph samples, sixteen distinct routes and four dense-weight sets. Rotation avoids repeating one resident expert or dense matrix set in L2. The complete expert bank is present. No L2-drain subtraction is inserted between serving kernels. Cache/weight residency is a documented regime, not a universal device constant.', '',
    '## Numerical and regression validation','',
    'Complete MoE numerical validation: batch1/8 × seeds17/43 × all8 ranks =32 successful checks. The independent reference uses FP32 dense GEMMs, explicit E2M1 decode, MXFP8 group32 rounding, SiTU(beta4, linear_beta25), weighted expert outputs, CPU/Gloo sums and RMSNorm. The existing FP8 error band was unchanged. Maximum normalized RMS error is '+f"{summary['numerical_typical_max']:.6f}, maximum normalized element error {summary['numerical_worst_max']:.6f}.", '',
    'The narrower distributed path also passed32 checks and the actual standalone TRTLLM API passed four checks. Final logs: final-runtime-tests.log (87 passed,140 subtests), final-integration-tests.log (6 passed), roofline-comm-regression-tests.log (6 selected passed). Earlier broader structural checks passed163 tests plus47 subtests. Do not sum these overlapping suites as distinct tests.', '',
    '## Intermediate result and cancellation','',
    'The latent-only correction predicted10.330ms at16K (+2.47%) and10.311ms at2K (+3.68%). It still added the shared branch serially. The measured complete block is44.856µs, whereas that intermediate model billed55.401µs per MoE block. Its smaller aggregate error therefore included cancellation. The final report uses the complete two-stream measurement even though the final absolute error is larger. Both intermediate artifacts remain for audit.', '',
    '## Reproduction and artifacts','',
    'Use whole-moe-overlay.json with the code on codex/b300-calibration. The native planner generates the keys; no operator key is invented. Run prepare_runtime.py --whole, the eight-rank whole_moe_tp8.py with the pinned SGLang PYTHONPATH, export_runtime.py --whole, then estimate_runtime.py --whole. The narrower diagnostic uses --span-scope latent in the estimator and an equivalent narrowed planner binding.', '',
    'Raw per-rank samples, numerical verdicts, serial controls, scripts, CSVs, exact native job keys and source snapshots are included. Critical MoE source files compared between historical SGLang1b2870d3 and recipe5792cf6e are byte-identical; hashes are in SUMMARY.json. The GPU stack is PyTorch2.14.1+cu130 / FlashInfer0.7.4, matching the recorded earlier benchmark.', '',
    f'All raw prediction outputs, including per-op symbolic laws, are archived in the [private results release]({release}). The source branch keeps their compact summaries and hardware measurements; large expanded prediction JSONs are omitted from the source checkout.', '',
    'Remaining limits: only two historical B1 contexts have end-to-end comparison; B8 is a numerical/timing span check. No claim is made for AgentX, DFLASH, other EP/TP layouts or arbitrary routing skew. About21% of final predicted time remains theoretical.']
report='\n'.join(lines)+'\n';(R/'REPORT.md').write_text(report)
(R/'README.md').write_text('# B300 runtime calibration repair\n\nSee [REPORT.md](REPORT.md) and SUMMARY.json. The final dataset is whole-moe-overlay.json. Raw rank samples and independent numerical checks are retained. No PR.\n')
prior=W/'benchmark/b300-calibration-followup/REPORT.md'
notice='# Superseded by runtime-aware calibration\n\nSee [the newer report](../b300-runtime-calibration/REPORT.md) for corrected TP dispatch, complete TRTLLM MoE spans and source two-stream timing. The results below are historical.\n\n'
if not prior.read_text().startswith('# Superseded'):
    prior.write_text(notice+prior.read_text())
(R/'release-notes.md').write_text('B300 K3.1 runtime calibration repair: native TP8/EP2 two-stream MoE block, all-rank FP32/Gloo validation, complete samples and prediction outputs. Final B1 errors:2K -6.07%,16K -7.15%. No PR.\n')
print(json.dumps({'final':final,'full_moe_timing':timing,'source_files':len(source_files)},indent=2))
