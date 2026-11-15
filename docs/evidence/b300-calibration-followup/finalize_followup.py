"""Document the measured scope and unresolved model error without fitting it."""
from pathlib import Path
import datetime
import hashlib
import json
import statistics
import subprocess
import collections

R = Path(__file__).resolve().parent
W = R.parents[1]
REPO = W / 'calibration-worktree'

def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

comparison = json.loads((R / 'controlled-comparison.json').read_text())
assert len(comparison) == 2
for row in comparison:
    assert row['fully_measured_s'] is None
    assert row['overlay_rows'] == 61
    detail = json.loads((R / 'estimates' / ('16384-controlled-' + row['regime'] + '.json')).read_text())
    by_op = collections.defaultdict(lambda: dict(aggregated_rows=0, calls_per_step=0, measured_calls_per_step=0, partial_ms=0.0))
    for op in detail['per_op_timing_decode']:
        entry = by_op[op['op']]
        entry['aggregated_rows'] += 1
        entry['calls_per_step'] += op['count_per_step']
        if op.get('t_measured_s') is not None:
            entry['measured_calls_per_step'] += op['count_per_step']
        entry['partial_ms'] += 1000 * (op['t_measured_s'] if op.get('t_measured_s') is not None else op['t_sol_s'])
    assert by_op['kimi_attn_res_rmsnorm']['calls_per_step'] == 184
    assert by_op['attn_res_combine']['calls_per_step'] == 2
    row['by_op'] = dict(by_op)
(R / 'controlled-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
assert 'passed' in (R / 'weighted-reference-tests.log').read_text()
tests = (R / 'fusion-followup-tests.log').read_text()
assert 'passed' in tests and 'FAILED' not in tests
all_rows = json.loads((R / 'calibration_rows.json').read_text())
assert len(all_rows) == 146
assert not any(r['op'] == 'attn_res_combine' for r in all_rows)
trace = json.loads((R / 'attnres-trace-comparison.json').read_text())
assert len(trace['checks']) == 14 and all(c['kernel_count'] == 186 for c in trace['checks'])
assert len(json.loads((R / 'tp8-communication/calibration_rows.json').read_text())) == 4

kernel = 'python/sglang/kernels/ops/attention/attn_res.py'
source = W / 'benchmark/agentx-yyh/sglang-recipe'
historical = W / 'sglang'
provenance = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  huggingarch_base=git(REPO, 'rev-parse', 'HEAD'),
                  huggingarch_working_diff_sha256=hashlib.sha256(subprocess.check_output(['git', '-C', str(REPO), 'diff', 'HEAD', '--', 'backend', 'kernel_bench', 'tests'])).hexdigest(),
                  sglang_current=git(source, 'rev-parse', 'HEAD'),
                  sglang_historical=git(historical, 'rev-parse', 'HEAD'),
                  attnres_kernel_current_sha256=sha(source / kernel),
                  attnres_kernel_historical_sha256=sha(historical / kernel))
provenance['source_files_sha256'] = {str(path.relative_to(REPO)): sha(path)
    for pattern in ['backend/inference/framework_bindings/sglang.yaml',
                    'backend/inference/fusion_plans/kimi_attn_res_rmsnorm.yaml',
                    'backend/inference/op_signatures_catalog.py', 'kernel_bench/refs.py',
                    'kernel_bench/providers/norm_act.py', 'kernel_bench/providers/gemm.py']
    for path in [REPO / pattern]}
(R / 'source-followup-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

lines = ['# B300 calibration: validated measurements and remaining error', '',
         'The calibration dataset and fusion fixes are saved on `codex/b300-calibration` in the user fork. No PR is opened. The original nine rows on `kimi-k31-reports` are unchanged. No AgentX rerun was started after cancellation.', '',
         'This pass adds four TP8 multicast shapes, sixteen learned-weight AttnRes-plus-RMSNorm shapes measured in two explicit timing regimes, and four public CUTLASS group-32 MXFP8×MXFP4 shapes. All forty new measurements passed numerical checks. These are forty rows, not forty distinct physical keys: the sixteen AttnRes keys have separate cold and warm measurements.', '',
         '| Measurement | Rows | Timing / usable scope |',
         '|---|---:|---|',
         '| SGLang TP8 multicast | 4 | Warm graph128; maximum rank median; residual=None, no fused norm |',
         '| SGLang weighted AttnRes + RMSNorm | 16 | Native isolated L2-drain |',
         '| Same AttnRes kernel | 16 | Warm graph64; separate implementation suffix |',
         '| FlashInfer CUTLASS MXFP8×MXFP4, group32 | 4 | Cross-library alternative; one logical row per expert padded to four |', '',
         'TP8 measured medians are 3.214 / 3.326 / 3.551 / 4.163 µs for 3584 / 7168 / 28672 / 57344 BF16 elements. The distributed checks use all eight ranks and an independent Gloo FP32 oracle, with seeds 17 and 43.', '',
         '## Residual reference correction', '',
         'The old unfused AttnRes signature and reference omitted the learned score gain/projection tensors already present in the model graph. Its sixteen old Torch measurements validated the wrong function and are retired into `legacy-reference-quarantine/`; they are excluded from the live corpus and both controlled overlays. The corrected unfused reference reads both weights. A separate SGLang fusion contract preserves both score inputs and includes the following RMSNorm. The provider implements the source BF16 cached score-weight product and matches an independent FP32 expression.', '',
         'For batch1, the new kernel measures 8.736–10.752 µs with isolated L2-drain and 2.235–3.738 µs in graph64 steady state. Historical serving CUPTI medians at context16385 are 5.264–7.024 µs across history depths1–8. All fourteen sampled decode steps have186 matching kernel calls. History depth is inferred from forward order, and CUPTI event durations differ from standalone CUDA-event timing; this diagnostic is not imported as calibration. The historical and current attn_res.py files have identical SHA256, although the overall SGLang revisions differ.', '',
         '## Controlled 16K comparison', '',
         'Each overlay keeps the original41 library-backed rows, adds four TP8 rows, then adds exactly one set of sixteen AttnRes rows. No latency threshold chooses rows. The four CUTLASS alternative GEMMs are excluded because the serving recipe selects TRTLLM fused MoE. Other kernels retain their original measurement regimes, so these are partial mixed-clock diagnostics.', '',
         '| AttnRes timing | Partial prediction ms | Historical TPOT ms | Error | Measured time share |',
         '|---|---:|---:|---:|---:|']
for row in comparison:
    lines.append(f"| {row['regime']} | {row['partial_ms']:.4f} | {row['historical_tpot_ms']:.4f} | {row['error_pct']:+.2f}% | {row['partial_measured_share']:.2%} |")
lines += ['', 'The 5–8% error target has not been validated. Adding standalone timings does not by itself reconstruct serving latency. Remaining gaps include the selected full TRTLLM MoE implementation, two unfused AttnRes boundary calls, and serving PDL/multi-stream overlap and cache conditions. The model still combines theoretical and measured costs; fully measured latency remains null. No aggregate fitted multiplier is applied.', '',
          '## Artifacts and checks', '',
          '`calibration_rows.json` contains146 validated rows including eager references and alternate timing regimes. It is an audit corpus, not a recommended universal overlay. The four distributed TP8 rows remain in `tp8-communication/calibration_rows.json`. Use the explicitly named `controlled-*-calibration_rows.json` files only with the scope above. `alternative_group32_calibration_rows.json` stores the alternate grouped GEMM rows separately.', '',
          'The native GPU cross-checks passed for every new measurement. Regression output is in `weighted-reference-tests.log` and `fusion-followup-tests.log`. Raw CSVs, graph samples, rank-local communication samples, source provenance, scripts and a SHA256 manifest are included. The historical serving comparison is target-only B1, TP8/DCP8/EP2, KV64, speculation disabled; it is not an AgentX measurement.', '',
          'Earlier broad-overlay reports are retained for history and are superseded by this report. Their retired AttnRes rows are invalid and must not be reused. Existing library-only41-row estimates preceded the new fusion contract and are not a controlled comparison with this pass.']
report = '\n'.join(lines) + '\n'
for filename in ['FOLLOWUP.md', 'REPORT.md', 'CALIBRATION_AUDIT.md']:
    (R / filename).write_text(report)
(R / 'README.md').write_text('# B300 calibration follow-up\n\nSee [REPORT.md](REPORT.md) for current scope, controlled estimates and remaining error.\n\nKimi-K3.1-Pro target-only decode, TP8/DCP8/EP2, batch1/8, 2K/16K inputs, output512, KV64. The current native plan has114 physical keys, of which five match the original nine-row dataset and109 are new. Unsupported keys remain explicitly uncovered. The latest AgentX comparison was cancelled by the user before calibration; no newer-version AgentX result is claimed.\n\nThe shared HuggingArch checkout is untouched. Code and evidence are stored in the separate codex/b300-calibration worktree/branch, with no PR. Legacy queue/controller scripts describe their original runs; they are not a daemon or an active monitor.\n')
summary = json.loads((R / 'summary.json').read_text())
summary.update(distributed_tp8_rows=4, audit_corpus_rows_including_distributed=150,
               retired_invalid_attnres_rows=16, controlled_overlay_rows=61,
               accuracy_target_validated=False,
               coverage_limit='Partial library overlay; selected TRTLLM full MoE, two AttnRes boundaries and serving cache/PDL overlap remain unresolved. Alternative backend and timing regimes are separated.')
(R / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
status = json.loads((R / 'status.json').read_text())
status.update(state='measurements-validated-accuracy-unresolved', step='save-branch', accuracy_validated=False,
              validated_library_rows=41, full_cross_library_rows=146,
              retired_invalid_rows=16, distributed_tp8_rows=4,
              active_work=False, background_jobs=[], updated_at_utc=provenance['time_utc'])
(R / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
print(report)
